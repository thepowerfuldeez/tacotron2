import argparse
import json
import numpy as np
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from collections import OrderedDict

try:
    from apex import amp
except ImportError:
    pass
import nltk

nltk.download('punkt')

from torch.utils.tensorboard import SummaryWriter

from model import Model
from data import TTSDataset, TTSCollate
from distributed import setup_distributed, apply_gradient_allreduce, cleanup_distributed
from util import reduce_tensor, show_figure, success_rate

import sys
sys.path.append("waveglow")
from denoiser import Denoiser


def setup_model(
        distributed: bool,
        rank: int,
        world_size: int,
        group_name: str,
        checkpoint: str,
        learning_rate: float,
        fp16: bool,
        sync_bn: bool
):
    """Create model, cast to fp16 if needed, load checkpoint, apply DDP and sync_bn if needed"""
    torch.cuda.set_device(rank)
    model = Model(fp16=fp16)
    model = model.to(rank)  # move model to appropriate gpu when using distributed training
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # order is important: cast to fp16 first, load fp16 checkpoint (with amp weights), apply DDP, apply sync_bn
    # as stated here: https://github.com/NVIDIA/apex/tree/master/examples/imagenet
    if fp16:
        # Initialization with apex
        opt_level = 'O2'
        model, opt = amp.initialize(model, opt, opt_level=opt_level)

    iteration = 0
    best_metric = 1e3
    if checkpoint:
        iteration, best_metric = load_checkpoint(checkpoint, model, opt, fp16, rank)
        print(f"resuming from {iteration} iteration")

    if distributed:
        # set default .to('cuda') behavior to current local rank
        setup_distributed(rank, world_size, group_name)
        model = apply_gradient_allreduce(model)
        # run with python -m torch.distributed.launch --nproc_per_node=NUM_GPUS train.py

    if sync_bn:
        from apex.parallel import convert_syncbn_model

        if rank == 0:
            print("using apex synced BN")
        model = convert_syncbn_model(model)
    return model, opt, iteration, best_metric


def train(model, train_loader, opt, writer, rank=0, iteration=0, log_every=100, fp16=False, distributed=False):
    """Train loop for one epoch"""
    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases.
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    np.random.seed(iteration % 42)
    # randomly choose index of loader to send images and attention to tensorboard (send only one per epoch)
    random_idx = np.random.choice(len(train_loader), 1)
    model.train()
    for i, (text, input_lengths, mel, stop_target) in enumerate(train_loader):
        text = text.to("cuda", non_blocking=True)
        input_lengths = input_lengths.to("cuda", non_blocking=True)
        mel = mel.to("cuda", non_blocking=True)
        stop_target = stop_target.to("cuda", non_blocking=True)
        iteration += 1

        start = time.time()
        opt.zero_grad()
        mel_pred, mel_pred_postnet, stop_predictions, alignment = model(text, input_lengths, mel)
        mel_loss = F.mse_loss(mel_pred, mel)
        mel_postnet_loss = F.mse_loss(mel_pred_postnet, mel)
        stop_loss = F.binary_cross_entropy(stop_predictions, stop_target)

        loss = mel_loss + mel_postnet_loss + stop_loss

        if distributed:
            n_gpus = torch.cuda.device_count()
            loss_value = reduce_tensor(loss.data, n_gpus).item()
        else:
            loss_value = loss.item()
        if rank == 0:
            writer.add_scalar('loss/train', loss_value, iteration)
            writer.add_scalar('loss/mel', mel_loss.item(), iteration)
            writer.add_scalar('loss/mel_postnet', mel_postnet_loss.item(), iteration)
            writer.add_scalar('loss/stop', stop_loss.item(), iteration)
            writer.add_scalar('success_rate/train', success_rate(alignment), iteration)

            if i == random_idx:
                writer.add_image("mel_pred/train", 
                                  show_figure(mel_pred_postnet[0].float().detach().cpu().numpy()), 
                                  iteration,
                                  dataformats='HWC')
                writer.add_image("alignment/train", 
                                  show_figure(alignment[0].float().detach().cpu().numpy(), origin='lower'), 
                                  iteration,
                                  dataformats='HWC')

        if fp16:
            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if fp16:
            grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(opt), 1.0)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if rank == 0 and not np.isnan(grad_norm):
            writer.add_scalar("grad_norm", grad_norm, iteration)

        opt.step()
        end = time.time()

        # print only if it's first device (rank == 0)
        if iteration % log_every == 0 and rank == 0:
            print(f"train {iteration}, grad_norm={grad_norm:.3f}, mel_loss.item()={mel_loss.item():.3f}, "
                  f"mel_postnet_loss.item()={mel_postnet_loss.item():.3f}, stop_loss.item()={stop_loss.item():.3f}, "
                  f"{end - start:.2f} s.")
    return iteration


def validate(
        model: Model,
        val_loader: DataLoader,
        writer: SummaryWriter,
        iteration: int,
        waveglow_nvidia_repo_dir: Path = None,
        waveglow_path: Path = None,
        fp16: bool = False
):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(iteration % 42)
    metric_values, success_rates = [], []
    model.eval()
    start = time.time()

    # choose random mel-spectogram and synthesize audio from it
    random_idx = np.random.choice(len(val_loader), 1)
    for i, (text, input_lengths, mel, stop_target) in enumerate(val_loader):
        text = text.to("cuda", non_blocking=True)
        input_lengths = input_lengths.to("cuda", non_blocking=True)
        mel = mel.to("cuda", non_blocking=True)
        stop_target = stop_target.to("cuda", non_blocking=True)
        with torch.no_grad():
            mel_pred, mel_pred_postnet, stop_predictions, alignment = model(text, input_lengths, mel)
            mel_loss = F.mse_loss(mel_pred, mel)
            mel_postnet_loss = F.mse_loss(mel_pred_postnet, mel)
            stop_loss = F.binary_cross_entropy(stop_predictions, stop_target)

            loss = mel_loss + mel_postnet_loss + stop_loss
            loss_value = loss.item()
            metric_values.append(loss_value)
            success_rates.append(success_rate(alignment))
            if i == random_idx:
                mel_to_gen = mel_pred_postnet[0]
                gen_alignment = alignment[0]
    avg_loss = np.mean(metric_values)
    avg_success_rate = np.mean(success_rates)
    writer.add_scalar('loss/validation', avg_loss, iteration)
    writer.add_scalar('success_rate/validation', avg_success_rate, iteration)
    writer.add_image("mel_pred/validation", 
                      show_figure(mel_to_gen.float().cpu().numpy()), 
                      iteration,
                      dataformats='HWC')
    writer.add_image("alignment/validation", 
                      show_figure(gen_alignment.float().cpu().numpy(), origin='lower'), 
                      iteration,
                      dataformats='HWC')
    if waveglow_nvidia_repo_dir and waveglow_path:
        print('start to audio synthesis')
        writer.add_audio("audio/validation",
                         waveglow_gen(waveglow_nvidia_repo_dir, waveglow_path, mel_to_gen[None], fp16=fp16), iteration, sample_rate=22050)
    end = time.time()
    print(f"validation {iteration}, loss={loss_value:.3f}, {end - start:.2f} s.")
    return avg_loss


def inference(model, sample_texts, writer, iteration, waveglow_nvidia_repo_dir=None, waveglow_path=None, fp16=False):
    """Perform inference on test set of only texts (no target mel provided)"""
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    model.eval()
    start = time.time()
    with torch.no_grad():
        mel_postnet, alignment = model.inference(sample_texts)
        
    random_idx = np.random.randint(mel_postnet.size(0))
    writer.add_scalar("success_rate/inference", success_rate(alignment), iteration)
    writer.add_image("mel_pred/inference", 
                      show_figure(mel_postnet[random_idx].float().cpu().numpy()), 
                      iteration,
                      dataformats='HWC')
    writer.add_image("alignment/inference", 
                      show_figure(alignment[random_idx].float().cpu().numpy(), origin='lower'), 
                      iteration,
                      dataformats='HWC')
    if waveglow_nvidia_repo_dir and waveglow_path:
        # generate audio of 1 random predicted mel
        audio = waveglow_gen(waveglow_nvidia_repo_dir, waveglow_path, mel_postnet[[random_idx]], fp16=fp16)
        print("gen success, audio shape", audio.shape, "audio min/max", audio.min(), audio.max())
        writer.add_audio("audio/inference", audio, iteration, sample_rate=22050)
        writer.add_text("text/inference", sample_texts[random_idx])
    end = time.time()
    print(f"inference took {end-start:.2f} s.")


def waveglow_gen(waveglow_nvidia_repo_dir, waveglow_path, mel, sigma=0.666, denoiser_strength=0.1, fp16=False):
    """Generate audio with waveglow from checkpoint"""
    torch.cuda.empty_cache()

    waveglow = torch.load(waveglow_path)['model']
    waveglow.cuda().eval()
    if fp16:
        waveglow = waveglow.half()
        mel = mel.half()
    for k in waveglow.convinv:
        k.float()
    denoiser = Denoiser(waveglow)
    with torch.no_grad():
        audio = denoiser(waveglow.infer(mel, sigma), denoiser_strength)
    del waveglow, denoiser
    torch.cuda.empty_cache()
    return audio.cpu().view(1, -1)


def save_checkpoint(model, opt, iteration, best_metric, out_path, fp16):
    checkpoint = {
        "iteration": iteration,
        "best_metric": best_metric,
        "state_dict": model.state_dict(),
        "opt": opt.state_dict()
    }
    if fp16:
        checkpoint.update({"amp": amp.state_dict()})
    torch.save(checkpoint, out_path)


def load_checkpoint(checkpoint_path, model, opt, fp16, rank):
    """Load amp, model and optimizer states"""
    # we should load checkpoint on appropriate gpu device, derived from local_rank, use lambda (example from
    # https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py )
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda(rank))
    state_dict = checkpoint['state_dict']
    if "module." in list(state_dict.keys())[0]:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k[len("module."):]] = v
        state_dict = new_state_dict
    model.load_state_dict(state_dict)
    opt.load_state_dict(checkpoint['opt'])
    if fp16 and 'amp' in checkpoint:
        amp.load_state_dict(checkpoint['amp'])
    return checkpoint['iteration'], checkpoint['best_metric']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--args", default=None, type=Path, help="Load arguments from file")
    p.add_argument("--exp-name", default="default", help="Experiment name")
    p.add_argument("--ds-path", type=Path, help="Dataset metadata path")
    p.add_argument("--validation-size", default=200, type=int, help="Dataset validation split size (seed is fixed)")
    p.add_argument("--checkpoints-path", type=Path)
    p.add_argument("--checkpoint", default=None)

    p.add_argument("--learning-rate", type=float, default=3e-4, help="learning rate for optimizer")
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--num-epochs", type=int, default=1000)
    p.add_argument("--log-every", help="log every (iterations)", type=int, default=100)
    p.add_argument("--save-every", help="save every (epochs)", type=int, default=5)

    p.add_argument("--fp16", action="store_true")
    p.add_argument("--distributed", help="DDP training", action="store_true")
    p.add_argument("--group-name", help="Distributed group name")
    p.add_argument("--rank", help="for DDP, default device", default=0, type=int)
    p.add_argument("--sync-bn", help="synchronized batchnorm for distributed", action="store_true")

    p.add_argument("--infer-every", help="inference on test phrases every (epochs)", default=5, type=int)
    p.add_argument("--waveglow_nvidia_repo_dir", help="repo for waveglow if infer", default=None, type=Path)
    p.add_argument("--waveglow_path", help="path to waveglow checkpoint", default=None, type=Path)
    return p.parse_args()


def main():
    args = parse_args()
    # load arguments from file
    if args.args is not None:
        args_dict = json.load(args.args.open())
        serialized_keys = args_dict['serialized_keys']
        for k, v in args_dict.items():
            if k in serialized_keys:
                args.__dict__[k] = Path(v)
            else:
                args.__dict__[k] = v

    exp_name = args.exp_name
    checkpoint = args.checkpoint
    ds_path = args.ds_path
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    log_every = args.log_every
    save_every = args.save_every
    fp16 = args.fp16
    rank = args.rank
    distributed = args.distributed
    group_name = args.group_name
    infer_every = args.infer_every
    waveglow_nvidia_repo_dir = args.waveglow_nvidia_repo_dir
    waveglow_path = args.waveglow_path
    
    print("rank", rank, "parsed logs")

    torch.backends.cudnn.enabled = True
    exp_dir = Path("experiments").joinpath(exp_name)
    exp_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(str(exp_dir), flush_secs=30)
    with exp_dir.joinpath('commandline_args.txt').open('w') as f:
        args_for_serialize = {}
        serialized_keys = []
        for k, v in args.__dict__.items():
            if isinstance(v, Path):
                args_for_serialize[k] = str(v)
                serialized_keys.append(k)
            else:
                args_for_serialize[k] = v
        args_for_serialize['serialized_keys'] = serialized_keys
        json.dump(args_for_serialize, f, indent=2)

    world_size = torch.cuda.device_count() * 1  # as we have only 1 node
    model, opt, iteration, best_metric = setup_model(distributed, rank, world_size, group_name, checkpoint, args.learning_rate,
                                                     fp16, args.sync_bn)
    print("rank", rank, "setup model")

    train_ds = TTSDataset(ds_path, ds_type="train", validation_size=args.validation_size)
    validation_ds = TTSDataset(ds_path, ds_type="validation", validation_size=args.validation_size)

    sampler = None
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(train_ds, num_replicas=world_size, rank=rank)

    data_collate = TTSCollate()
    train_loader = DataLoader(train_ds, batch_size=batch_size, collate_fn=data_collate, sampler=sampler)
    val_loader = DataLoader(validation_ds, batch_size=batch_size // 4, collate_fn=data_collate)
    print("rank", rank, "dataset created")
    for epoch in range(1, num_epochs + 1):
        if rank == 0:
            print("epoch", epoch)
        iteration = train(model, train_loader, opt, writer, rank,
                          iteration=iteration, log_every=log_every, fp16=fp16, distributed=distributed)
        if rank == 0:
            try:
                avg_metric = validate(model, val_loader, writer, iteration,
                                      waveglow_nvidia_repo_dir, waveglow_path, fp16)
                if epoch % infer_every:
                    inference(model, Path("sample_phrases.txt").read_text().split("\n"),
                              writer, iteration, waveglow_nvidia_repo_dir, waveglow_path, fp16)
                if avg_metric < best_metric:
                    # TODO: save last n checkpoints in terms of target metric
                    best_metric = avg_metric
                    save_checkpoint(model, opt, iteration, best_metric, exp_dir.joinpath("model_best.pth"), fp16)
                if epoch % save_every == 0:
                    save_checkpoint(model, opt, iteration, best_metric, exp_dir.joinpath("model_last.pth"), fp16)
            except Exception as e:
                print("exception occured", e)
                save_checkpoint(model, opt, iteration, best_metric, exp_dir.joinpath("model_fail.pth"), fp16)
    cleanup_distributed()


if __name__ == "__main__":
    main()
