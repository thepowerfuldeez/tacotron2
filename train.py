import argparse
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from collections import OrderedDict
import nltk
nltk.download('punkt')

from model import Model
from data import TTSDataset, TTSCollate


def train(model, loader, opt, local_rank=0, iteration=0, log_every=100, fp16=False):
    """Train loop"""
    metric_values = []
    for text, input_lengths, mel, stop_target in loader:
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
        metric_values.append(loss.item())
        end = time.time()
        
        # print only if it's first device (local_rank == 0)
        if iteration % log_every == 0 and local_rank == 0:
            print(f"{iteration}, mel_loss.item()={mel_loss.item():.2f}, mel_postnet_loss.item()={mel_postnet_loss.item():.2f}, stop_loss.item()={stop_loss.item():.3f}, {end-start:.2f} s.")
        if fp16:
            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        opt.step()
    return iteration, np.mean(metric_values)


def test_alignment(batch, model):
    """For use in jupyter notebook or for tensorboard logging"""
    text, input_lengths, mel, stop_target = TTSCollate()(batch)
    with torch.no_grad():
        text = text.to("cuda", non_blocking=True)
        input_lengths = input_lengths.to("cuda", non_blocking=True)
        mel = mel.to("cuda", non_blocking=True)
        mel_pred, mel_pred_postnet, stop_predictions, alignment = model(text, input_lengths, mel)
    plt.imshow(alignment.cpu().numpy()[0][::-1], aspect='auto')
    plt.show()


def save_checkpoint(model, iteration, best_metric, out_path, fp16):
    checkpoint = {
        "iteration": iteration,
        "best_metric": best_metric,
        "state_dict": model.state_dict()
    }
    if fp16:
        checkpoint.update({"amp": amp.state_dict()})
    torch.save(checkpoint, out_path)


def load_checkpoint(checkpoint_path, model, fp16, local_rank):
    # we should load checkpoint on appropriate gpu device, derived from local_rank, use lambda (example from
    # https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py )
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda(local_rank))
    state_dict = checkpoint['state_dict'] 
    if "module." in list(state_dict.keys())[0]:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k[len("module."):]] = v
        state_dict = new_state_dict
    model.load_state_dict(state_dict)
    if fp16 and 'amp' in checkpoint:
        amp.load_state_dict(checkpoint['amp'])
    return checkpoint['iteration'], checkpoint['best_metric']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ds-path", type=Path, help="Dataset metadata path")
    p.add_argument("--checkpoints-path", type=Path)
    p.add_argument("--checkpoint", default=None)

    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--num-epochs", type=int, default=1000)
    p.add_argument("--log-every", help="log every (iterations)", type=int, default=100)
    p.add_argument("--save-every", help="save every (epochs)", type=int, default=5)

    p.add_argument("--fp16", action="store_true")
    p.add_argument("--distributed", help="DDP training", action="store_true")
    p.add_argument("--local_rank", help="for DDP, default device", default=0, type=int)
    p.add_argument("--sync-bn", help="synchronized batchnorm for distributed", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    checkpoints_path = args.checkpoints_path
    checkpoint = args.checkpoint
    ds_path = args.ds_path
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    log_every = args.log_every
    save_every = args.save_every
    fp16 = args.fp16
    local_rank = args.local_rank
    
    ds = TTSDataset(ds_path)

    model = Model(fp16=fp16)
    model = model.to(local_rank)  # move model to appropriate gpu when using distributed training
    opt = torch.optim.Adam(model.parameters())
    
    # order is important: cast to fp16 first, load fp16 checkpoint (with amp weights), apply DDP, apply sync_bn
    # as stated here: https://github.com/NVIDIA/apex/tree/master/examples/imagenet
    if fp16:
        # Initialization with apex
        from apex import amp

        opt_level = 'O2'
        model, opt = amp.initialize(model, opt, opt_level=opt_level)
    
    iteration = 0
    best_metric = 1e3
    if checkpoint:
        iteration, best_metric = load_checkpoint(checkpoint, model, fp16, local_rank)
        print(f"resuming from {iteration} iteration")

    sampler = None
    if args.distributed:
        from apex.parallel import DistributedDataParallel
        
        # set default .to('cuda') behavior to current local_rank
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        model = DistributedDataParallel(model, delay_allreduce=True)
        # run with python -m torch.distributed.launch --nproc_per_node=NUM_GPUS train.py

        sampler = torch.utils.data.distributed.DistributedSampler(ds)

    if args.sync_bn:
        from apex.parallel import convert_syncbn_model
        
        if local_rank == 0:
            print("using apex synced BN")
        model = convert_syncbn_model(model)

    loader = DataLoader(ds, num_workers=4, batch_size=batch_size, collate_fn=TTSCollate(), sampler=sampler)
    for epoch in range(1, num_epochs + 1):
        if local_rank == 0:
            print("epoch", epoch)
        iteration, avg_metric = train(model, loader, opt, local_rank, iteration=iteration, log_every=log_every, fp16=fp16)
        # for jupyter notebook
        # if iteration % 25 == 0:
        # clear_output(True)
        # test_alignment([ds[0]], model)
        if local_rank == 0:
            if avg_metric < best_metric:
                best_metric = avg_metric
                save_checkpoint(model, iteration, best_metric, checkpoints_path.joinpath("model_best.pth"), fp16)
            if epoch % 10 == 0:
                save_checkpoint(model, iteration, best_metric, checkpoints_path.joinpath("model_last.pth"), fp16)
