import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import nltk
nltk.download('punkt')

from model import Model
from data import TTSDataset, TTSCollate


def train(model, loader, opt, device="cuda", iteration=0, log_every=100, fp16=False):
    """Train loop"""
    min_metric = 1e3
    for text, input_lengths, mel, stop_target in loader:
        text = text.to(device)
        input_lengths = input_lengths.to(device)
        mel = mel.to(device)
        stop_target = stop_target.to(device)
        iteration += 1

        opt.zero_grad()
        mel_pred, mel_pred_postnet, stop_predictions, alignment = model(text, input_lengths, mel)
        mel_loss = F.mse_loss(mel_pred, mel)
        mel_postnet_loss = F.mse_loss(mel_pred_postnet, mel)
        stop_loss = F.binary_cross_entropy(stop_predictions, stop_target)
        if iteration % log_every == 0:
            print(f"{iteration}, mel_loss.item()={mel_loss.item():.2f}, mel_postnet_loss.item()={mel_postnet_loss.item():.2f}, stop_loss.item()={stop_loss.item():.3f}")
        loss = mel_loss + mel_postnet_loss + stop_loss
        min_metric = min(min_metric, loss.item())
        if fp16:
            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        opt.step()
    return iteration, min_metric


def test_alignment(batch, model, device):
    """For use in jupyter notebook or for tensorboard logging"""
    text, input_lengths, mel, stop_target = TTSCollate()(batch)
    with torch.no_grad():
        text = text.to(device)
        input_lengths = input_lengths.to(device)
        mel = mel.to(device)
        mel_pred, mel_pred_postnet, stop_predictions, alignment = model(text, input_lengths, mel)
    plt.imshow(alignment.cpu().numpy()[0][::-1], aspect='auto')
    plt.show()


def save_checkpoint(model, iteration, out_path, fp16):
    checkpoint = {
        "iteration": iteration,
        "state_dict": model.state_dict(),
    }
    if fp16:
        checkpoint.update({"amp": amp.state_dict()})
    torch.save(checkpoint, out_path)


def load_checkpoint(checkpoint_path, model, fp16):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    if fp16 and 'amp' in checkpoint:
        amp.load_state_dict(checkpoint['amp'])
    return checkpoint['iteration']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ds-path", type=Path, help="Dataset metadata path")
    p.add_argument("--checkpoints-path", type=Path)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device", default="cuda")

    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--num-epochs", type=int, default=1000)
    p.add_argument("--log-every", help="log every (iterations)", type=int, default=100)
    p.add_argument("--save-every", help="save every (epochs)", type=int, default=5)

    p.add_argument("--fp16", action="store_true")
    p.add_argument("--distributed", help="DDP training", action="store_true")
    p.add_argument("--local_rank", help="for DDP", default=0, type=int)
    p.add_argument("--sync-bn", help="synchronized batchnorm for distributed", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = args.device
    checkpoints_path = args.checkpoints_path
    checkpoint = args.checkpoint
    ds_path = args.ds_path
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    log_every = args.log_every
    save_every = args.save_every
    fp16 = args.fp16

    model = Model(fp16=fp16)
    model = model.to(device)
    iteration = 0
    if checkpoint:
        iteration = load_checkpoint(checkpoint, model, fp16)
        print(f"resuming from {iteration} iteration")
    opt = torch.optim.Adam(model.parameters())
    if fp16:
        # Initialization with apex
        from apex import amp

        opt_level = 'O2'
        model, opt = amp.initialize(model, opt, opt_level=opt_level)

    sampler = None
    if args.distributed:
        from apex.parallel import DistributedDataParallel

        model = DistributedDataParallel(model, delay_allreduce=True)
        # run with python -m torch.distributed.launch --nproc_per_node=NUM_GPUS train.py

        sampler = torch.utils.data.distributed.DistributedSampler(ds)

    if args.sync_bn:
        from apex.parallel import convert_syncbn_model

        print("using apex synced BN")
        model = convert_syncbn_model(model)

    ds = TTSDataset(ds_path)
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=TTSCollate(), sampler=sampler)
    best_metric = 1e3
    for epoch in range(1, num_epochs + 1):
        print("epoch", epoch)
        iteration, min_metric = train(model, loader, opt, device, iteration=iteration, log_every=log_every, fp16=fp16)
        # for jupyter notebook
        # if iteration % 25 == 0:
        # clear_output(True)
        # test_alignment([ds[0]], model)
        if args.local_rank == 0:
            if min_metric < best_metric:
                best_metric = min_metric
                save_checkpoint(model, iteration, checkpoints_path.joinpath("model_best.pth"), fp16)
            if epoch % 10 == 0:
                save_checkpoint(model, iteration, checkpoints_path.joinpath("model_last.pth"), fp16)
