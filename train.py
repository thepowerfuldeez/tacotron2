import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from collections import OrderedDict
try:
    from apex import amp
except ImportError: pass
import nltk
nltk.download('punkt')

from torch.utils.tensorboard import SummaryWriter

from model import Model
from data import TTSDataset, TTSCollate
from distributed import setup_distributed, apply_gradient_allreduce, cleanup_distributed


def reduce_tensor(tensor, n_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= n_gpus
    return rt
    

def setup_model(distributed, rank, world_size, group_name, checkpoint, fp16, sync_bn):
    """Create model, cast to fp16 if needed, load checkpoint, apply DDP and sync_bn if needed"""
    torch.cuda.set_device(rank)
    model = Model(fp16=fp16)
    model = model.to(rank)  # move model to appropriate gpu when using distributed training
    opt = torch.optim.Adam(model.parameters())
    
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


def success_rate(alignments):
    """Success rate of attention batch.
    Check if all attentions are monotonic and non flat"""
    als_values_batch = alignments.detach().cpu().max(1)[1]
    success_count = 0

    for als_values in als_values_batch:
        differences = als_values[1:] - als_values[:-1]
        attn_is_monotonic = int(len(als_values) * 0.8) < (differences < 3).sum()
        attn_not_flat = differences[differences.abs() < 3].sum() > int(len(als_values) * 0.1)
        if bool(attn_is_monotonic & attn_not_flat):
            success_count += 1
    return success_count / alignments.size(0)


def train(model, loader, opt, writer, rank=0, iteration=0, log_every=100, fp16=False, distributed=False):
    """Train loop for one epoch"""
    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases.
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
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
        
        if distributed:
            n_gpus = torch.cude.device_count()
            loss_value = reduce_tensor(loss.data, n_gpus).item()
        else:
            loss_value = loss.item()
        metric_values.append(loss_value)
        writer.add_scalar('loss/train', loss_value, iteration)
        writer.add_scalar('loss/mel', mel_loss.item(), iteration)
        writer.add_scalar('loss/mel_postnet', mel_postnet_loss.item(), iteration)
        writer.add_scalar('loss/stop', stop_loss.item(), iteration)
        writer.add_scalar('success_rate/train', success_rate(alignment), iteration)
        
        if fp16:
            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()            
        
        if fp16:
            grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(opt), 1.0)
            is_overflow = np.isnan(grad_norm)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        opt.step()
        end = time.time()
        
        # print only if it's first device (rank == 0)
        if iteration % log_every == 0 and rank == 0:
            print(f"{iteration}, grad_norm={grad_norm:.3f}, mel_loss.item()={mel_loss.item():.3f}, mel_postnet_loss.item()={mel_postnet_loss.item():.3f}, stop_loss.item()={stop_loss.item():.3f}, {end-start:.2f} s.")
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
    p.add_argument("--ds-path", type=Path, help="Dataset metadata path")
    p.add_argument("--checkpoints-path", type=Path)
    p.add_argument("--checkpoint", default=None)

    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--num-epochs", type=int, default=1000)
    p.add_argument("--log-every", help="log every (iterations)", type=int, default=100)
    p.add_argument("--save-every", help="save every (epochs)", type=int, default=5)

    p.add_argument("--fp16", action="store_true")
    p.add_argument("--distributed", help="DDP training", action="store_true")
    p.add_argument("--group-name", help="Distributed group name")
    p.add_argument("--rank", help="for DDP, default device", default=0, type=int)
    p.add_argument("--sync-bn", help="synchronized batchnorm for distributed", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    # load arguments from file
    if args.args is not None:
        args.__dict__ = json.load(args.args.open())

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
    
    torch.backends.cudnn.enabled = True
    exp_dir = Path("experiments").joinpath(exp_name)
    exp_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(str(exp_dir), flush_secs=30)
    with exp_dir.joinpath('commandline_args.txt').open('w') as f:
        json.dump(args.__dict__, f, indent=2)

    world_size = torch.cuda.device_count() * 1  # as we have only 1 node
    model, opt, iteration, best_metric = setup_model(distributed, rank, world_size, group_name, checkpoint, fp16, args.sync_bn)
    
    ds = TTSDataset(ds_path)
    sampler = None
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(ds, num_replicas=world_size, rank=rank)

    loader = DataLoader(ds, batch_size=batch_size, collate_fn=TTSCollate(), sampler=sampler)
    for epoch in range(1, num_epochs + 1):
        if rank == 0:
            print("epoch", epoch)
        iteration, avg_metric = train(model, loader, opt, writer, rank,
                                      iteration=iteration, log_every=log_every, fp16=fp16, distributed=distributed)
        # for jupyter notebook
        # if iteration % 25 == 0:
        # clear_output(True)
        # test_alignment([ds[0]], model)
        if rank == 0:
            if avg_metric < best_metric:
                best_metric = avg_metric
                save_checkpoint(model, opt, iteration, best_metric, exp_dir.joinpath("model_best.pth"), fp16)
            if epoch % save_every == 0:
                save_checkpoint(model, opt, iteration, best_metric, exp_dir.joinpath("model_last.pth"), fp16)
    cleanup_distributed()


if __name__ == "__main__":
    main()
