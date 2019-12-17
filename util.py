import matplotlib.pyplot as plt
import torch.distributed as dist
import torch
import numpy as np


def reduce_tensor(
        tensor: torch.Tensor,
        n_gpus: int
):
    """Reduces torch tensor across multiple gpus in distributed training"""
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= n_gpus
    return rt


def success_rate(alignments: torch.Tensor):
    """Success rate metric of attention batch.
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


def show_figure(
        image: np.array,
        **kwargs
):
    """Renders matplotlib figure for use in tensorboard.
    Keep in mind that for attention you should use origin=lower keyword arg
    from lower left angle to upper right"""
    fig = plt.figure()
    fig.imshow(image, aspect='auto', **kwargs)
    return fig

