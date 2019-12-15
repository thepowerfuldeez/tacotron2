import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

from model import Model
from data import TTSDataset, TTSCollate


def train(model, loader, opt, device="cuda", iteration=0, log_every=100):
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
            print(f"{iteration}, {mel_loss.item()=:.2f}, {mel_postnet_loss.item()=:.2f}, {stop_loss.item()=:.3f}")
        loss = mel_loss + mel_postnet_loss + stop_loss
        min_metric = min(min_metric, loss.item())
        loss.backward()
        opt.step()
    return iteration, min_metric


def test_alignment(batch, model):
    """For use in jupyter notebook or for tensorboard logging"""
    text, input_lengths, mel, stop_target = TTSCollate()(batch)
    with torch.no_grad():
        text = text.to(device)
        input_lengths = input_lengths.to(device)
        mel = mel.to(device)
        mel_pred, mel_pred_postnet, stop_predictions, alignment = model(text, input_lengths, mel)
    plt.imshow(alignment.cpu().numpy()[0][::-1], aspect='auto')
    plt.show()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ds-path", type=Path, help="Dataset metadata path")
    p.add_argument("--checkpoints-path", type=Path)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--num-epochs", type=int, default=1000)
    p.add_argument("--save-every", help="save every (epochs)", type=int, default=10)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ds = TTSDataset(args.ds_path)
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=TTSCollate())
    device = args.device
    checkpoints_path = args.checkpoints_path
    checkpoint = args.checkpoint

    model = Model()
    model = model.to(device)
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location=device))
    opt = torch.optim.Adam(model.parameters())
    iteration = 0
    best_metric = 1e3
    for epoch in range(1, args.num_epochs + 1):
        print("epoch", epoch)
        iteration, min_metric = train(model, loader, opt, device, iteration=iteration)
        # for jupyter notebook
        # if iteration % 25 == 0:
            # clear_output(True)
            # test_alignment([ds[0]], model)
        if min_metric < best_metric:
            best_metric = min_metric
            torch.save(model.state_dict(), checkpoints_path.joinpath("model_best.pth"))
        if epoch % 10 == 0:
            torch.save(model.state_dict(), checkpoints_path.joinpath("model_last.pth"))