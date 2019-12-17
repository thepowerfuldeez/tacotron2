import torch
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
from torch.utils.data import Dataset

from text import encode_text


class TTSDataset(Dataset):
    def __init__(
            self,
            data_path: Path,
            ds_type: str = "train",
            validation_size: int = 200,
            seed: int = 42
    ):
        np.random.seed(seed)
        self.path = data_path.parent
        self.metadata = [line.split("|") for line in data_path.read_text().split("\n")[:-1]]
        assert len(self.metadata) > validation_size
        self.metadata = np.array([line for line in self.metadata
                                  if self.path.joinpath("mels").joinpath(line[0]).with_suffix(".npy").exists()])

        # apply proper indices
        validation_indices = np.random.choice(len(self.metadata), validation_size)
        if ds_type == "train":
            self.metadata = self.metadata[~np.in1d(np.arange(len(self.metadata)), validation_indices)]
        elif ds_type == "validation":
            self.metadata = self.metadata[np.in1d(np.arange(len(self.metadata)), validation_indices)]

        self.processed_texts = []
        for line in tqdm(self.metadata):
            self.processed_texts.append(encode_text(line[1]))
        self.processed_texts = np.array(self.processed_texts)

        self.input_lengths = np.array([len(it) for it in self.processed_texts])
        sorted_idx = np.argsort(self.input_lengths)
        self.input_lengths = self.input_lengths[sorted_idx]
        self.processed_texts = self.processed_texts[sorted_idx]
        self.metadata = self.metadata[sorted_idx]

    def __getitem__(self, idx):
        file_id, text, *_ = self.metadata[idx]
        mel = np.load(self.path.joinpath("mels").joinpath(file_id).with_suffix(".npy"))
        text_encoded = self.processed_texts[idx]
        return torch.from_numpy(text_encoded).long(), torch.from_numpy(mel)

    def __len__(self):
        return len(self.metadata)


class TTSCollate:
    def __call__(self, batch):
        input_lengths = torch.tensor(sorted([len(x[0]) for x in batch], reverse=True))
        mel_channels = batch[0][1].size(0)
        max_mel_len = max(x[1].size(-1) for x in batch)
        text_padded = torch.zeros(len(batch), input_lengths[0]).long()
        mel_padded = torch.zeros(len(batch), mel_channels, max_mel_len).float()
        stop_target = torch.ones(len(batch), max_mel_len).float()
        for i, (text, mel) in enumerate(batch):
            text_padded[i, :text.size(-1)] = text
            mel_padded[i, :, :mel.size(-1)] = mel
            stop_target[i, :mel.size(-1)] = 0
        return text_padded, input_lengths, mel_padded, stop_target.unsqueeze(-1)

