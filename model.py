import torch.nn as nn
import torch
import torch.nn.functional as F

p_dropout = 0.5
MAX_LENGTH = 1000


class Encoder(nn.Module):
    """Embedding - 3x 5x1 convs with same padding - BiLSTM"""

    def __init__(self, n_symbols, padding_idx, embedding_dim=512, rnn_size=512):
        super().__init__()
        n_conv_blocks = 3
        kernel_size = 5
        padding_size = (kernel_size - 1) // 2

        self.char_embedding = nn.Embedding(n_symbols, embedding_dim, padding_idx=padding_idx)
        conv_layers = []
        for _ in range(n_conv_blocks):
            conv_layers.extend([
                nn.Conv1d(embedding_dim, embedding_dim, kernel_size, padding=padding_size),
                nn.BatchNorm1d(embedding_dim),
                nn.ReLU(True),
                nn.Dropout(p_dropout)
            ])
        self.conv_layers = nn.Sequential(*conv_layers)
        self.rnn = nn.LSTM(embedding_dim, rnn_size // 2, bidirectional=True, batch_first=True)

    def forward(self, x, input_lengths):
        # retrieve embeddings for every input char
        embedding = self.char_embedding(x)
        # (b, n, emb_dim) -> (b, emb_dim, n)
        embedding = embedding.transpose(1, 2)
        # 3x (5, 1) convs with padding (b, emb_dim, n) -> (b, emb_dim, n)
        features = self.conv_layers(embedding)
        features = features.transpose(1, 2)
        features = nn.utils.rnn.pack_padded_sequence(features, input_lengths, batch_first=True, enforce_sorted=True)
        # when training on multiple gpu, you probably need to flatten params of rnn to ensure they are on the one
        # gpu after splitting
        # more: https://discuss.pytorch.org/t/why-do-we-need-flatten-parameters-when-using-rnn-with-dataparallel/46506/2
        # self.rnn.flatten_parameters()
        # compute rnn features
        features, _ = self.rnn(features)
        features = nn.utils.rnn.pad_packed_sequence(features, batch_first=True, total_length=input_lengths[0])[0]
        return features


class Attention(nn.Module):
    """Simple linear attention mechanism from http://arxiv.org/abs/1409.0473
    Input: (last_prenet_output, current_hidden_state, encoder_outputs)
    Output: context vector (weighted sum of encoder_outputs, b x 1 x encoder_rnn_size), attention_weights (b x seq_len)"""

    def __init__(self, prenet_size=256, hidden_size=1024, max_length=MAX_LENGTH):
        super().__init__()

        self.attention_linear = nn.Linear(prenet_size + hidden_size, max_length)

    def forward(self, decoder_input, hidden, encoder_outputs):
        energies = self.attention_linear(torch.cat((decoder_input, hidden), -1))
        energies = F.dropout(energies, p_dropout, self.training)
        mask = encoder_outputs.sum(-1) == 0
        attention_weights = F.softmax(
            energies[:, :encoder_outputs.size(1)].masked_fill(mask, float('-inf')), -1
        ).unsqueeze(1)
        attention_context = torch.bmm(attention_weights, encoder_outputs)
        return attention_context, attention_weights


class Prenet(nn.Module):
    """Prenet network that acts as information bottleneck,
    has dropout on both training and inference"""

    def __init__(self, input_dim, prenet_dim, n_layers=2):
        super().__init__()
        self.linears = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size, output_size),
                nn.ReLU(True)
            ) for input_size, output_size in zip(
                [input_dim] + [prenet_dim] * (n_layers - 1),
                [prenet_dim] * n_layers
            )
        ])

    def forward(self, x):
        for lin in self.linears:
            x = F.dropout(lin(x), p_dropout, training=True)
        return x


class Postnet(nn.Module):
    """Postnet for improving overall mel-spectrogram prediction.
    Decoder predicts mel auto-regressively frame-by-frame, so this
    network outputs residual to the whole prediction"""

    def __init__(self, postnet_dim=512, n_layers=5, kernel_size=5):
        super().__init__()

        padding_size = (kernel_size - 1) // 2
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_size, output_size, kernel_size, padding=padding_size),
                nn.BatchNorm1d(output_size),
                nn.Tanh(),
                nn.Dropout(p_dropout)
            )
            for input_size, output_size in zip([80] + [postnet_dim] * (n_layers - 2), [postnet_dim] * (n_layers - 1))
        ])
        self.convs.append(nn.Sequential(
            nn.Conv1d(postnet_dim, 80, kernel_size, padding=padding_size),
            nn.BatchNorm1d(80),
            nn.Dropout(p_dropout)
        ))

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x


class Decoder(nn.Module):
    """Auto-regressive decoder.
    Input: encoder_outputs, mel_target for teacher-forcing.

    Teacher-forcing: treat ground-truth spectrogram frames into rnn
    instead of last predicted frame to improve stability of training."""

    def __init__(self, prenet_dim=256, postnet_dim=512, encoder_rnn_size=512, hidden_size=1024, fp16=False):
        super().__init__()
        # as we simply weigh different parts of encoder outputs
        attention_context_size = encoder_rnn_size
        self.hidden_size = hidden_size
        self.fp16 = fp16

        self.attention = Attention()
        self.mel_channels = 80
        # instead of previous hidden state we will use last mel frame prediction
        self.prenet = Prenet(self.mel_channels, prenet_dim)

        self.decoder_n_layers = 2
        self.decoder_rnn = nn.LSTM(prenet_dim + attention_context_size, hidden_size,
                                   num_layers=self.decoder_n_layers, batch_first=True)

        self.linear_target = nn.Linear(hidden_size + attention_context_size, self.mel_channels)
        self.linear_stop_pred = nn.Linear(hidden_size + attention_context_size, 1)
        self.postnet = Postnet(postnet_dim)

    def init_hidden(self, batch_size, device="cpu", fp16=False):
        """Initialize hidden rnn state with tuple of zeros"""
#         return tuple(torch.zeros(self.decoder_n_layers, batch_size, self.hidden_size, device=device)
#                      for _ in range(self.decoder_n_layers))
        hidden = tuple(torch.zeros(self.decoder_n_layers, batch_size, self.hidden_size, device=device)
                       for _ in range(self.decoder_n_layers))
        if fp16:
            hidden = tuple([h.half() for h in hidden])
        return hidden
                     

    def init_mel_input(self, batch_size, device="cpu", fp16=False):
        """Initialize mel_input with zeros"""
#         return torch.zeros(batch_size, 1, self.mel_channels, device=device)
        mel_input = torch.zeros(batch_size, 1, self.mel_channels, device=device)
        if fp16:
            mel_input = mel_input.half()
        return mel_input

    def decode(self, mel_input, hidden, encoder_outputs):
        """Decode one mel_input frame to predict next frame using hidden rnn state
        Outputs next_frame, new_hidden_state, stop_predition, attention_weights"""
        B = encoder_outputs.size(0)

        if mel_input is None:
            mel_input = self.init_mel_input(B, device=encoder_outputs.device, fp16=self.fp16)
        if hidden is None:
            hidden = self.init_hidden(B, device=encoder_outputs.device, fp16=self.fp16)
        decoder_input = self.prenet(mel_input)
        attention_context, attention_weights = self.attention(decoder_input.squeeze(1), hidden[0][0], encoder_outputs)
        # when training on multiple gpu, you probably need to flatten params of rnn to ensure they are on the one
        # gpu after splitting
        # self.decoder_rnn.flatten_parameters()
        output, hidden = self.decoder_rnn(torch.cat((decoder_input, attention_context), -1), hidden)
        mel_frame = self.linear_target(torch.cat((output, attention_context), -1))
        stop_prediction = self.linear_stop_pred(torch.cat((output, attention_context), -1))
        return mel_frame, hidden, stop_prediction, attention_weights

    def forward(self, encoder_outputs, mel_target):
        """Auto-regressive decoding of sequence of encoder_outputs. Use teacher-forcing with target mel"""
        mel_frames, stop_preds, alignment = [], [], []
        mel_frame, hidden = None, None
        for i in range(mel_target.size(-1)):
            mel_pred, hidden, stop_prediction, attention_weights = self.decode(mel_frame, hidden, encoder_outputs)
            # currently teacher forcing is always on. TODO: prob of teacher forcing
            mel_frame = mel_target[:, :, i].unsqueeze(1)
            mel_frames.append(mel_pred)
            stop_preds.append(stop_prediction)
            alignment.append(attention_weights)
        mel = torch.cat(mel_frames, 1).transpose(1, 2)
        mel_postnet = mel + self.postnet(mel)
        stop_predictions = torch.cat(stop_preds, 1)
        stop_predictions = torch.sigmoid(stop_predictions)
        alignment = torch.cat(alignment, 1).transpose(1, 2)
        return mel, mel_postnet, stop_predictions, alignment

    def inference(self, encoder_outputs, stop_threshold=0.5):
        """Auto-regressive inference (with no target mel)"""
        mel_frames, alignment = [], []
        mel_frame, hidden = None, None
        i = 0
        while i < MAX_LENGTH:
            mel_frame, hidden, stop_prediction, attention_weights = self.decode(mel_frame, hidden, encoder_outputs)
            mel_frames.append(mel_frame)
            alignment.append(attention_weights)
            if stop_prediction[0].item() > stop_threshold:
                # predicted end of inference
                break
            i += 1
        mel = torch.cat(mel_frames, 1).transpose(1, 2)
        mel_postnet = mel + self.postnet(mel)
        alignment = torch.cat(alignment, 1).transpose(1, 2)
        return mel_postnet, alignment


class Model(nn.Module):
    """Tacotron2 model from https://arxiv.org/abs/1712.05884 (except Attention part for now)"""

    def __init__(self, n_symbols=100, pad_idx=0, fp16=False):
        super().__init__()

        self.encoder = Encoder(n_symbols, pad_idx)
        self.decoder = Decoder(fp16=fp16)

    def forward(self, x, input_lengths, mel_target):
        encoder_outputs = self.encoder(x, input_lengths)
        mel, mel_postnet, stop_predictions, alignment = self.decoder(encoder_outputs, mel_target)
        return mel, mel_postnet, stop_predictions, alignment

    def inference(self, x):
        with torch.no_grad():
            encoder_outputs = self.encoder(x, torch.tensor([len(x[0])], device=x.device))
            mel_postnet, alignment = self.decoder.inference(encoder_outputs)
        return mel_postnet, alignment