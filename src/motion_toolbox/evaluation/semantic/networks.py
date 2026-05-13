"""Neural network modules for the T2M text-motion co-embedding evaluator.

These modules implement the feature extraction networks used in the standard
HumanML3D/T2M evaluation protocol. They map motion sequences and text
descriptions into a shared 512-d embedding space for metric computation.

Architecture:
    Motion path:  raw_motion (263-d) -> MovementConvEncoder (512-d per step)
                  -> MotionEncoderBiGRUCo (512-d summary)
    Text path:    GloVe+POS (315-d) -> TextEncoderBiGRUCo (512-d summary)

The pre-trained weights are available from the HumanML3D evaluation checkpoints:
    - HumanML3D: t2m/text_mot_match/model/finest.tar
    - KIT-ML: kit/text_mot_match/model/finest.tar

References:
    - Guo et al., "Generating Diverse and Natural 3D Human Motions from Text" (CVPR 2022)
    - MoMask: https://github.com/EricGuo5513/momask-codes/blob/main/models/t2m_eval_modules.py
    - MDM: https://github.com/GuyTevet/motion-diffusion-model
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


def _init_weight(m: nn.Module) -> None:
    """Xavier normal initialization for Conv1d, Linear, and ConvTranspose1d layers."""
    if isinstance(m, (nn.Conv1d, nn.Linear, nn.ConvTranspose1d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class MovementConvEncoder(nn.Module):
    """1-D convolutional encoder that converts per-frame motion features to movement latents.

    Takes HumanML3D motion features (without the last 4 contact dims) and produces
    a temporally downsampled (4x) sequence of movement latent vectors.

    Architecture:
        Conv1d(4,2,1) + Dropout + LeakyReLU -> Conv1d(4,2,1) + Dropout + LeakyReLU -> Linear

    The temporal downsampling factor is 4 (two Conv1d layers with stride 2).

    Args:
        input_size: Dimension of input motion features (259 for HumanML3D, 247 for KIT).
        hidden_size: Hidden dimension of the convolutional layers (default 512).
        output_size: Output movement latent dimension (default 512).
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_size, output_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_net = nn.Linear(output_size, output_size)
        self.main.apply(_init_weight)
        self.out_net.apply(_init_weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode motion features to movement latents.

        Args:
            inputs: Motion features, shape (batch, seq_len, input_size).

        Returns:
            Movement latent vectors, shape (batch, seq_len // 4, output_size).
        """
        inputs = inputs.permute(0, 2, 1)  # (B, D, T)
        outputs = self.main(inputs).permute(0, 2, 1)  # (B, T//4, D)
        return self.out_net(outputs)


class TextEncoderBiGRUCo(nn.Module):
    """Bidirectional GRU text encoder for the T2M co-embedding space.

    Takes GloVe word embeddings and POS one-hot vectors, processes them through
    a bidirectional GRU, and projects to the shared embedding space.

    Args:
        word_size: Dimension of word embeddings (300 for GloVe).
        pos_size: Dimension of POS one-hot vectors (15 tags from spacy).
        hidden_size: GRU hidden dimension (default 512).
        output_size: Co-embedding output dimension (default 512).
        device: Device for parameter initialization.
    """

    def __init__(
        self,
        word_size: int,
        pos_size: int,
        hidden_size: int,
        output_size: int,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.device = device

        self.pos_emb = nn.Linear(pos_size, word_size)
        self.input_emb = nn.Linear(word_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size),
        )

        self.input_emb.apply(_init_weight)
        self.pos_emb.apply(_init_weight)
        self.output_net.apply(_init_weight)
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn((2, 1, self.hidden_size), requires_grad=True))

    def forward(
        self,
        word_embs: torch.Tensor,
        pos_onehot: torch.Tensor,
        cap_lens: list[int] | torch.Tensor,
    ) -> torch.Tensor:
        """Encode text descriptions to embedding vectors.

        Args:
            word_embs: GloVe word embeddings, shape (batch, max_text_len, 300).
            pos_onehot: POS one-hot vectors, shape (batch, max_text_len, 15).
            cap_lens: Caption lengths for each sample in the batch.

        Returns:
            Text embedding vectors, shape (batch, output_size).
        """
        num_samples = word_embs.shape[0]

        pos_embs = self.pos_emb(pos_onehot)
        inputs = word_embs + pos_embs
        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)

        if isinstance(cap_lens, torch.Tensor):
            cap_lens = cap_lens.data.tolist()

        emb = pack_padded_sequence(input_embs, cap_lens, batch_first=True, enforce_sorted=False)
        _gru_seq, gru_last = self.gru(emb, hidden)
        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)

        return self.output_net(gru_last)


class MotionEncoderBiGRUCo(nn.Module):
    """Bidirectional GRU motion encoder for the T2M co-embedding space.

    Takes movement latent vectors (from MovementConvEncoder) and produces
    a single motion embedding vector in the shared embedding space.

    Args:
        input_size: Dimension of movement latent vectors (default 512).
        hidden_size: GRU hidden dimension (default 1024).
        output_size: Co-embedding output dimension (default 512).
        device: Device for parameter initialization.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.device = device

        self.input_emb = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size),
        )

        self.input_emb.apply(_init_weight)
        self.output_net.apply(_init_weight)
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn((2, 1, self.hidden_size), requires_grad=True))

    def forward(
        self,
        inputs: torch.Tensor,
        m_lens: list[int] | torch.Tensor,
    ) -> torch.Tensor:
        """Encode movement latent sequence to a single motion embedding.

        Args:
            inputs: Movement latent vectors from MovementConvEncoder,
                shape (batch, seq_len, input_size).
            m_lens: Sequence lengths (after temporal downsampling by
                MovementConvEncoder's factor of 4).

        Returns:
            Motion embedding vectors, shape (batch, output_size).
        """
        num_samples = inputs.shape[0]

        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)

        if isinstance(m_lens, torch.Tensor):
            m_lens = m_lens.data.tolist()

        emb = pack_padded_sequence(input_embs, m_lens, batch_first=True, enforce_sorted=False)
        _gru_seq, gru_last = self.gru(emb, hidden)
        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)

        return self.output_net(gru_last)
