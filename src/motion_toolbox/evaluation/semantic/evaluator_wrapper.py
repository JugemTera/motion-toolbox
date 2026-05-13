"""Evaluator wrapper for loading and using pre-trained text-motion co-embedding models.

Provides a clean interface to load pre-trained T2M evaluation checkpoints and
extract text/motion embeddings for semantic metric computation.

Designed with an abstract base class (BaseEvaluatorWrapper) so that alternative
evaluators (e.g., CLIP-based, custom-trained) can be swapped in later without
changing the evaluation pipeline.

References:
    - MoMask: https://github.com/EricGuo5513/momask-codes/blob/main/models/t2m_eval_wrapper.py
    - MDM: https://github.com/GuyTevet/motion-diffusion-model
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from os.path import join as pjoin
from pathlib import Path

import numpy as np
import torch

from .networks import MotionEncoderBiGRUCo, MovementConvEncoder, TextEncoderBiGRUCo
from .word_vectorizer import POS_enumerator


class BaseEvaluatorWrapper(ABC):
    """Abstract base class for text-motion co-embedding evaluators.

    Subclass this to implement alternative evaluators (e.g., CLIP-based).

    Future extension: Implement this interface with different evaluator architectures
    to support alternative embedding spaces beyond the standard T2M evaluator.
    For example:
        - CLIPEvaluatorWrapper: Uses CLIP text encoder + motion encoder
        - TMREvaluatorWrapper: Uses TMR (Text-Motion Retrieval) embeddings
        - CustomEvaluatorWrapper: Uses project-specific trained models
    """

    @abstractmethod
    def get_co_embeddings(
        self,
        word_embs: torch.Tensor,
        pos_ohot: torch.Tensor,
        cap_lens: torch.Tensor,
        motions: torch.Tensor,
        m_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract paired text and motion embeddings.

        Args:
            word_embs: Word embeddings, shape (batch, max_text_len, word_dim).
            pos_ohot: POS one-hot vectors, shape (batch, max_text_len, pos_dim).
            cap_lens: Caption lengths, shape (batch,).
            motions: Motion features, shape (batch, seq_len, motion_dim).
            m_lens: Motion lengths, shape (batch,).

        Returns:
            Tuple of (text_embeddings, motion_embeddings), each shape (batch, embed_dim).
        """
        ...

    @abstractmethod
    def get_motion_embeddings(
        self,
        motions: torch.Tensor,
        m_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Extract motion embeddings only (without text).

        Args:
            motions: Motion features, shape (batch, seq_len, motion_dim).
            m_lens: Motion lengths, shape (batch,).

        Returns:
            Motion embeddings, shape (batch, embed_dim).
        """
        ...


class T2MEvaluatorWrapper(BaseEvaluatorWrapper):
    """Standard T2M evaluation wrapper using the HumanML3D co-embedding model.

    Loads the pre-trained MovementConvEncoder, TextEncoderBiGRUCo, and
    MotionEncoderBiGRUCo from the T2M evaluation checkpoints.

    Checkpoint structure (finest.tar):
        - 'movement_encoder': MovementConvEncoder state dict
        - 'text_encoder': TextEncoderBiGRUCo state dict
        - 'motion_encoder': MotionEncoderBiGRUCo state dict
        - 'epoch': Training epoch number

    Args:
        dataset_name: Either "humanml" (HumanML3D, 263-d) or "kit" (KIT-ML, 251-d).
        checkpoint_dir: Path to the directory containing the evaluator checkpoints.
            Expected structure: {checkpoint_dir}/{dataset_ckpt}/text_mot_match/model/finest.tar
            where dataset_ckpt is "t2m" for humanml or "kit" for kit.
        device: Device to load the models on (default "cpu").

    Example:
        >>> wrapper = T2MEvaluatorWrapper(
        ...     dataset_name="humanml",
        ...     checkpoint_dir="./checkpoints",
        ...     device="cuda",
        ... )
        >>> text_emb, motion_emb = wrapper.get_co_embeddings(
        ...     word_embs, pos_ohot, cap_lens, motions, m_lens
        ... )
    """

    # Default hyperparameters matching the T2M evaluation protocol
    DEFAULT_CONFIG = {
        "dim_word": 300,
        "max_motion_length": 196,
        "dim_motion_hidden": 1024,
        "max_text_len": 20,
        "dim_text_hidden": 512,
        "dim_coemb_hidden": 512,
        "dim_movement_enc_hidden": 512,
        "dim_movement_latent": 512,
        "unit_length": 4,
    }

    # Motion feature dimensions per dataset
    DATASET_DIM_POSE = {
        "humanml": 263,
        "t2m": 263,
        "kit": 251,
    }

    # Checkpoint directory name per dataset
    DATASET_CKPT_DIR = {
        "humanml": "t2m",
        "t2m": "t2m",
        "kit": "kit",
    }

    def __init__(
        self,
        dataset_name: str = "humanml",
        checkpoint_dir: str = "./checkpoints",
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dataset_name = dataset_name

        if dataset_name not in self.DATASET_DIM_POSE:
            raise ValueError(f"Unknown dataset '{dataset_name}'. Supported: {list(self.DATASET_DIM_POSE.keys())}")

        self.dim_pose = self.DATASET_DIM_POSE[dataset_name]
        self.unit_length = self.DEFAULT_CONFIG["unit_length"]

        # Build network components
        self.movement_encoder = MovementConvEncoder(
            input_size=self.dim_pose - 4,  # Exclude last 4 contact dims
            hidden_size=self.DEFAULT_CONFIG["dim_movement_enc_hidden"],
            output_size=self.DEFAULT_CONFIG["dim_movement_latent"],
        )
        self.text_encoder = TextEncoderBiGRUCo(
            word_size=self.DEFAULT_CONFIG["dim_word"],
            pos_size=len(POS_enumerator),
            hidden_size=self.DEFAULT_CONFIG["dim_text_hidden"],
            output_size=self.DEFAULT_CONFIG["dim_coemb_hidden"],
            device=self.device,
        )
        self.motion_encoder = MotionEncoderBiGRUCo(
            input_size=self.DEFAULT_CONFIG["dim_movement_latent"],
            hidden_size=self.DEFAULT_CONFIG["dim_motion_hidden"],
            output_size=self.DEFAULT_CONFIG["dim_coemb_hidden"],
            device=self.device,
        )

        # Load pre-trained weights
        self._load_checkpoint(checkpoint_dir)

        # Move to device and set eval mode
        self.movement_encoder.to(self.device).eval()
        self.text_encoder.to(self.device).eval()
        self.motion_encoder.to(self.device).eval()

    def _load_checkpoint(self, checkpoint_dir: str) -> None:
        """Load pre-trained weights from the T2M evaluation checkpoint."""
        ckpt_dir = self.DATASET_CKPT_DIR[self.dataset_name]
        ckpt_path = pjoin(checkpoint_dir, ckpt_dir, "text_mot_match", "model", "finest.tar")

        if not Path(ckpt_path).exists():
            raise FileNotFoundError(
                f"Evaluator checkpoint not found at: {ckpt_path}\n"
                f"Please download the T2M evaluator checkpoints. See:\n"
                f"  scripts/download_t2m_evaluators.sh"
            )

        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.movement_encoder.load_state_dict(checkpoint["movement_encoder"])
        self.text_encoder.load_state_dict(checkpoint["text_encoder"])
        self.motion_encoder.load_state_dict(checkpoint["motion_encoder"])
        print(f"Loaded T2M Evaluator (dataset={self.dataset_name}, epoch={checkpoint.get('epoch', '?')})")

    @torch.no_grad()
    def get_co_embeddings(
        self,
        word_embs: torch.Tensor,
        pos_ohot: torch.Tensor,
        cap_lens: torch.Tensor,
        motions: torch.Tensor,
        m_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract paired text and motion embeddings.

        Note: Results are sorted by descending motion length internally
        (required by pack_padded_sequence), but the output order matches
        the input order.

        Args:
            word_embs: Word embeddings, shape (batch, max_text_len, 300).
            pos_ohot: POS one-hot vectors, shape (batch, max_text_len, 15).
            cap_lens: Caption lengths, shape (batch,).
            motions: HumanML3D motion features, shape (batch, seq_len, 263).
            m_lens: Motion frame lengths, shape (batch,).

        Returns:
            Tuple of (text_embeddings, motion_embeddings), each shape (batch, 512).
        """
        word_embs = word_embs.detach().to(self.device).float()
        pos_ohot = pos_ohot.detach().to(self.device).float()
        motions = motions.detach().to(self.device).float()

        # Sort by motion length (descending) for pack_padded_sequence
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        motions = motions[align_idx]
        m_lens = m_lens[align_idx]

        # Motion encoding: raw features -> movement latents -> motion embedding
        movements = self.movement_encoder(motions[..., :-4]).detach()
        m_lens_downsampled = m_lens // self.unit_length
        motion_embedding = self.motion_encoder(movements, m_lens_downsampled)

        # Text encoding
        text_embedding = self.text_encoder(word_embs, pos_ohot, cap_lens)
        # Re-order text embeddings to match the sorted motion order
        text_embedding = text_embedding[align_idx]

        return text_embedding, motion_embedding

    @torch.no_grad()
    def get_motion_embeddings(
        self,
        motions: torch.Tensor,
        m_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Extract motion embeddings only (for FID, Diversity, etc.).

        Args:
            motions: HumanML3D motion features, shape (batch, seq_len, 263).
            m_lens: Motion frame lengths, shape (batch,).

        Returns:
            Motion embeddings, shape (batch, 512).
        """
        motions = motions.detach().to(self.device).float()

        # Sort by motion length (descending)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        motions = motions[align_idx]
        m_lens = m_lens[align_idx]

        # Motion encoding
        movements = self.movement_encoder(motions[..., :-4]).detach()
        m_lens_downsampled = m_lens // self.unit_length
        motion_embedding = self.motion_encoder(movements, m_lens_downsampled)

        return motion_embedding
