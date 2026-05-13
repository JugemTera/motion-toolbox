"""Fast batched SMPLify3D for maximum speed.

This module provides an optimized batched version using FastSMPLify3D.
Includes optional fast IK initialization for better convergence.
"""

import os

import numpy as np
import smplx
import torch

from .fast_ik_init import fast_smpl_ik, get_smpl_rest_offsets
from .smplify_fast import FastSMPLify3D
from .src import config


class FastBatchedJoints2SMPL:
    """Fast batched interface for converting joints to SMPL parameters.

    Uses FastSMPLify3D with optimized defaults for maximum speed while
    maintaining acceptable quality.
    """

    def __init__(
        self,
        num_frames: int,
        device_id: int = 0,
        cuda: bool = True,
        num_smplify_iters: int = 50,  # Reduced default
        batch_size: int = 32,  # Larger batches
        temporal_smooth: bool = False,
        smooth_window: int = 5,
        use_lbfgs: bool = False,  # Adam by default (faster)
        single_stage: bool = True,  # Single-stage optimization
        joint_loss_weight: float = 800.0,  # Increased for faster convergence
        use_ik_init: bool = False,  # Use fast IK for initialization
    ):
        """Initialize FastBatchedJoints2SMPL converter.

        Args:
            num_frames: Total number of frames in sequence
            device_id: CUDA device ID
            cuda: Whether to use CUDA
            num_smplify_iters: Number of optimization iterations per batch (default 50)
            batch_size: Number of frames to process in parallel (default 32)
            temporal_smooth: Whether to apply temporal smoothing to results
            smooth_window: Window size for temporal smoothing
            use_lbfgs: Use LBFGS optimizer (False = Adam, which is faster)
            single_stage: Use single-stage optimization (faster)
            joint_loss_weight: Weight for joint loss (higher = faster convergence)
            use_ik_init: Use fast analytic IK for initialization (better convergence)
        """
        self.device = torch.device("cuda:" + str(device_id) if cuda and torch.cuda.is_available() else "cpu")
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.num_joints = 22  # HumanML3D
        self.joint_category = "AMASS"
        self.num_smplify_iters = num_smplify_iters
        self.temporal_smooth = temporal_smooth
        self.smooth_window = smooth_window
        self.use_lbfgs = use_lbfgs
        self.single_stage = single_stage
        self.joint_loss_weight = joint_loss_weight
        self.use_ik_init = use_ik_init

        # Create SMPL model
        self.smplmodel = smplx.create(
            config.SMPL_MODEL_DIR, model_type="smpl", gender="neutral", ext="pkl", batch_size=self.batch_size
        ).to(self.device)

        # Cache SMPL offsets for IK if enabled
        if self.use_ik_init:
            self.smpl_offsets = get_smpl_rest_offsets(self.smplmodel, device=self.device)

        # Load mean pose
        import h5py

        smpl_mean_file = config.SMPL_MEAN_FILE
        if os.path.exists(smpl_mean_file):
            file = h5py.File(smpl_mean_file, "r")
            self.init_mean_pose = torch.from_numpy(file["pose"][:]).float().to(self.device)
            self.init_mean_shape = torch.from_numpy(file["shape"][:]).float().to(self.device)
            file.close()
        else:
            self.init_mean_pose = torch.zeros(72).to(self.device)
            self.init_mean_shape = torch.zeros(10).to(self.device)

        self.cam_trans_zero = torch.zeros(3).to(self.device)

        # Initialize FastSMPLify for batched processing
        self.smplify = FastSMPLify3D(
            smplxmodel=self.smplmodel,
            batch_size=self.batch_size,
            joints_category=self.joint_category,
            num_iters=self.num_smplify_iters,
            use_lbfgs=self.use_lbfgs,
            device=self.device,
            single_stage=self.single_stage,
            joint_loss_weight=self.joint_loss_weight,
        )

    def joint2smpl(self, input_joints, init_params=None):
        """Convert joint positions to SMPL parameters using fast batched processing.

        Args:
            input_joints: Joint positions (num_frames, 22, 3)
            init_params: Optional initial parameters

        Returns:
            Dict with pose, betas, transl, vertices, joints
        """
        num_frames = input_joints.shape[0]

        # Initialize output arrays
        all_poses = np.zeros((num_frames, 72))
        all_betas = np.zeros((num_frames, 10))
        all_transl = np.zeros((num_frames, 3))
        all_vertices = []
        all_joints = []

        confidence_input = torch.ones(self.num_joints).to(self.device)

        # Process in batches
        num_batches = (num_frames + self.batch_size - 1) // self.batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, num_frames)
            current_batch_size = end_idx - start_idx

            # Get batch data
            batch_joints_raw = torch.from_numpy(input_joints[start_idx:end_idx]).float().to(self.device)

            # Pad last batch if needed to match self.batch_size
            if current_batch_size < self.batch_size:
                # Pad with zeros (or last frame repeated)
                padding = torch.zeros(
                    self.batch_size - current_batch_size,
                    batch_joints_raw.shape[1],
                    batch_joints_raw.shape[2],
                    device=self.device,
                )
                batch_joints = torch.cat([batch_joints_raw, padding], dim=0)
            else:
                batch_joints = batch_joints_raw

            # Initialize batch with full batch size
            if self.use_ik_init:
                # Use fast IK for initialization
                # Convert HumanML3D 22 joints to SMPL 24 joints by padding
                batch_joints_24 = torch.zeros(self.batch_size, 24, 3, device=self.device)
                batch_joints_24[:, :22] = batch_joints  # Copy available joints

                # Run fast IK
                ik_pose = fast_smpl_ik(batch_joints_24, self.smpl_offsets)  # (B, 24, 3)
                batch_pose = ik_pose.reshape(self.batch_size, -1)  # (B, 72)

                # Still use mean shape
                batch_betas = self.init_mean_shape.unsqueeze(0).repeat(self.batch_size, 1)

                # Estimate camera translation from pelvis position
                batch_cam_t = batch_joints[:, 0, :].clone()  # Use pelvis position
            else:
                # Use mean pose initialization
                batch_pose = self.init_mean_pose.unsqueeze(0).repeat(self.batch_size, 1)
                batch_betas = self.init_mean_shape.unsqueeze(0).repeat(self.batch_size, 1)
                batch_cam_t = self.cam_trans_zero.unsqueeze(0).repeat(self.batch_size, 1)

            # Optimize batch
            new_opt_vertices, new_opt_joints, new_opt_pose, new_opt_betas, new_opt_cam_t, _ = self.smplify(
                batch_pose.detach(),
                batch_betas.detach(),
                batch_cam_t.detach(),
                batch_joints,
                conf_3d=confidence_input,
                seq_ind=0,  # Optimize betas for first batch
            )

            # Only keep the real frames (not padding)
            new_opt_pose = new_opt_pose[:current_batch_size]
            new_opt_betas = new_opt_betas[:current_batch_size]
            new_opt_cam_t = new_opt_cam_t[:current_batch_size]

            # Store results
            all_poses[start_idx:end_idx] = new_opt_pose[:current_batch_size].detach().cpu().numpy()
            all_betas[start_idx:end_idx] = new_opt_betas[:current_batch_size].detach().cpu().numpy()
            all_transl[start_idx:end_idx] = new_opt_cam_t[:current_batch_size].detach().cpu().numpy().squeeze()
            all_vertices.append(new_opt_vertices[:current_batch_size].detach().cpu().numpy())
            all_joints.append(new_opt_joints[:current_batch_size].detach().cpu().numpy())

        # Apply temporal smoothing if requested
        if self.temporal_smooth:
            all_poses = self._temporal_smooth(all_poses)
            all_betas = self._temporal_smooth(all_betas)
            all_transl = self._temporal_smooth(all_transl)

        return {
            "pose": all_poses,
            "betas": all_betas,
            "transl": all_transl,
            "vertices": np.concatenate(all_vertices, axis=0),
            "joints": np.concatenate(all_joints, axis=0),
        }

    def _temporal_smooth(self, data: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing using moving average.

        Args:
            data: Input data (T, D)

        Returns:
            Smoothed data (T, D)
        """
        from scipy.ndimage import uniform_filter1d

        return uniform_filter1d(data, size=self.smooth_window, axis=0, mode="nearest")
