"""SMPLify3D optimization for fitting SMPL to 3D joints.

Ported from MDM repository:
https://github.com/GuyTevet/motion-diffusion-model
"""

import os

import numpy as np
import smplx
import torch

from .src import config
from .src.customloss import body_fitting_loss_3d, camera_fitting_loss_3d
from .src.prior import MaxMixturePrior


@torch.no_grad()
def guess_init_3d(model_joints, j3d, joints_category="orig"):
    """Initialize the translation via triangle similarity using torso joints.

    Args:
        model_joints: SMPL model joints
        j3d: Target 3D joints
        joints_category: Joint category ("orig" or "AMASS")

    Returns:
        Initial translation estimate
    """
    gt_joints = ["RHip", "LHip", "RShoulder", "LShoulder"]
    gt_joints_ind = [config.JOINT_MAP[joint] for joint in gt_joints]

    if joints_category == "orig":
        joints_ind_category = [config.JOINT_MAP[joint] for joint in gt_joints]
    elif joints_category == "AMASS":
        joints_ind_category = [config.AMASS_JOINT_MAP[joint] for joint in gt_joints]
    else:
        raise ValueError(f"Unknown joints category: {joints_category}")

    sum_init_t = (j3d[:, joints_ind_category] - model_joints[:, gt_joints_ind]).sum(dim=1)
    init_t = sum_init_t / 4.0
    return init_t


class SMPLify3D:
    """Implementation of SMPLify for 3D joints."""

    def __init__(
        self,
        smplxmodel,
        step_size=1e-2,
        batch_size=1,
        num_iters=100,
        use_collision=False,
        use_lbfgs=True,
        joints_category="orig",
        device=torch.device("cuda:0"),
    ):
        """Initialize SMPLify3D optimizer.

        Args:
            smplxmodel: SMPL model instance
            step_size: Optimization step size (default: 1e-2)
            batch_size: Batch size for processing (default: 1)
            num_iters: Number of optimization iterations (default: 100)
            use_collision: Whether to use collision detection (default: False)
            use_lbfgs: Whether to use LBFGS optimizer (default: True)
            joints_category: Joint category to use (default: "orig")
            device: Device to run on (default: cuda:0)
        """
        self.batch_size = batch_size
        self.device = device
        self.step_size = step_size
        self.num_iters = num_iters
        self.use_lbfgs = use_lbfgs
        self.use_collision = use_collision
        self.joints_category = joints_category

        # GMM pose prior
        self.pose_prior = MaxMixturePrior(prior_folder=config.GMM_MODEL_DIR, num_gaussians=8, dtype=torch.float32).to(
            device
        )

        self.smpl = smplxmodel
        self.model_faces = smplxmodel.faces_tensor.view(-1)

        # Select joint category
        if joints_category == "orig":
            self.smpl_index = config.full_smpl_idx
            self.corr_index = config.full_smpl_idx
        elif joints_category == "AMASS":
            self.smpl_index = config.amass_smpl_idx
            self.corr_index = config.amass_idx
        else:
            self.smpl_index = None
            self.corr_index = None

    def __call__(self, init_pose, init_betas, init_cam_t, j3d, conf_3d=1.0, seq_ind=0):
        """Perform body fitting.

        Args:
            init_pose: Initial SMPL pose estimate (batch, 72)
            init_betas: Initial SMPL betas estimate (batch, 10)
            init_cam_t: Initial camera translation estimate (batch, 3)
            j3d: Target 3D joints (batch, num_joints, 3)
            conf_3d: Confidence for 3D joints
            seq_ind: Sequence index (for fixing betas after first frame)

        Returns:
            vertices, joints, pose, betas, camera_translation, final_loss
        """
        # Split SMPL pose to body pose and global orientation
        body_pose = init_pose[:, 3:].detach().clone()
        global_orient = init_pose[:, :3].detach().clone()
        betas = init_betas.detach().clone()

        # Get initial estimate
        batch_size = init_pose.shape[0]
        smpl_output = self.smpl(global_orient=global_orient, body_pose=body_pose, betas=betas)
        model_joints = smpl_output.joints

        init_cam_t = guess_init_3d(model_joints, j3d, self.joints_category).unsqueeze(1).detach()
        camera_translation = init_cam_t.clone()

        preserve_pose = init_pose[:, 3:].detach().clone()

        # Step 1: Optimize camera translation and body orientation
        body_pose.requires_grad = False
        betas.requires_grad = False
        global_orient.requires_grad = True
        camera_translation.requires_grad = True

        camera_opt_params = [global_orient, camera_translation]

        if self.use_lbfgs:
            camera_optimizer = torch.optim.LBFGS(
                camera_opt_params, max_iter=self.num_iters, lr=self.step_size, line_search_fn="strong_wolfe"
            )
            for i in range(10):

                def closure():
                    camera_optimizer.zero_grad()
                    smpl_output = self.smpl(global_orient=global_orient, body_pose=body_pose, betas=betas)
                    model_joints = smpl_output.joints
                    loss = camera_fitting_loss_3d(
                        model_joints, camera_translation, init_cam_t, j3d, self.joints_category
                    )
                    loss.backward()
                    return loss

                camera_optimizer.step(closure)
        else:
            camera_optimizer = torch.optim.Adam(camera_opt_params, lr=self.step_size, betas=(0.9, 0.999))
            for i in range(20):
                smpl_output = self.smpl(global_orient=global_orient, body_pose=body_pose, betas=betas)
                model_joints = smpl_output.joints
                loss = camera_fitting_loss_3d(
                    model_joints[:, self.smpl_index],
                    camera_translation,
                    init_cam_t,
                    j3d[:, self.corr_index],
                    self.joints_category,
                )
                camera_optimizer.zero_grad()
                loss.backward()
                camera_optimizer.step()

        # Step 2: Optimize body joints
        body_pose.requires_grad = True
        global_orient.requires_grad = True
        camera_translation.requires_grad = True

        if seq_ind == 0:
            betas.requires_grad = True
            body_opt_params = [body_pose, betas, global_orient, camera_translation]
        else:
            betas.requires_grad = False
            body_opt_params = [body_pose, global_orient, camera_translation]

        if self.use_lbfgs:
            body_optimizer = torch.optim.LBFGS(
                body_opt_params, max_iter=self.num_iters, lr=self.step_size, line_search_fn="strong_wolfe"
            )
            for i in range(self.num_iters):

                def closure():
                    body_optimizer.zero_grad()
                    smpl_output = self.smpl(global_orient=global_orient, body_pose=body_pose, betas=betas)
                    model_joints = smpl_output.joints
                    model_vertices = smpl_output.vertices

                    loss = body_fitting_loss_3d(
                        body_pose,
                        preserve_pose,
                        betas,
                        model_joints[:, self.smpl_index],
                        camera_translation,
                        j3d[:, self.corr_index],
                        self.pose_prior,
                        joints3d_conf=conf_3d,
                        joint_loss_weight=600.0,
                        pose_preserve_weight=5.0,
                        use_collision=self.use_collision,
                        model_vertices=model_vertices,
                        model_faces=self.model_faces,
                    )
                    loss.backward()
                    return loss

                body_optimizer.step(closure)
        else:
            body_optimizer = torch.optim.Adam(body_opt_params, lr=self.step_size, betas=(0.9, 0.999))
            for i in range(self.num_iters):
                smpl_output = self.smpl(global_orient=global_orient, body_pose=body_pose, betas=betas)
                model_joints = smpl_output.joints
                model_vertices = smpl_output.vertices

                loss = body_fitting_loss_3d(
                    body_pose,
                    preserve_pose,
                    betas,
                    model_joints[:, self.smpl_index],
                    camera_translation,
                    j3d[:, self.corr_index],
                    self.pose_prior,
                    joints3d_conf=conf_3d,
                    joint_loss_weight=600.0,
                    use_collision=self.use_collision,
                    model_vertices=model_vertices,
                    model_faces=self.model_faces,
                )
                body_optimizer.zero_grad()
                loss.backward()
                body_optimizer.step()

        # Get final loss
        with torch.no_grad():
            smpl_output = self.smpl(
                global_orient=global_orient, body_pose=body_pose, betas=betas, return_full_pose=True
            )
            model_joints = smpl_output.joints
            model_vertices = smpl_output.vertices

            final_loss = body_fitting_loss_3d(
                body_pose,
                preserve_pose,
                betas,
                model_joints[:, self.smpl_index],
                camera_translation,
                j3d[:, self.corr_index],
                self.pose_prior,
                joints3d_conf=conf_3d,
                joint_loss_weight=600.0,
                use_collision=self.use_collision,
                model_vertices=model_vertices,
                model_faces=self.model_faces,
            )

        vertices = smpl_output.vertices.detach()
        joints = smpl_output.joints.detach()
        pose = torch.cat([global_orient, body_pose], dim=-1).detach()
        betas = betas.detach()

        return vertices, joints, pose, betas, camera_translation, final_loss


class Joints2SMPL:
    """High-level interface for converting joints to SMPL parameters.

    Processes sequences frame-by-frame for better temporal consistency.
    """

    def __init__(
        self,
        num_frames: int,
        device_id: int = 0,
        cuda: bool = True,
        num_smplify_iters: int = 150,
        batch_size: int = 1,
        use_lbfgs: bool = True,
    ):
        """Initialize Joints2SMPL converter.

        Args:
            num_frames: Total number of frames in sequence
            device_id: CUDA device ID
            cuda: Whether to use CUDA
            num_smplify_iters: Number of optimization iterations per frame
            batch_size: Batch size for processing (1 for frame-by-frame)
            use_lbfgs: Use LBFGS optimizer (True) or Adam (False)
        """
        self.device = torch.device("cuda:" + str(device_id) if cuda and torch.cuda.is_available() else "cpu")
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.num_joints = 22  # HumanML3D
        self.joint_category = "AMASS"
        self.num_smplify_iters = num_smplify_iters
        self.use_lbfgs = use_lbfgs

        # Create SMPL model
        self.smplmodel = smplx.create(
            config.SMPL_MODEL_DIR, model_type="smpl", gender="neutral", ext="pkl", batch_size=self.batch_size
        ).to(self.device)

        # Load mean pose
        import h5py

        smpl_mean_file = config.SMPL_MEAN_FILE
        if os.path.exists(smpl_mean_file):
            file = h5py.File(smpl_mean_file, "r")
            self.init_mean_pose = torch.from_numpy(file["pose"][:]).unsqueeze(0).float().to(self.device)
            self.init_mean_shape = torch.from_numpy(file["shape"][:]).unsqueeze(0).float().to(self.device)
            file.close()
        else:
            # Fallback to zero initialization
            self.init_mean_pose = torch.zeros(1, 72).to(self.device)
            self.init_mean_shape = torch.zeros(1, 10).to(self.device)

        self.cam_trans_zero = torch.Tensor([0.0, 0.0, 0.0]).unsqueeze(0).to(self.device)

        # Initialize SMPLify
        self.smplify = SMPLify3D(
            smplxmodel=self.smplmodel,
            batch_size=self.batch_size,
            joints_category=self.joint_category,
            num_iters=self.num_smplify_iters,
            use_lbfgs=self.use_lbfgs,
            device=self.device,
        )

    def joint2smpl(self, input_joints, init_params=None):
        """Convert joint positions to SMPL parameters.

        Processes frames sequentially for better temporal consistency.

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

        # Initialize with mean pose
        prev_pose = self.init_mean_pose.clone()
        prev_betas = self.init_mean_shape.clone()
        prev_cam_t = self.cam_trans_zero.clone()

        confidence_input = torch.ones(self.num_joints).to(self.device)

        # Process each frame
        for frame_idx in range(num_frames):
            keypoints_3d = torch.from_numpy(input_joints[frame_idx : frame_idx + 1]).float().to(self.device)

            # Use previous frame's result as initialization
            new_opt_vertices, new_opt_joints, new_opt_pose, new_opt_betas, new_opt_cam_t, _ = self.smplify(
                prev_pose.detach(),
                prev_betas.detach(),
                prev_cam_t.detach(),
                keypoints_3d,
                conf_3d=confidence_input,
                seq_ind=frame_idx,
            )

            # Store results
            all_poses[frame_idx] = new_opt_pose.detach().cpu().numpy()
            all_betas[frame_idx] = new_opt_betas.detach().cpu().numpy()
            all_transl[frame_idx] = new_opt_cam_t.detach().cpu().numpy().squeeze()
            all_vertices.append(new_opt_vertices.detach().cpu().numpy())
            all_joints.append(new_opt_joints.detach().cpu().numpy())

            # Update for next frame
            prev_pose = new_opt_pose
            prev_betas = new_opt_betas
            prev_cam_t = new_opt_cam_t

        return {
            "pose": all_poses,
            "betas": all_betas,
            "transl": all_transl,
            "vertices": np.concatenate(all_vertices, axis=0),
            "joints": np.concatenate(all_joints, axis=0),
        }
