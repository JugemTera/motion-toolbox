"""SMPL ↔ MHR converter using official MHR conversion tools."""

import warnings
from pathlib import Path

import numpy as np
import torch

from ..core.motion import Motion
from .base_converter import BaseMotionConverter


class SMPLToMHRConverter(BaseMotionConverter):
    """Convert SMPL Motion to MHR parameters.

    Uses the official MHR conversion tools to convert SMPL parameters to MHR.
    """

    input_format = "motion_smpl"
    output_format = "motion_mhr"

    def __init__(
        self,
        smpl_model_path: str | None = None,
        mhr_assets_path: str | None = None,
        device: str = "cpu",
        method: str = "pytorch",  # or "pymomentum"
        is_tracking: bool = True,  # Use temporal tracking for smoother results
        single_identity: bool = True,  # Use consistent identity across frames
        batch_size: int = 256,  # Batch size for processing
    ):
        """Initialize converter.

        Args:
            smpl_model_path: Path to SMPL model file (.pkl or .npz)
            mhr_assets_path: Path to MHR assets directory (default: /tmp/MHR/assets)
            device: Device for computation
            method: Conversion method ("pytorch" or "pymomentum")
            is_tracking: Whether to use temporal tracking (previous frame init)
            single_identity: Whether to use consistent identity across frames
            batch_size: Number of frames per batch
        """
        self.smpl_model_path = smpl_model_path
        self.mhr_assets_path = mhr_assets_path or "/tmp/MHR/assets"
        self.device = torch.device(device)
        self.method = method
        self.is_tracking = is_tracking
        self.single_identity = single_identity
        self.batch_size = batch_size
        self._mhr_model = None
        self._smpl_model = None
        self._converter = None

    def _get_models(self):
        """Lazy load MHR and SMPL models."""
        if self._mhr_model is None:
            try:
                from pathlib import Path

                from mhr.mhr import MHR

                assets_path = Path(self.mhr_assets_path)
                if not assets_path.exists():
                    raise RuntimeError(f"MHR assets not found at {assets_path}")

                self._mhr_model = MHR.from_files(folder=assets_path, lod=1, device=self.device)
            except ImportError:
                raise RuntimeError("mhr package required for SMPL→MHR conversion")

        if self._smpl_model is None and self.smpl_model_path:
            try:
                import os

                import smplx

                # If smpl_model_path is a directory, look for SMPL_NEUTRAL.pkl in smpl/ subdirectory
                if os.path.isdir(self.smpl_model_path):
                    smpl_file = os.path.join(self.smpl_model_path, "smpl", "SMPL_NEUTRAL.pkl")
                    if not os.path.exists(smpl_file):
                        smpl_file = os.path.join(self.smpl_model_path, "SMPL_NEUTRAL.pkl")
                else:
                    smpl_file = self.smpl_model_path

                self._smpl_model = smplx.SMPL(model_path=smpl_file)
            except Exception as e:
                warnings.warn(f"Could not load SMPL model: {e}")

        return self._mhr_model, self._smpl_model

    def _get_converter(self):
        """Lazy load the MHR converter."""
        if self._converter is None:
            try:
                # Import the conversion module from MHR tools
                import os
                import sys

                mhr_tools_path = Path("/tmp/MHR/tools/mhr_smpl_conversion")
                if mhr_tools_path.exists():
                    sys.path.insert(0, str(mhr_tools_path))

                    # Change to the tools directory so relative asset paths work
                    original_cwd = os.getcwd()
                    os.chdir(str(mhr_tools_path))

                    try:
                        from conversion import Conversion

                        mhr_model, smpl_model = self._get_models()
                        self._converter = Conversion(mhr_model=mhr_model, smpl_model=smpl_model, method=self.method)
                    finally:
                        # Restore original directory
                        os.chdir(original_cwd)
                else:
                    raise RuntimeError(
                        "MHR conversion tools not found. "
                        "Please ensure /tmp/MHR is set up with: "
                        "cd /tmp && git clone https://github.com/facebookresearch/MHR.git"
                    )
            except ImportError as e:
                raise RuntimeError(f"Could not import MHR conversion tools: {e}")

        return self._converter

    def convert(self, motion: Motion) -> Motion:
        """Convert SMPL Motion to MHR Motion.

        Args:
            motion: Motion object with SMPL parameters in model_data

        Returns:
            Motion object with MHR parameters in model_data
        """
        if motion.body_model_type != "smpl":
            raise ValueError(f"Expected SMPL motion, got {motion.body_model_type}")

        if "betas" not in motion.model_data or "body_pose" not in motion.model_data:
            raise ValueError("Motion must have SMPL parameters in model_data")

        # Get models and converter
        converter = self._get_converter()
        _, smpl_model = self._get_models()

        if smpl_model is None:
            raise RuntimeError("SMPL model required for conversion")

        # Convert Motion to SMPL parameters using MotionToSMPLConverter
        # This handles the conversion from joint_rotations to proper SMPL format
        from ..converters.motion_converters import MotionToSMPLConverter

        smpl_converter = MotionToSMPLConverter()
        smpl_params_dict = smpl_converter.convert(motion)

        # Extract parameters
        body_pose = smpl_params_dict["body_pose"].to(self.device)
        global_orient = smpl_params_dict["global_orient"].to(self.device)
        betas = smpl_params_dict["betas"].to(self.device)
        transl = smpl_params_dict["transl"].to(self.device)

        # Generate SMPL vertices
        print(f"  Generating SMPL vertices for {motion.num_frames} frames...")
        smpl_vertices = []
        with torch.no_grad():
            for i in range(motion.num_frames):
                output = smpl_model(
                    global_orient=global_orient[i : i + 1],
                    body_pose=body_pose[i : i + 1],
                    betas=betas[i : i + 1],
                    transl=transl[i : i + 1],
                )
                smpl_vertices.append(output.vertices)

        smpl_vertices = torch.cat(smpl_vertices, dim=0)
        print(f"  SMPL vertices shape: {smpl_vertices.shape}")

        # Convert SMPL vertices to MHR using the converter
        # Need to change to MHR tools directory for relative asset paths
        import os

        original_cwd = os.getcwd()
        mhr_tools_path = Path("/tmp/MHR/tools/mhr_smpl_conversion")
        os.chdir(str(mhr_tools_path))

        try:
            print(f"  Converting SMPL→MHR using {self.method} method...")
            print(
                f"    is_tracking={self.is_tracking}, single_identity={self.single_identity}, batch_size={self.batch_size}"
            )
            conversion_results = converter.convert_smpl2mhr(
                smpl_vertices=smpl_vertices,
                smpl_parameters=None,  # Use vertices, not parameters
                single_identity=self.single_identity,
                is_tracking=self.is_tracking,  # Use temporal tracking for smoother results
                return_mhr_parameters=True,
                return_mhr_vertices=False,
                return_mhr_meshes=False,
                return_fitting_errors=True,
                batch_size=self.batch_size,
            )
        finally:
            os.chdir(original_cwd)

        if conversion_results.result_errors is not None and len(conversion_results.result_errors) > 0:
            errors = np.array(conversion_results.result_errors)
            print(f"  Conversion errors: mean={np.mean(errors):.4f}, max={np.max(errors):.4f}")

        mhr_params = conversion_results.result_parameters

        # Create MHR Motion object
        from ..converters.motion_converters import MHRToMotionConverter

        # Prepare MHR data dict
        # Note: MHR conversion tools use 'lbs_model_params', but MHR model uses 'model_parameters'
        mhr_data = {
            "model_parameters": mhr_params["lbs_model_params"].detach().cpu(),
            "identity_coeffs": mhr_params["identity_coeffs"].detach().cpu(),
            "face_expr_coeffs": mhr_params.get("face_expr_coeffs", torch.zeros(motion.num_frames, 72)).detach().cpu(),
        }

        # Convert to Motion using MHRToMotionConverter
        mhr_converter = MHRToMotionConverter()
        mhr_motion = mhr_converter.convert(mhr_data)

        print("  ✅ Conversion complete!")

        return mhr_motion


class MHRToSMPLConverter(BaseMotionConverter):
    """Convert MHR Motion to SMPL parameters.

    Uses the official MHR conversion tools to convert MHR parameters to SMPL.
    """

    input_format = "motion_mhr"
    output_format = "motion_smpl"

    def __init__(
        self,
        smpl_model_path: str | None = None,
        mhr_assets_path: str | None = None,
        device: str = "cpu",
        method: str = "pytorch",
    ):
        """Initialize converter.

        Args:
            smpl_model_path: Path to SMPL model file
            mhr_assets_path: Path to MHR assets directory (default: /tmp/MHR/assets)
            device: Device for computation
            method: Conversion method ("pytorch" or "pymomentum")
        """
        self.smpl_model_path = smpl_model_path
        self.mhr_assets_path = mhr_assets_path or "/tmp/MHR/assets"
        self.device = torch.device(device)
        self.method = method
        self._mhr_model = None
        self._smpl_model = None
        self._converter = None

    def _get_models(self):
        """Lazy load MHR and SMPL models."""
        if self._mhr_model is None:
            try:
                from pathlib import Path

                from mhr.mhr import MHR

                assets_path = Path(self.mhr_assets_path)
                if not assets_path.exists():
                    raise RuntimeError(f"MHR assets not found at {assets_path}")

                self._mhr_model = MHR.from_files(folder=assets_path, lod=1, device=self.device)
            except ImportError:
                raise RuntimeError("mhr package required for MHR→SMPL conversion")

        if self._smpl_model is None and self.smpl_model_path:
            try:
                import os

                import smplx

                # If smpl_model_path is a directory, look for SMPL_NEUTRAL.pkl in smpl/ subdirectory
                if os.path.isdir(self.smpl_model_path):
                    smpl_file = os.path.join(self.smpl_model_path, "smpl", "SMPL_NEUTRAL.pkl")
                    if not os.path.exists(smpl_file):
                        smpl_file = os.path.join(self.smpl_model_path, "SMPL_NEUTRAL.pkl")
                else:
                    smpl_file = self.smpl_model_path

                self._smpl_model = smplx.SMPL(model_path=smpl_file)
            except Exception as e:
                warnings.warn(f"Could not load SMPL model: {e}")

        return self._mhr_model, self._smpl_model

    def _get_converter(self):
        """Lazy load the MHR converter."""
        if self._converter is None:
            try:
                import os
                import sys

                mhr_tools_path = Path("/tmp/MHR/tools/mhr_smpl_conversion")
                if mhr_tools_path.exists():
                    sys.path.insert(0, str(mhr_tools_path))

                    # Change to the tools directory so relative asset paths work
                    original_cwd = os.getcwd()
                    os.chdir(str(mhr_tools_path))

                    try:
                        from conversion import Conversion

                        mhr_model, smpl_model = self._get_models()
                        self._converter = Conversion(mhr_model=mhr_model, smpl_model=smpl_model, method=self.method)
                    finally:
                        # Restore original directory
                        os.chdir(original_cwd)
                else:
                    raise RuntimeError("MHR conversion tools not found at /tmp/MHR")
            except ImportError as e:
                raise RuntimeError(f"Could not import MHR conversion tools: {e}")

        return self._converter

    def convert(self, motion: Motion) -> Motion:
        """Convert MHR Motion to SMPL Motion.

        Args:
            motion: Motion object with MHR parameters in model_data

        Returns:
            Motion object with SMPL parameters in model_data
        """
        if motion.body_model_type != "mhr":
            raise ValueError(f"Expected MHR motion, got {motion.body_model_type}")

        if "model_parameters" not in motion.model_data:
            raise ValueError("Motion must have MHR parameters in model_data")

        # Get converter
        converter = self._get_converter()

        # Prepare MHR parameters
        mhr_params = {
            "model_parameters": torch.as_tensor(motion.model_data["model_parameters"], device=self.device).float(),
            "identity_coeffs": torch.as_tensor(
                motion.model_data.get("identity_coeffs", torch.zeros(45)), device=self.device
            ).float(),
            "face_expr_coeffs": torch.as_tensor(
                motion.model_data.get("face_expr_coeffs", torch.zeros((motion.num_frames, 72))), device=self.device
            ).float(),
        }

        # Convert MHR to SMPL
        conversion_results = converter.convert_mhr2smpl(
            mhr_vertices=None,
            mhr_parameters=mhr_params,
            single_identity=True,
            return_smpl_parameters=True,
            return_smpl_vertices=False,
            return_smpl_meshes=False,
            return_fitting_errors=False,
        )

        smpl_params = conversion_results.result_parameters

        # Convert to Motion using SMPLToMotionConverter
        from ..converters.motion_converters import SMPLToMotionConverter

        smpl_data = {
            "body_pose": smpl_params["body_pose"].cpu().numpy(),
            "global_orient": smpl_params.get("global_orient", torch.zeros(motion.num_frames, 3)).cpu().numpy(),
            "transl": smpl_params.get("transl", torch.zeros(motion.num_frames, 3)).cpu().numpy(),
            "betas": smpl_params["betas"].cpu().numpy(),
        }

        smpl_converter = SMPLToMotionConverter()
        smpl_motion = smpl_converter.convert(smpl_data)

        return smpl_motion
