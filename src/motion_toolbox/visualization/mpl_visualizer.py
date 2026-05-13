"""Matplotlib-based lightweight motion visualizer.

This module provides CPU-based 3D motion rendering using matplotlib.
Serves as a fallback when PyTorch3D is not available or GPU acceleration is not needed.
"""
# TODO: consider removing this now that the html visualizer is working.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from ..core.motion import Motion


class MplVisualizer:
    """Lightweight CPU-based motion visualizer using matplotlib."""

    def __init__(
        self,
        figsize: tuple[int, int] = (10, 8),
        dpi: int = 100,
        joint_color: str = "blue",
        bone_color: str = "darkblue",
        joint_size: int = 50,
        bone_width: float = 2.0,
        ground_color: str = "lightgray",
        background_color: str = "white",
    ):
        """Initialize the matplotlib visualizer.

        Args:
            figsize: Figure size (width, height)
            dpi: Figure resolution
            joint_color: Color for joint markers
            bone_color: Color for bone lines
            joint_size: Size of joint markers
            bone_width: Width of bone lines
            ground_color: Color for ground plane
            background_color: Background color
        """
        self.figsize = figsize
        self.dpi = dpi
        self.joint_color = joint_color
        self.bone_color = bone_color
        self.joint_size = joint_size
        self.bone_width = bone_width
        self.ground_color = ground_color
        self.background_color = background_color

        # Default skeleton connectivity (SMPL-like)
        self.default_skeleton_connections = [
            (0, 1),
            (0, 2),
            (0, 3),  # Root to spine, left hip, right hip
            (1, 4),
            (2, 5),
            (3, 6),  # Spine to head, left knee, right knee
            (5, 7),
            (6, 8),  # Knees to ankles
            (1, 9),
            (1, 12),  # Spine to shoulders
            (9, 10),
            (12, 13),  # Shoulders to elbows
            (10, 11),
            (13, 14),  # Elbows to wrists
        ]

    def plot_skeleton_3d(
        self,
        joint_positions: np.ndarray,
        connections: list[tuple[int, int]] | None = None,
        ax: plt.Axes | None = None,
        title: str = "3D Skeleton",
        up_axis: str = "y",
    ) -> plt.Axes:
        """Plot 3D skeleton from joint positions.

        Args:
            joint_positions: (N, 3) array of joint positions
            connections: List of (parent, child) joint indices
            ax: Optional matplotlib 3D axes
            title: Plot title
            up_axis: 'y' for Y-up, 'z' for Z-up coordinate system

        Returns:
            Matplotlib 3D axes
        """
        if connections is None:
            connections = self.default_skeleton_connections

        if ax is None:
            fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
            ax = fig.add_subplot(111, projection="3d")

        # Clear previous plot
        ax.clear()

        # Transform coordinates for proper axis orientation
        if up_axis.lower() == "y":
            # Swap Y and Z to make Y the vertical axis in matplotlib
            plot_positions = joint_positions[:, [0, 2, 1]].copy()  # X, Z, Y -> X, Y, Z in plot
        else:
            # Z is already up, no transformation needed
            plot_positions = joint_positions.copy()

        # Plot joints
        ax.scatter(
            plot_positions[:, 0],
            plot_positions[:, 1],
            plot_positions[:, 2],
            c=self.joint_color,
            s=self.joint_size,
            alpha=0.8,
        )

        # Plot bones
        for parent_idx, child_idx in connections:
            if parent_idx >= len(joint_positions) or child_idx >= len(joint_positions):
                continue

            parent_pos = plot_positions[parent_idx]
            child_pos = plot_positions[child_idx]

            ax.plot(
                [parent_pos[0], child_pos[0]],
                [parent_pos[1], child_pos[1]],
                [parent_pos[2], child_pos[2]],
                c=self.bone_color,
                linewidth=self.bone_width,
                alpha=0.7,
            )

        # Set equal aspect ratio and labels
        self._set_3d_axes_equal(ax, plot_positions)
        if up_axis.lower() == "y":
            ax.set_xlabel("X")
            ax.set_ylabel("Z")  # Z is plotted as Y
            ax.set_zlabel("Y (Up)")  # Y is plotted as Z (up)
            # Set camera view for Y-up: looking slightly down and angled
            ax.view_init(elev=15, azim=45)
        else:  # Z-up
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z (Up)")
            # Set camera view for Z-up: isometric view (not top-down!)
            ax.view_init(elev=20, azim=45)
        ax.set_title(title)
        ax.set_facecolor(self.background_color)

        # Add ground plane (using transformed coordinates)
        self._add_ground_plane(ax, plot_positions, up_axis)

        return ax

    def _set_3d_axes_equal(self, ax: plt.Axes, points: np.ndarray):
        """Set equal aspect ratio for 3D plot."""
        # Get data bounds
        x_bounds = [points[:, 0].min(), points[:, 0].max()]
        y_bounds = [points[:, 1].min(), points[:, 1].max()]
        z_bounds = [points[:, 2].min(), points[:, 2].max()]

        # Find max range
        max_range = max(x_bounds[1] - x_bounds[0], y_bounds[1] - y_bounds[0], z_bounds[1] - z_bounds[0]) / 2.0

        # Set bounds
        mid_x = (x_bounds[0] + x_bounds[1]) * 0.5
        mid_y = (y_bounds[0] + y_bounds[1]) * 0.5
        mid_z = (z_bounds[0] + z_bounds[1]) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    def _add_ground_plane(self, ax: plt.Axes, points: np.ndarray, up_axis: str = "y"):
        """Add ground plane to 3D plot.

        Args:
            ax: Matplotlib 3D axes
            points: Joint positions array (already transformed for plotting)
            up_axis: 'y' for Y-up, 'z' for Z-up coordinate system
        """
        if up_axis.lower() == "y":
            # Y-up coordinate system: ground plane is XY at minimum Z (since we swapped Y<->Z for plotting)
            # In the transformed plot coordinates, the original Y (up) is now Z (up in plot)
            ground_coord = points[:, 2].min() - 0.1  # Minimum Z in plot (original Y)
            x_range = points[:, 0].max() - points[:, 0].min()
            y_range = points[:, 1].max() - points[:, 1].min()  # This is original Z
            max_range = max(x_range, y_range) * 1.2

            center_x = (points[:, 0].max() + points[:, 0].min()) / 2
            center_y = (points[:, 1].max() + points[:, 1].min()) / 2

            xx, yy = np.meshgrid(
                np.linspace(center_x - max_range / 2, center_x + max_range / 2, 10),
                np.linspace(center_y - max_range / 2, center_y + max_range / 2, 10),
            )
            zz = np.full_like(xx, ground_coord)

            ax.plot_surface(xx, yy, zz, alpha=0.3, color=self.ground_color, shade=True)
        else:  # Z-up coordinate system
            # Z-up coordinate system: ground plane is XY at minimum Z
            ground_coord = points[:, 2].min() - 0.1
            x_range = points[:, 0].max() - points[:, 0].min()
            y_range = points[:, 1].max() - points[:, 1].min()
            max_range = max(x_range, y_range) * 1.2

            center_x = (points[:, 0].max() + points[:, 0].min()) / 2
            center_y = (points[:, 1].max() + points[:, 1].min()) / 2

            xx, yy = np.meshgrid(
                np.linspace(center_x - max_range / 2, center_x + max_range / 2, 10),
                np.linspace(center_y - max_range / 2, center_y + max_range / 2, 10),
            )
            zz = np.full_like(xx, ground_coord)

            ax.plot_surface(xx, yy, zz, alpha=0.3, color=self.ground_color, shade=True)

    def render_motion_frame(
        self,
        motion: Motion,
        frame_idx: int,
        connections: list[tuple[int, int]] | None = None,
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        """Render a single motion frame.

        Args:
            motion: Motion instance
            frame_idx: Frame index to render
            connections: Optional skeleton connectivity
            save_path: Optional path to save image

        Returns:
            Matplotlib figure
        """
        # Get joint positions for this frame
        joint_positions = motion.positions  # (frames, joints, 3)
        frame_positions = joint_positions[frame_idx]

        # Use skeleton connections if available and not overridden
        if connections is None and hasattr(motion.skeleton, "get_bone_connections"):
            connections = motion.skeleton.get_bone_connections()
        elif connections is None:
            connections = self.default_skeleton_connections

        # Create plot
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax = fig.add_subplot(111, projection="3d")

        # Get coordinate system info from motion if available
        up_axis = getattr(motion, "up_axis", "y")

        # Plot skeleton
        self.plot_skeleton_3d(
            frame_positions,
            connections,
            ax,
            f"Motion Frame {frame_idx} (t={frame_idx / motion.framerate:.2f}s)",
            up_axis,
        )

        # Save if path provided
        if save_path is not None:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        return fig

    def create_motion_animation(
        self,
        motion: Motion,
        connections: list[tuple[int, int]] | None = None,
        interval: int = 33,  # ~30 FPS
        save_path: str | Path | None = None,
    ) -> FuncAnimation:
        """Create animated motion visualization.

        Args:
            motion: Motion instance
            connections: Optional skeleton connectivity
            interval: Animation interval in milliseconds
            save_path: Optional path to save animation

        Returns:
            Matplotlib animation object
        """
        # Use skeleton connections if available and not overridden
        if connections is None and hasattr(motion.skeleton, "get_bone_connections"):
            connections = motion.skeleton.get_bone_connections()
        elif connections is None:
            connections = self.default_skeleton_connections

        # Get all joint positions
        joint_positions = motion.positions  # (frames, joints, 3)

        # Get coordinate system info from motion if available
        up_axis = getattr(motion, "up_axis", "y")

        # Transform coordinates for proper axis orientation
        if up_axis.lower() == "y":
            # Swap Y and Z to make Y the vertical axis in matplotlib
            joint_positions = joint_positions[:, :, [0, 2, 1]]  # X, Y, Z -> X, Z, Y for plotting

        # Create figure and axis
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax = fig.add_subplot(111, projection="3d")

        # Initialize plot elements
        joint_scatter = ax.scatter([], [], [], c=self.joint_color, s=self.joint_size)
        bone_lines = []

        # Create line objects for bones
        for _ in connections:
            (line,) = ax.plot([], [], [], c=self.bone_color, linewidth=self.bone_width)
            bone_lines.append(line)

        # Set up axes limits (using all frames for proper bounds)
        all_positions = joint_positions.reshape(-1, 3)
        self._set_3d_axes_equal(ax, all_positions)

        # Add ground plane (using transformed coordinates)
        self._add_ground_plane(ax, all_positions, up_axis)

        # Labels and title
        if up_axis.lower() == "y":
            ax.set_xlabel("X")
            ax.set_ylabel("Z")  # Z is plotted as Y
            ax.set_zlabel("Y (Up)")  # Y is plotted as Z (up)
            # Set camera view for Y-up: looking slightly down and angled
            ax.view_init(elev=15, azim=45)
        else:  # Z-up
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z (Up)")
            # Set camera view for Z-up: looking from the side
            ax.view_init(elev=90, azim=0)
        title = ax.set_title("Motion Animation")

        def animate(frame_idx):
            """Animation function for each frame."""
            frame_positions = joint_positions[frame_idx]

            # Update joints
            joint_scatter._offsets3d = (frame_positions[:, 0], frame_positions[:, 1], frame_positions[:, 2])

            # Update bones
            for i, (parent_idx, child_idx) in enumerate(connections):
                if parent_idx >= len(frame_positions) or child_idx >= len(frame_positions):
                    continue

                parent_pos = frame_positions[parent_idx]
                child_pos = frame_positions[child_idx]

                bone_lines[i].set_data_3d(
                    [parent_pos[0], child_pos[0]], [parent_pos[1], child_pos[1]], [parent_pos[2], child_pos[2]]
                )

            # Update title with timing info
            time_seconds = frame_idx / motion.framerate
            title.set_text(f"Motion Animation - Frame {frame_idx}/{motion.num_frames - 1} (t={time_seconds:.2f}s)")

            return [joint_scatter] + bone_lines + [title]

        # Create animation
        anim = FuncAnimation(
            fig,
            animate,
            frames=motion.num_frames,
            interval=interval,
            blit=False,  # Blitting doesn't work well with 3D plots
            repeat=True,
        )

        # Save animation if path provided
        if save_path is not None:
            save_path = Path(save_path)

            if save_path.suffix.lower() == ".gif":
                anim.save(save_path, writer="pillow", fps=1000 // interval)
            elif save_path.suffix.lower() == ".mp4":
                anim.save(save_path, writer="ffmpeg", fps=1000 // interval)
            else:
                raise ValueError("Unsupported animation format. Use .gif or .mp4")

        return anim

    def create_motion_comparison(
        self,
        motions: dict[str, Motion],
        frame_idx: int,
        connections: list[tuple[int, int]] | None = None,
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        """Create side-by-side comparison of multiple motions.

        Args:
            motions: Dictionary of motion_name -> motion_object
            frame_idx: Frame index to compare
            connections: Optional skeleton connectivity
            save_path: Optional path to save image

        Returns:
            Matplotlib figure
        """
        num_motions = len(motions)

        # Calculate subplot layout
        cols = min(num_motions, 3)  # Max 3 columns
        rows = (num_motions + cols - 1) // cols

        fig = plt.figure(figsize=(self.figsize[0] * cols, self.figsize[1] * rows), dpi=self.dpi)

        for i, (motion_name, motion) in enumerate(motions.items()):
            ax = fig.add_subplot(rows, cols, i + 1, projection="3d")

            # Get joint positions for this frame
            joint_positions = motion.positions
            frame_positions = joint_positions[frame_idx]

            # Plot skeleton
            self.plot_skeleton_3d(frame_positions, connections, ax, f"{motion_name}\nFrame {frame_idx}")

        plt.tight_layout()

        # Save if path provided
        if save_path is not None:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        return fig

    def plot_motion_trajectory(
        self,
        motion: Motion,
        joint_indices: list[int] | None = None,
        save_path: str | Path | None = None,
    ) -> plt.Figure:
        """Plot motion trajectory over time.

        Args:
            motion: Motion instance
            joint_indices: List of joint indices to plot (default: root joint)
            save_path: Optional path to save image

        Returns:
            Matplotlib figure
        """
        if joint_indices is None:
            joint_indices = [0]  # Root joint

        joint_positions = motion.positions  # (frames, joints, 3)

        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax = fig.add_subplot(111, projection="3d")

        # Plot trajectories
        colors = plt.cm.tab10(np.linspace(0, 1, len(joint_indices)))

        for i, joint_idx in enumerate(joint_indices):
            if joint_idx >= joint_positions.shape[1]:
                continue

            trajectory = joint_positions[:, joint_idx, :]

            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                trajectory[:, 2],
                color=colors[i],
                linewidth=2,
                alpha=0.8,
                label=f"Joint {joint_idx}",
            )

            # Mark start and end points
            ax.scatter(*trajectory[0], color=colors[i], s=100, marker="o", alpha=0.8)
            ax.scatter(*trajectory[-1], color=colors[i], s=100, marker="s", alpha=0.8)

        # Set labels and legend
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Motion Trajectory")
        ax.legend()

        # Set equal aspect ratio
        all_points = joint_positions[:, joint_indices, :].reshape(-1, 3)
        self._set_3d_axes_equal(ax, all_points)

        # Save if path provided
        if save_path is not None:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        return fig
