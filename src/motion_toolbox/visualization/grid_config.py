"""Configuration file support for SpatialAnimationGrid."""

import json
from pathlib import Path
from typing import Any

import yaml


class GridConfig:
    """Configuration for SpatialAnimationGrid visualization."""

    def __init__(self, config_dict: dict[str, Any] | None = None):
        """Initialize configuration from a dictionary.

        Args:
            config_dict: Configuration dictionary
        """
        if config_dict is None:
            config_dict = {}

        # Output settings
        self.output_dir = config_dict.get("output_dir", "outputs/spatial_grid")
        self.output_html = config_dict.get("output_html", "spatial_grid.html")
        self.title = config_dict.get("title", "Spatial Animation Grid")

        # Animation sources
        self.animations = config_dict.get("animations", [])

        # Grid layout
        self.grid_cols = config_dict.get("grid_cols", None)  # Auto-calculate if None
        self.spacing = config_dict.get("spacing", 3.0)

        # Visual settings
        self.material_type = config_dict.get("material_type", "physical")
        self.mesh_color = config_dict.get("mesh_color", "blue")
        self.show_ground = config_dict.get("show_ground", True)
        self.checkerboard_floor = config_dict.get("checkerboard_floor", True)

        # GLB embedding
        self.embed_glbs = config_dict.get("embed_glbs", True)

        # Labels and details
        self.labels = config_dict.get("labels", None)
        self.details = config_dict.get("details", None)

        # Body model path (for Motion objects)
        self.body_model_path = config_dict.get("body_model_path", None)

    @classmethod
    def from_file(cls, config_path: str | Path) -> "GridConfig":
        """Load configuration from a YAML or JSON file.

        Args:
            config_path: Path to configuration file (.yaml, .yml, or .json)

        Returns:
            GridConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If file format is not supported
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Determine file format
        suffix = config_path.suffix.lower()

        if suffix in [".yaml", ".yml"]:
            with open(config_path) as f:
                config_dict = yaml.safe_load(f)
        elif suffix == ".json":
            with open(config_path) as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {suffix}. Use .yaml, .yml, or .json")

        return cls(config_dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration as dictionary
        """
        return {
            "output_dir": self.output_dir,
            "output_html": self.output_html,
            "title": self.title,
            "animations": self.animations,
            "grid_cols": self.grid_cols,
            "spacing": self.spacing,
            "material_type": self.material_type,
            "mesh_color": self.mesh_color,
            "show_ground": self.show_ground,
            "checkerboard_floor": self.checkerboard_floor,
            "embed_glbs": self.embed_glbs,
            "labels": self.labels,
            "details": self.details,
            "body_model_path": self.body_model_path,
        }

    def save(self, output_path: str | Path) -> None:
        """Save configuration to file.

        Args:
            output_path: Path to save configuration (.yaml, .yml, or .json)
        """
        output_path = Path(output_path)
        suffix = output_path.suffix.lower()

        config_dict = self.to_dict()

        if suffix in [".yaml", ".yml"]:
            with open(output_path, "w") as f:
                yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)
        elif suffix == ".json":
            with open(output_path, "w") as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported config file format: {suffix}. Use .yaml, .yml, or .json")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"GridConfig(output_dir='{self.output_dir}', "
            f"animations={len(self.animations)}, "
            f"material='{self.material_type}', "
            f"color='{self.mesh_color}')"
        )
