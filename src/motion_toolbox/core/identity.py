"""Identity class for representing person-specific attributes in motion data.

This module provides the Identity class that encapsulates physical and demographic
characteristics of the person performing the motion, such as body shape parameters,
gender, age, and other identity-related properties.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Identity:
    """Represents the identity/physical characteristics of a person in motion data.

    This class encapsulates person-specific attributes that affect how motion should
    be visualized or processed, particularly for SMPL-based body models.

    Args:
        betas: SMPL shape parameters (10,) defining body morphology (height, weight, proportions)
        gender: Gender for model selection ("neutral", "male", "female")
        age: Age category ("adult", "child") - for future extensions
        name: Optional identifier/name for the identity

    Examples:
        >>> import numpy as np
        >>> from motion_toolbox import Identity
        >>>
        >>> # Default identity (average body shape)
        >>> default_identity = Identity()
        >>>
        >>> # Custom body shape
        >>> custom_shape = np.array([0.5, -0.2, 0.1, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0])
        >>> identity = Identity(betas=custom_shape, gender="female", name="Person1")
        >>>
        >>> # Access properties
        >>> print(f"Gender: {identity.gender}")
        >>> print(f"Shape parameters: {identity.betas}")
    """

    betas: np.ndarray | None = None  # SMPL shape parameters (10,)
    gender: str | None = "neutral"  # Gender for SMPL models
    age: str | None = "adult"  # Age category
    name: str | None = None  # Optional identifier

    def __post_init__(self):
        """Validate identity parameters."""
        # Validate betas if provided
        if self.betas is not None:
            if not isinstance(self.betas, np.ndarray):
                object.__setattr__(self, "betas", np.array(self.betas))
            if self.betas.shape != (10,):
                raise ValueError(f"betas must have shape (10,), got {self.betas.shape}")
            # Make betas read-only
            self.betas.flags.writeable = False

        # Validate gender
        valid_genders = ["neutral", "male", "female"]
        if self.gender.lower() not in valid_genders:
            raise ValueError(f"gender must be one of {valid_genders}, got '{self.gender}'")

        # Validate age
        valid_ages = ["adult", "child"]
        if self.age.lower() not in valid_ages:
            raise ValueError(f"age must be one of {valid_ages}, got '{self.age}'")

    @classmethod
    def create_default(cls) -> "Identity":
        """Create a default identity with neutral characteristics.

        Returns:
            Identity with default/average characteristics
        """
        return cls(
            betas=np.zeros(10),  # Average body shape
            gender="neutral",
            age="adult",
            name="Default",
        )

    @classmethod
    def create_random(cls, gender: str | None = None, seed: int | None = None) -> "Identity":
        """Create an identity with random body shape parameters.

        Args:
            gender: Optional gender override, otherwise randomly selected
            seed: Random seed for reproducibility

        Returns:
            Identity with randomized shape parameters
        """
        if seed is not None:
            np.random.seed(seed)

        # Generate random shape parameters (typically in range [-3, 3])
        betas = np.random.normal(0, 1.0, 10).astype(np.float32)
        betas = np.clip(betas, -3.0, 3.0)  # Reasonable range for SMPL

        # Random gender if not specified
        if gender is None:
            gender = np.random.choice(["neutral", "male", "female"])

        return cls(betas=betas, gender=gender, name=f"Random_{gender}")

    def get_smpl_params(self) -> dict:
        """Get SMPL-compatible parameters from this identity.

        Returns:
            Dictionary with SMPL parameters
        """
        return {"betas": self.betas if self.betas is not None else np.zeros(10), "gender": self.gender, "age": self.age}

    def to_dict(self) -> dict:
        """Convert identity to dictionary representation.

        Returns:
            Dictionary representation suitable for serialization
        """
        return {
            "betas": self.betas.tolist() if self.betas is not None else None,
            "gender": self.gender,
            "age": self.age,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Identity":
        """Create identity from dictionary representation.

        Args:
            data: Dictionary with identity data

        Returns:
            Identity instance
        """
        betas = None
        if data.get("betas") is not None:
            betas = np.array(data["betas"])

        return cls(
            betas=betas, gender=data.get("gender", "neutral"), age=data.get("age", "adult"), name=data.get("name")
        )

    def __str__(self) -> str:
        """String representation of the identity."""
        shape_info = "default" if self.betas is None else f"custom({np.linalg.norm(self.betas):.2f})"
        name_info = f"'{self.name}'" if self.name else "unnamed"
        return f"Identity({name_info}, {self.gender}, {self.age}, shape={shape_info})"
