"""Base converter class for motion representation transformations."""

import os
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch


class BaseMotionConverter(ABC):
    """Abstract base class for converting between different motion representations.

    This class provides a common interface for motion format converters,
    handling file I/O and providing a framework for implementing specific
    motion representation transformations.

    Attributes:
        input_format: String identifier for the input motion format
        output_format: String identifier for the output motion format

    Examples:
        >>> class MyConverter(BaseMotionConverter):
        ...     input_format = 'format_a'
        ...     output_format = 'format_b'
        ...
        ...     def convert(self, data):
        ...         # Transform data from format_a to format_b
        ...         return transformed_data
        >>>
        >>> converter = MyConverter()
        >>> result = converter.process('input.pt', 'output_dir/')
    """

    def __init__(self):
        """Initialize the converter and validate format definitions."""
        if not hasattr(self, "input_format"):
            raise NotImplementedError("Subclasses must define an 'input_format' class attribute")
        if not hasattr(self, "output_format"):
            raise NotImplementedError("Subclasses must define an 'output_format' class attribute")

    @abstractmethod
    def convert(self, data: Any) -> Any:
        """Convert data from input format to output format.

        This method contains the core conversion logic and must be
        implemented by subclasses.

        Args:
            data: Input data in the source format

        Returns:
            Converted data in the target format

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement the convert() method")

    def process(
        self,
        input_data: str | np.ndarray | torch.Tensor | list | dict,
        output_path: str | None = None,
        backup_original: bool = True,
    ) -> tuple[Any, str | None]:
        """Process input data, handling both file paths and direct data inputs.

        This method handles loading data from files, converting it using the
        convert() method, and optionally saving the results.

        Args:
            input_data: Input data as file path, numpy array, torch tensor, list, or dict
            output_path: Directory to save converted data (None to return without saving)
            backup_original: Whether to save a backup of the original data

        Returns:
            Tuple of (converted_data, output_file_path)

        Raises:
            ValueError: If input_data format is not supported
            FileNotFoundError: If input file path doesn't exist
        """
        # Handle file input
        if isinstance(input_data, str):
            if not os.path.isfile(input_data):
                raise FileNotFoundError(f"Input file not found: {input_data}")

            # Load data from file
            try:
                data = torch.load(input_data, weights_only=False)
            except TypeError:
                # Fallback for PyTorch < 1.13 which lacks the weights_only kwarg.
                data = torch.load(input_data)

            # Convert torch tensor to numpy if needed
            if isinstance(data, torch.Tensor):
                data = data.numpy()
            elif not isinstance(data, (list, dict, np.ndarray)):
                # Allow custom objects (like our Motion classes) to pass through
                # The convert() method will handle type validation
                pass

            # Generate output paths if saving
            if output_path is not None:
                parent_dir_name = os.path.basename(os.path.dirname(input_data))
                original_file = f"{parent_dir_name}_{self.input_format}.pt"
                # Use .pkl for complex objects, .npy for simple arrays
                converted_ext = ".pkl" if hasattr(self, "_saves_objects") else ".npy"
                converted_file = f"{parent_dir_name}_{self.output_format}{converted_ext}"
                original_path = os.path.join(output_path, original_file)
                converted_path = os.path.join(output_path, converted_file)
            else:
                original_path = None
                converted_path = None

        # Handle direct data input
        elif isinstance(input_data, (list, dict, np.ndarray, torch.Tensor)):
            data = input_data

            # Convert torch tensor to numpy if needed
            if isinstance(data, torch.Tensor):
                data = data.numpy()

            # Generate default output paths if saving
            if output_path is not None:
                original_path = os.path.join(output_path, f"original_{self.input_format}.pt")
                # Use .pkl for complex objects, .npy for arrays
                converted_ext = ".pkl" if hasattr(self, "_saves_objects") else ".npy"
                converted_path = os.path.join(output_path, f"converted_{self.output_format}{converted_ext}")
            else:
                original_path = None
                converted_path = None
        else:
            raise ValueError("Input data must be a file path, numpy array, torch.Tensor, list, or dict")

        # Convert the data using subclass implementation
        converted_data = self.convert(data)

        # Save data if output path is provided
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)

            if backup_original and original_path is not None:
                torch.save(data, original_path)

            if converted_path is not None:
                # Save converted data - use pickle for complex objects, numpy for arrays
                if isinstance(converted_data, np.ndarray):
                    np.save(converted_path, converted_data)
                else:
                    # Use pickle for complex objects like Motion
                    import pickle

                    pickle_path = converted_path.replace(".npy", ".pkl")
                    with open(pickle_path, "wb") as f:
                        pickle.dump(converted_data, f)
                    # Update the returned path
                    converted_path = pickle_path

        return converted_data, converted_path

    def __str__(self) -> str:
        """String representation of the converter."""
        return f"{self.__class__.__name__}({self.input_format} -> {self.output_format})"

    def __repr__(self) -> str:
        """Detailed string representation of the converter."""
        return f"{self.__class__.__name__}(input_format='{self.input_format}', output_format='{self.output_format}')"
