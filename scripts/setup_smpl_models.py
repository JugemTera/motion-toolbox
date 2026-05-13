#!/usr/bin/env python3
"""
SMPL Model Setup Script

This script downloads and sets up SMPL body models for use with the motion toolbox.
It removes chumpy dependencies and organizes models in the expected directory structure.

TODO: check overlap with convert_smpl_models_standalone.py which does similar cleaning.

Usage:
    python setup_smpl_models.py [--clean-only] [--output-dir PATH]
"""

import argparse
import pickle
import sys
import urllib.request
from pathlib import Path

import numpy as np


def download_file(url: str, output_path: Path, description: str = ""):
    """Download a file with progress indication."""
    print(f"Downloading {description}...")
    print(f"  URL: {url}")
    print(f"  Output: {output_path}")

    def show_progress(block_num, block_size, total_size):
        if total_size > 0:
            percent = min(100, (block_num * block_size * 100) // total_size)
            print(f"\r  Progress: {percent}% ({block_num * block_size:,} / {total_size:,} bytes)", end="")

    try:
        urllib.request.urlretrieve(url, output_path, show_progress)
        print("\n  ✓ Download completed")
        return True
    except Exception as e:
        print(f"\n  ❌ Download failed: {e}")
        return False


def clean_chumpy_objects(model_data):
    """Remove Chumpy objects from model data, replacing them with numpy arrays.

    Args:
        model_data: Dictionary containing model parameters

    Returns:
        Dictionary with Chumpy objects replaced by numpy arrays
    """
    cleaned_data = {}
    chumpy_objects_found = []

    for key, value in model_data.items():
        if hasattr(value, "r"):  # Chumpy object has .r attribute containing the actual array
            chumpy_objects_found.append(key)
            cleaned_data[key] = np.array(value.r)
        elif hasattr(value, "full") and callable(value.full):  # Some chumpy objects use .full()
            chumpy_objects_found.append(key)
            cleaned_data[key] = np.array(value.full())
        else:
            cleaned_data[key] = value

    if chumpy_objects_found:
        print(f"    Cleaned chumpy objects: {', '.join(chumpy_objects_found)}")
    else:
        print("    No chumpy objects found")

    return cleaned_data


def process_smpl_model(input_path: Path, output_path: Path):
    """Process a single SMPL model file to remove chumpy dependencies."""
    print(f"  Processing: {input_path.name}")

    try:
        # Try loading with different protocols and encodings
        model_data = None
        load_methods = [
            lambda f: pickle.load(f, encoding="latin1"),
            lambda f: pickle.load(f, encoding="bytes"),
            lambda f: pickle.load(f),
        ]

        for i, load_method in enumerate(load_methods):
            try:
                with open(input_path, "rb") as f:
                    model_data = load_method(f)
                print(f"    Loaded successfully (method {i + 1})")
                break
            except Exception as e:
                if i == len(load_methods) - 1:
                    raise e
                continue

        if model_data is None:
            raise ValueError("Failed to load model with any method")

        print(f"    Model keys: {list(model_data.keys())}")

        # Clean chumpy objects
        cleaned_data = clean_chumpy_objects(model_data)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save cleaned model
        with open(output_path, "wb") as f:
            pickle.dump(cleaned_data, f, protocol=2)

        print(f"    ✓ Saved cleaned model to: {output_path}")
        return True

    except Exception as e:
        print(f"    ❌ Error processing {input_path.name}: {e}")
        return False


def setup_smpl_models(output_dir: Path, clean_only: bool = False):
    """Set up SMPL models by downloading and cleaning them."""

    print("🚀 SMPL Model Setup")
    print("=" * 50)

    # Create directory structure
    smpl_dir = output_dir / "smpl"
    smplx_dir = output_dir / "smplx"
    smpl_dir.mkdir(parents=True, exist_ok=True)
    smplx_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    total_count = 0

    if not clean_only:
        # Note: These URLs are placeholders - in practice, SMPL models need to be downloaded
        # from official sources after registration and agreement to terms
        print("\n📋 SMPL Model Download Information:")
        print("-" * 40)
        print("SMPL models must be downloaded manually from official sources:")
        print("1. SMPL: https://smpl.is.tue.mpg.de/")
        print("2. SMPL-X: https://smpl-x.is.tue.mpg.de/")
        print("3. Register and agree to license terms")
        print("4. Download the model files")
        print("\nAfter downloading, place the files in the input directory and run with --clean-only")
        print("\nExpected file structure:")
        print("  SMPL_NEUTRAL.pkl, SMPL_MALE.pkl, SMPL_FEMALE.pkl")
        print("  SMPLX_NEUTRAL.pkl (or .npz), SMPLX_MALE.pkl, SMPLX_FEMALE.pkl")

    # Look for existing model files to clean
    print("\n🔍 Searching for SMPL models to clean...")

    # Check common locations for existing models
    search_paths = [
        Path.home() / ".motion_toolbox/datasets/body_models",
        Path.home() / ".motion_toolbox" / "datasets" / "body_models",
        output_dir,
    ]

    found_models = []
    for search_path in search_paths:
        if search_path.exists():
            print(f"  Checking: {search_path}")
            for pattern in ["*.pkl", "*.npz"]:
                found_models.extend(list(search_path.glob(pattern)))
                found_models.extend(list(search_path.glob(f"*/{pattern}")))
                found_models.extend(list(search_path.glob(f"*/*/{pattern}")))

    if found_models:
        print(f"  Found {len(found_models)} model files to process")

        print("\n🧹 Cleaning SMPL models...")
        for model_file in found_models:
            total_count += 1

            # Determine output path based on model type
            if "smplx" in model_file.name.lower():
                output_path = smplx_dir / model_file.name
            elif "smpl" in model_file.name.lower():
                output_path = smpl_dir / model_file.name
            else:
                # Default to smpl directory
                output_path = smpl_dir / model_file.name

            if process_smpl_model(model_file, output_path):
                success_count += 1

    else:
        print("  No SMPL model files found in common locations")
        print("\n📥 Manual Setup Required:")
        print("  1. Download SMPL/SMPL-X models from official sources")
        print("  2. Place them in one of these directories:")
        for path in search_paths:
            print(f"     - {path}")
        print("  3. Run this script again with --clean-only")

    print("\n📊 Processing Summary:")
    print(f"  Total files processed: {total_count}")
    print(f"  Successfully cleaned: {success_count}")
    print(f"  Failed: {total_count - success_count}")

    if success_count > 0:
        print("\n✅ Setup completed!")
        print(f"  Clean models saved to: {output_dir}")
        print("  Directory structure:")
        print(f"    {smpl_dir}/ - SMPL models")
        print(f"    {smplx_dir}/ - SMPL-X models")

        # List created files
        all_files = list(smpl_dir.glob("*")) + list(smplx_dir.glob("*"))
        if all_files:
            print("  Created files:")
            for file_path in sorted(all_files):
                rel_path = file_path.relative_to(output_dir)
                print(f"    - {rel_path}")
    else:
        print("\n⚠️  No models were processed successfully")
        print("  Please check that SMPL model files are available and accessible")


def main():
    parser = argparse.ArgumentParser(description="Set up SMPL models for motion toolbox")
    parser.add_argument(
        "--clean-only", action="store_true", help="Only clean existing models, do not attempt downloads"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path.home() / ".motion_toolbox/datasets/body_models_clean"),
        help="Output directory for cleaned models",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()

    print(f"Output directory: {output_dir}")

    try:
        setup_smpl_models(output_dir, args.clean_only)
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
