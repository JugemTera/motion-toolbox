# This script is inspired by and modified from the original script by Mathis
# Petrovich, available in the SMPL-X repository by V. Choutas.
# https://github.com/vchoutas/smplx/tree/master/tools
# it also uses https://github.com/Mathux/TEMOS as inspiration to convert a full directory of pickled body models.

"""Convert SMPL, SMPL-H, and SMPL-X models from .pkl to .npz format, removing 'chumpy' dependencies.

This script processes a directory of SMPL family models, cleans them of chumpy
dependencies, and saves them in a standardized .npz format.
It expects the models to be in the following structure:
smpl
├── SMPL_FEMALE.pkl
├── SMPL_MALE.pkl
├── SMPL_NEUTRAL.pkl
smplx
├── SMPLX_FEMALE.pkl
├── SMPLX_MALE.pkl
├── SMPLX_NEUTRAL.pkl
smplh
├── SMPLH_FEMALE.pkl
├── SMPLH_MALE.pkl
├── SMPLH_NEUTRAL.pkl
mano  # MANO layout — used by the hand-add path; layout has not yet been re-verified.
└── MANO_LEFT.pkl
└── MANO_RIGHT.pkl

Usage:
    python scripts/convert_smpl_models_standalone.py --input-folder /path/to/raw/models --output-folder /path/to/cleaned/models
"""

import argparse
import os
import os.path as osp
import pickle

import numpy as np


def remove_chumpy_dependency(data_dict):
    """Iterates through a dictionary and converts any chumpy objects into numpy arrays."""
    cleaned_dict = {}
    for key, val in data_dict.items():
        if "chumpy" in str(type(val)):
            cleaned_dict[key] = np.array(val)
        else:
            cleaned_dict[key] = val
    return cleaned_dict


def load_model(path):
    """Loads a model from a .pkl file and removes the chumpy dependency."""
    if not osp.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    _, ext = osp.splitext(path)
    if ext != ".pkl":
        raise TypeError(f"Unsupported file extension: '{ext}'. Must be .pkl.")

    with open(path, "rb") as f:
        import warnings

        warnings.filterwarnings("ignore", category=DeprecationWarning)
        data = pickle.load(f, encoding="latin1")

    return remove_chumpy_dependency(data)


def get_gender_from_filename(filename):
    """Determines the gender from the model filename."""
    fname_lower = filename.lower()
    if "female" in fname_lower:
        return "FEMALE"
    if "male" in fname_lower:
        return "MALE"
    if "neutral" in fname_lower:
        return "NEUTRAL"
    return None


def process_generic_model(model_path, output_path, model_type="SMPL", output_format="npz"):
    """Processes a generic SMPL or SMPLX model and saves in the specified format."""
    model_data = load_model(model_path)
    filename = osp.basename(model_path)
    gender = get_gender_from_filename(filename)

    if gender:
        out_filename = f"{model_type.upper()}_{gender}.{output_format}"
    else:
        out_filename = osp.splitext(filename)[0] + f".{output_format}"
        print(f"Warning: Could not determine gender for {filename}. Saving as {out_filename}")

    out_path = osp.join(output_path, out_filename)
    print(f"-> Saving cleaned {model_type} model to: {out_path}")
    if output_format == "npz":
        np.savez_compressed(out_path, **model_data)
    elif output_format == "pkl":
        with open(out_path, "wb") as f:
            pickle.dump(model_data, f)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def process_smplh_model(smplh_path, mano_left_path, mano_right_path, output_path, output_format="npz"):
    """Loads and merges SMPL-H and MANO models, saves in specified format."""
    body_data = load_model(smplh_path)
    lhand_data = load_model(mano_left_path)
    rhand_data = load_model(mano_right_path)

    filename = osp.basename(smplh_path)
    gender = get_gender_from_filename(filename)
    if gender:
        out_filename = f"SMPLH_{gender}.{output_format}"
    else:
        out_filename = osp.splitext(filename)[0] + f".{output_format}"
        print(f"Warning: Could not determine gender for {filename}. Saving as {out_filename}")

    output_data = body_data.copy()
    output_data["hands_componentsl"] = lhand_data["hands_components"]
    output_data["hands_componentsr"] = rhand_data["hands_components"]
    output_data["hands_coeffsl"] = lhand_data["hands_coeffs"]
    output_data["hands_coeffsr"] = rhand_data["hands_coeffs"]
    output_data["hands_meanl"] = lhand_data["hands_mean"]
    output_data["hands_meanr"] = rhand_data["hands_mean"]

    output_data = remove_chumpy_dependency(output_data)

    out_path = osp.join(output_path, out_filename)
    print(f"-> Saving merged SMPL-H model to: {out_path}")
    if output_format == "npz":
        np.savez_compressed(out_path, **output_data)
    elif output_format == "pkl":
        with open(out_path, "wb") as f:
            pickle.dump(output_data, f)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def main():
    """Main function to parse arguments and run the model processing."""
    parser = argparse.ArgumentParser(
        description="A tool to prepare SMPL, SMPL-H, and SMPLX models from a directory tree by removing the 'chumpy' dependency."
    )
    parser.add_argument(
        "--input-folder", required=True, type=str, help="Path to the folder containing the raw .pkl model files."
    )
    parser.add_argument(
        "--output-folder", required=True, type=str, help="The directory where the prepared models will be saved."
    )
    parser.add_argument(
        "--output-format",
        choices=["npz", "pkl"],
        default="pkl",
        help="Output format for cleaned models (npz or pkl). Default: npz.",
    )
    args = parser.parse_args()

    if not osp.exists(args.output_folder):
        print(f"Output directory not found. Creating: {args.output_folder}")
        os.makedirs(args.output_folder)

    # Traverse all subdirectories to find .pkl files
    model_files = []
    for root, dirs, files in os.walk(args.input_folder):
        for file in files:
            if file.endswith(".pkl"):
                abs_path = osp.join(root, file)
                rel_dir = osp.relpath(root, args.input_folder)
                model_files.append((abs_path, rel_dir, file))

    # Find MANO models in each subdirectory
    processed_count = 0
    for abs_path, rel_dir, filename in model_files:
        fname_upper = filename.upper()
        output_subdir = osp.join(args.output_folder, rel_dir)
        if not osp.exists(output_subdir):
            os.makedirs(output_subdir, exist_ok=True)

        # Look for MANO models in the same subdir for SMPL-H
        mano_left_path = osp.join(osp.dirname(abs_path), "MANO_LEFT.pkl")
        mano_right_path = osp.join(osp.dirname(abs_path), "MANO_RIGHT.pkl")
        has_mano_models = osp.exists(mano_left_path) and osp.exists(mano_right_path)

        if fname_upper.startswith("SMPLH_"):
            if has_mano_models:
                print(f"\nProcessing SMPL-H model: {abs_path}")
                process_smplh_model(
                    abs_path, mano_left_path, mano_right_path, output_subdir, output_format=args.output_format
                )
                processed_count += 1
            else:
                print(f"\nSkipping SMPL-H model {abs_path} due to missing MANO files in {osp.dirname(abs_path)}.")

        elif fname_upper.startswith("SMPL_"):
            print(f"\nProcessing SMPL model: {abs_path}")
            process_generic_model(abs_path, output_subdir, model_type="smpl", output_format=args.output_format)
            processed_count += 1

        elif fname_upper.startswith("SMPLX_"):
            print(f"\nProcessing SMPLX model: {abs_path}")
            process_generic_model(abs_path, output_subdir, model_type="smplx", output_format=args.output_format)
            processed_count += 1

    if processed_count == 0:
        print("\nNo models were processed. Ensure your input folder contains correctly named .pkl files.")
    else:
        print(f"\n✅ Processing complete. {processed_count} models converted.")


if __name__ == "__main__":
    main()
