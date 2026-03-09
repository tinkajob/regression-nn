import json, os, argparse, sys
import numpy as np
from modules.normalizer import Normalizer
from datetime import datetime
from pathlib import Path

def load_json(path):
    with open(path) as file:
        return json.load(file)

def save_model(genes, metrics, parameters, name:str, base_dir = "models"):
    """Saves model info in it's folder, along with the metadata. If the name of the model directory is name if it is given, otherwise timestamp is used."""

    timestamp = datetime.now().isoformat(timespec="seconds").replace(":", "-")
    model_dir = os.path.join(base_dir, name)

    os.makedirs(model_dir, exist_ok = False)

    with open(os.path.join(model_dir, "genes.json"), "w") as f:
        json.dump(genes, f)

    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f)

    with open(os.path.join(model_dir, "parameters.json"), "w") as f:
        json.dump(parameters, f)

def get_folders(path):
    """Lists all folders in given directory."""
    folders = []
    for p in Path(path).iterdir():
        if p.is_dir():
            folders.append(p.name)
    return sorted(folders)

def normalize_input(values:dict, features:list, norm:Normalizer):
    """
    Normalizes a list of raw feature values using the provided Normalizer.

    Parameters:
    - values: list of raw input values, e.g. [3, 2, 2560, ...]
    - features: list of feature names, same order as the model expects
    - norm: Normalizer object that has means and stds

    Returns:
    - List of normalized values, ready for model prediction
    """
    if len(values) != len(features):
        print("ERROR: Lengths of values and features don't match!")
        return []

    normalized = []
    for feature in features:
        if feature not in values:
            print(f"ERROR: feature '{feature}' not in values!")
            normalized.append(0.0)  # fallback to 0 if missing
            continue
        mean = norm.means[feature]
        std = norm.stds[feature]

        if std == 0:
            normalized_value = 0.0
        else:
            normalized_value = (values[feature] - mean) / std

        normalized.append(normalized_value)
    return normalized

def parse_args():
    parser = argparse.ArgumentParser(description="Train house price model")

    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Name of output model (without extension)"
    )

    return parser.parse_args()

def get_model_name(model_name:str = "") -> str:
    """Return the name of the model based on parsed arguments and environment."""

    if model_name:
        return model_name
    
    if sys.stdin.isatty():
        if name := input("How would you like to name your model? (leave empty for default)\n:").strip():
            return name
    
    timestamp = datetime.now().isoformat(timespec="seconds").replace(":", "-")
    return f"model-{timestamp}"

def resize_matrix(matrix, new_shape:tuple[int, int]):
    old_rows, old_cols = matrix.shape
    new_rows, new_cols = new_shape

    # Create a new blank matrix on new_size
    new_matrix = np.random.uniform(-1, 1, new_shape)

    rows = min(old_rows, new_rows)
    cols = min(old_cols, new_cols)
    
    # Copy old matrix into the new one (as much as it fits)
    new_matrix[:rows, :cols] = matrix[:rows, :cols]

    return new_matrix

def resize_vector(vector, new_size):
    new_vector = np.random.uniform(-1, 1, new_size)

    size = min(len(vector), new_size)
    new_vector[:size] = vector[:size]

    return new_vector