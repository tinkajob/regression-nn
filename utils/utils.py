import json, os, argparse, sys, pandas, time
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
    # Don't resize if both shapes match
    if matrix.shape == new_shape:
        return matrix
    
    old_rows, old_cols = matrix.shape
    new_rows, new_cols = new_shape

    # If rows changed
    if old_cols == new_cols:
        if new_rows > old_rows:
            extra = np.random.uniform(-1, 1, (new_rows - old_rows, new_cols))
            return np.vstack((matrix, extra))
        else:
            return matrix[:new_rows, :]

    # If columns changed
    if old_rows == new_rows:
        if new_cols > old_cols:
            extra = np.random.uniform(-1, 1, (new_rows, new_cols - old_cols))
            return np.hstack((matrix, extra))
        else:
            return matrix[:, :new_cols]

    # Both changed
    new_matrix = np.random.uniform(-1, 1, new_shape)

    rows = min(old_rows, new_rows)
    cols = min(old_cols, new_cols)

    new_matrix[:rows, :cols] = matrix[:rows, :cols]

    return np.ascontiguousarray(new_matrix)

def resize_vector(vector, new_size):
    old_size = len(vector)
    
    # Don't resize if lengths match
    if old_size == new_size:
        return vector
    
    if new_size > old_size:
        extra = np.random.uniform(-1, 1, new_size - old_size)
        return np.concatenate((vector, extra))
    
    return vector[:new_size]

def get_model_metrics(generation, validation_mae, layer_sizes, means, stds):
    metrics = {
        "timestamp": datetime.now().isoformat(timespec="seconds").replace(":", "-"), 
        "generation": generation,
        "MAE": validation_mae,
        "layer_sizes": layer_sizes,
        "normalization": {
            "means": means,
            "stds": stds
        }
    }
    return metrics

def split_dataset(norm, data_path, data_split_index, features, target):
    # We load the dataset, then we clean it
    df = pandas.read_csv(data_path)
    df = df.dropna()
    df = df[df["price"] > 0]

    split_idx = int(len(df) * data_split_index)
    training_data = df[:split_idx]
    validation_data = df[split_idx:]

    # We 'configure' normalizer only on training set, not on test set, and use only this configuration to normalize BOTH subsets (so that they are normalized in the same way)
    # We don't normalize the price as we use log-scaling!
    norm.fit(training_data, features)
    training_data = norm.transform(training_data, features)
    validation_data = norm.transform(validation_data, features)

    X_train = training_data[features].values
    y_train = np.log1p(training_data[target[0]].values)

    X_validate = validation_data[features].values
    y_validate = np.log1p(validation_data[target[0]].values)

    return X_train, y_train, X_validate, y_validate

def linear(x):
    return x

def leaky_relu(x):
    return np.where(x > 0, x, x * 0.01)

def print_gen_info(gen, raw_mae, log_mae, patience_used):
    print(f"\nCOMPLETED TRAINING GENERATION: {gen}")
    print(f"    - Best MAE (dollars): {raw_mae:,.2f}")
    print(f"    - Best MAE (log-scaled): {log_mae:,.10f}")
    print(f"    - Patience used: {patience_used}")

def print_additional_info(gen, validation_mae, layer_sizes, avg_neurons, avg_layers, max_neurons, max_layers, time):
    if gen % 20 != 0:
        return
    print(f"    - Validation MAE (of best model): {validation_mae:,.10f}")
    print(f"    - Layer sizes of the best model: {layer_sizes}")
    print(f"    - Average neuron count: {avg_neurons}")
    print(f"    - Average layer count: {avg_layers}")
    print(f"    - Max neurons: {max_neurons}")
    print(f"    - Max layers: {max_layers}")
    print(f"    - Time used: {time:.4f}s")

def print_validation_info(validation_mae, raw_mae):
    print("================================\n      VALIDATING MODEL\n================================\n")
    print(f"MODEL'S PERFORMANCE:")
    print(f"    - Dollars MAE: {raw_mae:,.2f}")
    print(f"    - Log-scaled MAE: {validation_mae:,.10f}")

def evaluate_child(args):
    child, traning_batch = args
    log_mae, raw_mae = child.evaluate(traning_batch, uses_log_scaling=True)
    return (child, log_mae, raw_mae)

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"[TIMER] {func.__name__}: {end - start:.4f}s")
        return result
    return wrapper

def worker_loop(networks, task_queue, result_queue):
    while True:
        task = task_queue.get()

        if task is None:
            break

        batch = task

        results = []
        for net in networks:
            log_mae, raw_mae = net.evaluate(batch, uses_log_scaling=True)
            results.append((log_mae, raw_mae))

        result_queue.put(results)