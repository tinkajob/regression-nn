from modules.network import Network
from modules.normalizer import Normalizer
from utils.utils import *
import numpy as np

norm = Normalizer()

folders = get_folders(os.path.join("models"))

print("===== MODELS AVAILABLE =====")
for folder in folders:
    print(f"  {folder}")
print("===== MODELS AVAILABLE =====\n")

selected_model = input("Please select a model: ")
if selected_model not in folders:
    selected_model = folders[-1]

genes = load_json(os.path.join("models", selected_model, "genes.json"))
metrics = load_json(os.path.join("models", selected_model, "metrics.json"))
parameters = load_json(os.path.join("models", selected_model, "parameters.json"))

layer_sizes = metrics["layer_sizes"]
norm.means = metrics["normalization"]["means"]
norm.stds = metrics["normalization"]["stds"]

min_layer_size = parameters["min_layer_size"]
features = parameters["features"]

model = Network(layer_sizes)
model.set_genes(genes, min_layer_size=min_layer_size)

user_input = {}
for feature in features:
    user_input[feature] = float(input(f"{feature}: "))

# Normalizing the values with the same means and stds as during training
user_input = normalize_input(user_input, parameters["features"], norm)

prediction = model.predict(user_input)
prediction_value = float(np.squeeze(prediction))

print("\n==========================================")
print(f"Model's prediction: {np.expm1(prediction_value):,.2f} ({prediction_value:,.6f})")
print("==========================================")