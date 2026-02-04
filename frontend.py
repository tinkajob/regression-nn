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

model = Network(layer_sizes)
model.set_genes(genes)


out = model.layers[-1][0]
print("OUTPUT WEIGHTS COUNT:", len(out.weights))
print("OUTPUT BIAS:", out.bias)
out = model.layers[-1][0]
print("OUTPUT WEIGHT KEYS:", out.weights.keys())

prev = model.layers[-2]
print("PREV LAYER IDS:", [n.id for n in prev])

test1 = model.predict([0.0] * len(parameters["features"]))
test2 = model.predict([10000000.0] * len(parameters["features"]))
print("SANITY:", test1, test2, test1 == test2)


user_input = {
    "bedrooms": int(input("Enter the number of bedrooms: ")),
    "bathrooms": float(input("Enter the number of bathrooms: ")),
    "sqft_living": int(input("Enter the square feet of house: ")),
    "sqft_lot": int(input("Enter the square feet of the lot: ")),
    "floors": float(input("Enter the number of floors: ")),
    "waterfront": int(input("Does the house have a waterfront (0 or 1): ")),
    "view": float(input("Enter the number of the view (0-4): ")),
    "condition": float(input("Enter the number of the condition (0-5): ")),
    "sqft_above": int(input("Enter the number of the square feet above: ")),
    "sqft_basement": int(input("Enter the number of the square feet of the basement: ")),
    "yr_built": int(input("Enter the year it was built: ")),
    "yr_renovated": int(input("Enter the year it was renovated (0 if it wasn't): "))
}

# build ordered + normalized input vector
inputs = []
for feature in parameters["features"]:
    raw = user_input[feature]
    mean = norm.means[feature]
    std = norm.stds[feature]
    inputs.append((raw - mean) / std)

# Normalizing the values with the same means and stds as during training
user_input = normalize_input(user_input, parameters["features"], norm)

prediction = model.predict(user_input)

print("\n========================================")
print(f"Model's prediction: {np.expm1(prediction):,.2f}$ ({prediction})")
print("========================================")