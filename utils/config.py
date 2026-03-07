import os
from utils.utils import parse_args, load_json

args = parse_args()
model_name = args.model_name

data_path = "houses.csv"

parameters = load_json(os.path.join("training_parameters.json"))

network_size = parameters["network_size"]
population_size = parameters["population_size"]
survivors_count = parameters["survivors_count"]
max_generations = parameters["max_generations"]

# Pass them into function for mutating!
new_layer_rate = parameters["new_layer_rate"]
delete_layer_rate = parameters["delete_layer_rate"]
new_neuron_rate = parameters["new_neuron_rate"]
delete_neuron_rate = parameters["delete_neuron_rate"]
topology_mutation_treshold = parameters["topology_mutation_treshold"]

features = parameters["features"]
target = parameters["target"]
patience = parameters["patience"]
sort_key = parameters["sort_key"] # 1 - log-scaled MAE, 2 - raw dollars MAE

mutation_rate = parameters["mutation_rate"]
mutation_strength = parameters["mutation_strength"]
mutation_strength_decay = parameters["mutation_strength_decay"]

data_split_index = parameters["data_split_index"]
batch_size = parameters["batch_size"]

max_neurons = parameters["max_neurons"]
max_layer_size = parameters["max_layer_size"]
min_layer_size = parameters["min_layer_size"]
max_layers = parameters["max_layers"]
min_layers = parameters["min_layers"]

best_model_score = 99999999999
gens_without_improvement = 0
last_gen = 0

network_size[0] = len(features)

training_interrupted = False