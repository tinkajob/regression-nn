import pandas, random
import numpy as np
from utils.utils import *
from modules.normalizer import Normalizer
from modules.network import Network

args = parse_args()
model_name = args.model_name

parameters = load_json(os.path.join("training_parameters.json"))

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

norm = Normalizer()

# We load the dataset, then we clean it
df = pandas.read_csv("houses.csv")
df = df.dropna()
df = df[df["price"] > 0]

training_data = df[:3200]
validation_data = df[3200:]

# We 'configure' normalizer only on training set, not on test set, and use only this configuration to normalize BOTH subsets (so that they are normalized in the same way)
# We don't normalize the price as we use log-scaling!
norm.fit(df[:3200], features) 
training_data = norm.transform(training_data, features)
validation_data = norm.transform(validation_data, features)

# This are lists of tuples: ([a, b, c, d, e, f], g)
training_dataset = [(row[features].values.tolist(), np.log1p(row[target[0]])) for _, row in training_data.iterrows()]
validation_dataset = [(row[features].values.tolist(), np.log1p(row[target[0]])) for _, row in validation_data.iterrows()]

network_size = [len(features), 32, 16, 1]
population = [Network(network_size) for _ in range(population_size)]

best_model_score = 99999999999
gens_without_improvement = 0
last_gen = 0

print("================================\n      STARTING TRAINING\n================================\n")

# Training loop
for generation in range(1, max_generations + 1):
    gen_performance = []
    for child in population:
        # This mae should be as low as possible
        log_mae, raw_mae = child.evaluate(training_dataset, uses_log_scaling = True)
        gen_performance.append((child, log_mae, raw_mae))

    # After we have finished training a generation, we sort networks by their performance
    gen_performance.sort(key = lambda x:x[sort_key])
    last_gen = generation

    # To keep track of the best model so far
    if gen_performance[0][sort_key] < best_model_score:
        best_model = gen_performance[0][0]
        best_model_score = gen_performance[0][sort_key]
        log_scaled_mae = gen_performance[0][1]
        dollar_mae = gen_performance[0][2]
        gens_without_improvement = 0
    else:
        gens_without_improvement += 1
        if gens_without_improvement >= patience:
            print("MODEL EXCEEDED PATIENCE MAXIMUM!\nSTOPPING NOW")
            break

    print(f"COMPLETED TRAINING GENERATION: {generation}")
    print(f"    - Best MAE (dollars): {dollar_mae:,.2f}")
    print(f"    - Best MAE (log-scaled): {log_scaled_mae:,.10f}")
    print(f"    - Patience used: {gens_without_improvement}\n")

    survivors = [network for network, log_mae, raw_mae in gen_performance[:survivors_count]]
    remaining = [network for network, log_mae, raw_mae in gen_performance[survivors_count:]]

    # The non-survivors are ovverwritten by copies of new mutations of survivors
    for child in remaining:
        parent = random.choice(survivors)
        child.set_genes(parent.get_genes())                                                                             
        child.mutate_genes(mutation_rate = mutation_rate, mutation_strength = min(0.01, mutation_strength * (mutation_strength_decay ** generation)), new_layer_rate=new_layer_rate, delete_layer_rate=delete_layer_rate, mutate_topology=generation > topology_mutation_treshold)
    
    population = survivors + remaining

print("================================\n      VALIDATING MODEL\n================================\n")
validation_mae, raw_validation_mae = best_model.evaluate(validation_dataset, uses_log_scaling = True)
print(f"MODEL'S PERFORMANCE:")
print(f"    - Dollars MAE: {raw_validation_mae:,.2f}")
print(f"    - Log-scaled MAE: {validation_mae:,.10f}")

best_model_genes = best_model.get_genes()
metrics = {
    "timestamp": datetime.now().isoformat(timespec="seconds").replace(":", "-"), 
    "generation": last_gen,
    "MAE": validation_mae,
    "layer_sizes": [len(features)] + best_model.get_layer_sizes(),
    "normalization": {
        "means": norm.means,
        "stds": norm.stds
    }
}

model_name = get_model_name(model_name)
save_model(best_model_genes, metrics, parameters, model_name)

# Command for running container
# docker images, docker rmi [name]
# docker build -t house_model .
# docker run -d --rm --name house_train --user 1000:1000 -v /home/tinkajob/programming/house_prices:/app -v /home/tinkajob/programming/house_prices/models:/app/models house_model --model-name [model_name]