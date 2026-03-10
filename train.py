import random, copy
import numpy as np
from utils.utils import *
from modules.normalizer import Normalizer
from modules.network import Network
from utils.config import *

norm = Normalizer()

X_train, y_train, X_validate, y_validate = split_dataset(norm=norm, data_path=data_path, data_split_index=data_split_index, features=features, target=target)
training_dataset = (X_train, y_train)
validation_dataset = (X_validate, y_validate)

population = [Network(network_size) for _ in range(population_size)]

print("================================\n      STARTING TRAINING\n================================\n")

# Training loop
for generation in range(1, max_generations + 1):
    if training_interrupted:
        break
    
    # Create a mini batch of data (to prevent memorization and speed up training)
    indices = np.random.choice(len(X_train), batch_size, replace=False) # Select random rows from dataset
    training_batch = (X_train[indices], y_train[indices])

    gen_performance = []
    for child in population:
        try:
            # This mae should be as low as possible
            log_mae, raw_mae = child.evaluate(training_batch, uses_log_scaling = True)
            gen_performance.append((child, log_mae, raw_mae))
        
        except KeyboardInterrupt:
            print("KEYBOARD INTERRUPT!")
            print("Exiting and saving the best model!")
            training_interrupted = True
            break

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
    print(f"    - Patience used: {gens_without_improvement}")
    
    if generation % 20 == 0:
        best_model_validation_mae = best_model.evaluate(validation_dataset, uses_log_scaling = True)[0]
        print(f"    - Validation MAE (of best model): {best_model_validation_mae:,.10f}")
        print(f"    - Layer sizes of the best model: {best_model.get_layer_sizes()}")
    print()

    survivors = [network for network, log_mae, raw_mae in gen_performance[:survivors_count]]
    remaining = [network for network, log_mae, raw_mae in gen_performance[elites_count:]]

    mutation_strength *= mutation_strength_decay
    mutation_strength = max(0.05, mutation_strength)
    # The non-survivors are ovverwritten by copies of new mutations of survivors
    for i in range(len(remaining)):
        parent = random.choice(survivors)
        child = copy.deepcopy(parent)
        child.mutate_genes(mutation_rate=mutation_rate, mutation_strength=mutation_strength, new_layer_rate=new_layer_rate, delete_layer_rate=delete_layer_rate, mutate_topology=generation < topology_mutation_treshold, min_layers_count=min_layers, min_layer_size=min_layer_size, max_layer_size=max_layer_size, max_neurons=max_neurons, max_layers_count=max_layers)
        remaining[i] = child

    population = survivors + remaining

validation_mae, raw_validation_mae = best_model.evaluate(validation_dataset, uses_log_scaling = True)
print("================================\n      VALIDATING MODEL\n================================\n")
print(f"MODEL'S PERFORMANCE:")
print(f"    - Dollars MAE: {raw_validation_mae:,.2f}")
print(f"    - Log-scaled MAE: {validation_mae:,.10f}")

best_model_genes = best_model.get_genes()
layer_sizes = [len(features)] + best_model.get_layer_sizes()
metrics = get_model_metrics(generation=last_gen, validation_mae=validation_mae, layer_sizes=layer_sizes, means=norm.means, stds=norm.stds)
# metrics = {
#     "timestamp": datetime.now().isoformat(timespec="seconds").replace(":", "-"), 
#     "generation": last_gen,
#     "MAE": validation_mae,
#     "layer_sizes": ,
#     "normalization": {
#         "means": norm.means,
#         "stds": norm.stds
#     }
# }

model_name = get_model_name(model_name)
save_model(best_model_genes, metrics, parameters, model_name)