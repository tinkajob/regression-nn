import random, gc
import numpy as np
import multiprocessing as mp
from utils.utils import *
from modules.normalizer import Normalizer
from modules.network import Network
from utils.config import *

norm = Normalizer()

X_train, y_train, X_validate, y_validate = split_dataset(norm=norm, data_path=data_path, data_split_index=data_split_index, features=features, target=target)
training_dataset = (X_train, y_train)
validation_dataset = (X_validate, y_validate)

population = [Network(network_size) for _ in range(population_size)]

context = mp.get_context("fork")
processes = min(mp.cpu_count(), cpu_cores)
pool = context.Pool(processes=processes)

print("================================\n      STARTING TRAINING\n================================\n")

for generation in range(1, max_generations + 1):
    if training_interrupted:
        break
    
    start = time.perf_counter()
    
    # Create a mini batch of data (to prevent memorization and speed up training)
    indices = np.random.choice(len(X_train), batch_size, replace=False) # Select random rows from dataset
    training_batch = (X_train[indices], y_train[indices])

    try:
        gen_performance = list(pool.imap_unordered(evaluate_child, [(child, training_batch) for child in population]))
    except KeyboardInterrupt:
        print("KEYBOARD INTERRUPT!")
        print("Exiting and saving the best model!")
        training_interrupted = True
        pool.terminate()
        pool.join()

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
    
    end = time.perf_counter()
    print_gen_info(gen=generation, raw_mae=dollar_mae, log_mae=log_scaled_mae, patience_used=gens_without_improvement)
    if generation % 20 == 0:
        print_additional_info(gen=generation, validation_mae=best_model.evaluate(validation_dataset, uses_log_scaling = True)[0], layer_sizes=best_model.get_layer_sizes(), avg_neurons=np.mean([net.get_total_neurons() for net in population]), avg_layers=np.mean([len(net.layers) for net in population]), max_neurons=max(n.get_total_neurons() for n in population), max_layers=max(len(n.layers) for n in population))
        print(f"[TIMER]: {end - start:.4f}s")

    survivors = [network for network, log_mae, raw_mae in gen_performance[:survivors_count]]
    remaining = [network for network, log_mae, raw_mae in gen_performance[elites_count:]]

    mutation_strength *= mutation_strength_decay
    mutation_strength = max(0.05, mutation_strength)
    # The non-survivors are ovverwritten by copies of survivors (which are then mutated)
    for i in range(len(remaining)):
        parent = random.choice(survivors)
        child = parent.clone()
        child.mutate_genes(mutation_rate=mutation_rate, mutation_strength=mutation_strength, new_layer_rate=new_layer_rate, delete_layer_rate=delete_layer_rate, mutate_topology=generation < topology_mutation_treshold, min_layers_count=min_layers, min_layer_size=min_layer_size, max_layer_size=max_layer_size, max_neurons=max_neurons, max_layers_count=max_layers)
        remaining[i] = child

    population = survivors + remaining
    # Rebuild population every few generations
    if generation % 50 == 0:
        population = [net.clone() for net in population]
        gc.collect()

validation_mae, raw_mae = best_model.evaluate(validation_dataset, uses_log_scaling = True)
print_validation_info(validation_mae=validation_mae, raw_mae=raw_mae)

best_model_genes = best_model.get_genes()
layer_sizes = [len(features)] + best_model.get_layer_sizes()
metrics = get_model_metrics(generation=last_gen, validation_mae=validation_mae, layer_sizes=layer_sizes, means=norm.means, stds=norm.stds)

model_name = get_model_name(model_name)
save_model(best_model_genes, metrics, parameters, model_name)