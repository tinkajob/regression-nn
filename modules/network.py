import random
import numpy as np
from .neuron import Neuron

class Network:
    def __init__(self, layer_sizes):
        self.input_size = layer_sizes[0]
        self.layers = []

        for size in layer_sizes[1:]:
            layer = []
            for _ in range(size):
                layer.append(
                    Neuron(
                        weights={},
                        bias=random.uniform(-1, 1),
                        activation=lambda x: max(0, x)
                    )
                )
            self.layers.append(layer)

        # fix output activation
        for neuron in self.layers[-1]:
            neuron.activation = lambda x: x

        self.connect_layers()


    def connect_layers(self):
        prev_ids = list(range(self.input_size))

        for layer in self.layers:
            for neuron in layer:
                neuron.weights = {
                    pid: random.uniform(-1, 1)
                    for pid in prev_ids
                }
            prev_ids = [n.id for n in layer]

    def fix_connections(self):
        for i in range(1, len(self.layers)):
            prev_ids = {neuron.id for neuron in self.layers[i - 1]}
            for neuron in self.layers[i]:
                
                # Remove dead inputs
                neuron.weights = {key: value for key, value in neuron.weights.items() if key in prev_ids}
                
                # Add missing inputs
                for prev in self.layers[i - 1]:
                    if prev.id not in neuron.weights:
                        neuron.weights[prev.id] = random.uniform(-1, 1)

    def add_layer(self, probability:float = 0.0):
        """Add layer to random place in network based on the probability given."""

        if random.random() >= probability:
            return
            
        # Where to insert new layer; inserting pushes everything after it 1 index up
        insert_index = random.randint(1, len(self.layers) - 1)
        new_layer_size = random.randint(2, 6) #IDK just random numbers
        prev_layer = self.layers[insert_index - 1]

        new_layer = []
        for _ in range(new_layer_size):
            weights = {neuron.id: random.uniform(-1,1) for neuron in prev_layer}
            bias = random.uniform(-1,1)
            new_layer.append(Neuron(weights=weights, bias=bias, activation=self.get_activation(insert_index, len(self.layers) + 1)))
        self.layers.insert(insert_index, new_layer)
        self.fix_connections()

    def remove_layer(self, probability:float = 0.0):
        """Remove layer from random place in network based on the probability given. We only do that if there are enough hidden layers"""
        if len(self.layers) < 4:
            return

        if random.random() >= probability:
            return
        
        remove_index = random.randint(1, len(self.layers) - 2)
        self.layers.pop(remove_index)
        self.fix_connections()

    def add_neuron(self, probability = 0.0):
        if len(self.layers) <= 2 or random.random() >= probability:
            return
 
        layer_index = random.randint(1, len(self.layers) - 2)
        layer = self.layers[layer_index]
        layer.append(Neuron(bias=random.uniform(-1, 1), activation=self.get_activation(layer_index)))

    def remove_neuron(self, probability = 0.0):
        # If there are no hidden layers or we didn't get chosen we exit
        if len(self.layers) <= 2 or random.random() >= probability:
            return
        
        layer_index = random.randint(1, len(self.layers) - 2)
        layer = self.layers[layer_index]

        if len(layer) <= 1:
            return
        
        remove_index = random.randrange(len(layer))
        layer.pop(remove_index)

    def predict(self, inputs):
        values = {i: v for i, v in enumerate(inputs)}

        for layer in self.layers:
            next_values = {}
            for neuron in layer:
                next_values[neuron.id] = neuron.forward(values)
            values = next_values

        return next(iter(values.values()))


    def evaluate(self, dataset:list[tuple[list[float], float]], uses_log_scaling:bool = False):
        errors = []
        raw_errors = []
        predictions = []
        MAX_LOG_PRICE = 25

        for inputs, target in dataset:
            pred = self.predict(inputs)
            predictions.append(pred)

            if uses_log_scaling:
                errors.append(abs(pred - target))
                pred_c = np.clip(pred, -5, MAX_LOG_PRICE)
                tgt_c  = np.clip(target, -5, MAX_LOG_PRICE)
                raw_errors.append(abs(np.expm1(pred_c) - np.expm1(tgt_c)))

        MAE = sum(errors) / len(errors)
        raw_MAE = sum(raw_errors) / len(raw_errors)

        preds = np.clip(np.array(predictions), -5, MAX_LOG_PRICE)

        # variance penalty
        raw_preds = np.expm1(preds)
        log_raw_preds = np.log1p(raw_preds)
        
        pred_var = np.var(log_raw_preds)
        if random.random() < 0.001: #SANITY CHECK, DEBUG
            print("pred std:", np.std(log_raw_preds))


        if not np.isfinite(pred_var):
            return float("inf"), float("inf")

        min_var = 0.01  # tune
        penalty = max(0.0, (min_var - pred_var) / min_var)
        MAE += 5.0 * penalty

        return MAE, raw_MAE

    def get_genes(self): #IDK
        genes = []
        for layer in self.layers:
            layer_genes = []
            for neuron in layer:
                layer_genes.append({"id": neuron.id, "weights": neuron.weights, "bias": neuron.bias})
            genes.append(layer_genes)
        return genes

    def set_genes(self, genes):
        for layer, layer_genes in zip(self.layers, genes):
            for neuron, gene in zip(layer, layer_genes):
                neuron.id = gene["id"]
                neuron.weights = gene["weights"].copy()  # overwrite weights
                neuron.bias = gene["bias"]               # overwrite bias
        Neuron.id_counter = max(neuron.id for layer in self.layers for neuron in layer) + 1 # Resetting id counter
        self.fix_connections()

    def get_activation(self, index, total_layers = 0):
        """Returns the correct activation function based on the rules we set."""
        all_layers = max(total_layers, len(self.layers))

        if index == all_layers - 1:
            return lambda x:x # Linear for the output
        else:
            return lambda x: x if x > 0 else x * 0.01 # Leaky ReLU for the hidden layers
        
    def mutate_genes(self, current_gen:int, mutation_rate = 0.1, mutation_strength = 0.1, new_neuron_rate = 0.0, delete_neuron_rate = 0.0, new_layer_rate = 0.0, delete_layer_rate = 0.0, use_gaussian_dist = False):
        """
        Mutate a given set of genes (weights and biases) and return a new set of genes.
        
        Parameters:
        - genes: list of layers, each layer is a list of [weights, bias] pairs
        - mutation_rate: probability of mutating each weight/bias
        - mutation_strength: maximum mutation magnitude
        - use_gaussian_dist: if True, use Gaussian mutation; otherwise uniform
        
        Returns:
        - new_genes: mutated copy of the input genes
        """
        for layer in self.layers:
            for neuron in layer:
                for key in neuron.weights:
                    # Mutate weights
                    if random.random() < mutation_rate:
                        if use_gaussian_dist:
                            neuron.weights[key] += random.gauss(0, mutation_strength)
                        else:
                            neuron.weights[key] += random.uniform(-mutation_strength, mutation_strength)
                    # Clamp weight
                    neuron.weights[key] = max(min(neuron.weights[key], 5.0), -5.0)

                # Mutate bias
                if random.random() < mutation_rate:
                    if use_gaussian_dist:
                        neuron.bias += random.gauss(0, mutation_strength)
                    else:
                        neuron.bias += random.uniform(-mutation_strength, mutation_strength)
                # Clamp bias
                neuron.bias = max(min(neuron.bias, 5.0), -5.0)

        # Optionally we add/remove neurons/layers
        if current_gen > 100: # Arbitrary
            self.add_neuron(probability=new_neuron_rate)
            self.remove_neuron(probability=delete_neuron_rate)
            self.add_layer(probability=new_layer_rate)
            self.remove_layer(probability=delete_layer_rate)
        
        self.fix_connections()
    
    def get_layer_sizes(self):
        """Returns a list with sizes of each layer of the network."""
        return [len(layer) for layer in self.layers]