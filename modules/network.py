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
                        weights=[random.uniform(-1, 1)],
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
        """For each neuron in every layer of network it assigns correct number of *random* weights."""
        prev_size = self.input_size

        for layer in self.layers:
            for neuron in layer:
                neuron.weights = [random.uniform(-1, 1) for _ in range(prev_size)]
            prev_size = len(layer)

    def fix_connections(self):
        """Ensures there are the correct number of weights for each neuron in each layer"""
        prev_size = self.input_size

        for layer in self.layers:
            for neuron in layer:
                # So that there aren't too many weights
                neuron.weights = neuron.weights[:prev_size] 

                if len(neuron.weights) < prev_size:
                    neuron.weights += [random.uniform(-1, 1) for _ in range(prev_size - len(neuron.weights))]
            prev_size = len(layer)

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
            weights = [random.uniform(-1,1) for _ in range(len(prev_layer))]
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
        layer.append(Neuron(bias=random.uniform(-1, 1), activation=self.get_activation(layer_index), weights=[random.uniform(-1, 1) for _ in range(len(self.layers[layer_index - 1]))]))

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
        values = inputs

        for layer in self.layers:
            next_values = []
            for neuron in layer:
                next_values.append(neuron.forward(values))
            values = next_values
        return values[0]

    def evaluate(self, dataset, uses_log_scaling:bool = False):
        X, y = dataset
        errors = []
        raw_errors = []
        MAX_LOG_PRICE = 20

        for inputs, target in zip(X, y):
            prediction = self.predict(inputs)

            if uses_log_scaling:
                errors.append(abs(prediction - target))
                prediction_clamped = np.clip(prediction, -5, MAX_LOG_PRICE)
                target_clamped  = np.clip(target, -5, MAX_LOG_PRICE)
                raw_errors.append(abs(np.expm1(prediction_clamped) - np.expm1(target_clamped)))

        MAE = np.mean(errors)
        raw_MAE = np.mean(raw_errors)

        return MAE, raw_MAE

    def get_genes(self):
        genes = []
        for layer in self.layers:
            layer_genes = []
            for neuron in layer:
                layer_genes.append({
                    "weights": neuron.weights, 
                    "bias": neuron.bias
                    })
            genes.append(layer_genes)
        return genes

    def set_genes(self, genes):
        for layer, layer_genes in zip(self.layers, genes):
            for neuron, gene in zip(layer, layer_genes):
                neuron.weights = gene["weights"].copy()  # overwrite weights
                neuron.bias = gene["bias"]               # overwrite bias
        self.fix_connections()

    def get_activation(self, index, total_layers = 0):
        """Returns the correct activation function based on the rules we set."""
        layers_count = max(total_layers, len(self.layers))

        if index == layers_count - 1:
            return lambda x:x # Linear for the output
        else:
            return lambda x: x if x > 0 else x * 0.01 # Leaky ReLU for the hidden layers (NOT ReLU)
        
    def mutate_genes(self, mutation_rate = 0.1, mutation_strength = 0.1, new_neuron_rate = 0.0, delete_neuron_rate = 0.0, new_layer_rate = 0.0, delete_layer_rate = 0.0, use_gaussian_dist = False, mutate_topology:bool = False):
        """Mutate a given set of genes (weights and biases)."""
        for layer in self.layers:
            for neuron in layer:
                for i in range(len(neuron.weights)):
                    # Mutate and clamp weights
                    if random.random() < mutation_rate:
                        neuron.weights[i] += (random.gauss(0, mutation_strength) 
                                              if use_gaussian_dist 
                                              else random.uniform(-mutation_strength, mutation_strength))
                    neuron.weights[i] = max(min(neuron.weights[i], 20.0), -20.0)

                # Mutate and clamp bias
                if random.random() < mutation_rate:
                    neuron.bias += (random.gauss(0, mutation_strength) 
                                    if use_gaussian_dist 
                                    else random.uniform(-mutation_strength, mutation_strength))
                neuron.bias = max(min(neuron.bias, 5.0), -5.0)

        # Optionally we add/remove neurons/layers
        if mutate_topology:
            self.add_neuron(probability=new_neuron_rate)
            self.remove_neuron(probability=delete_neuron_rate)
            self.add_layer(probability=new_layer_rate)
            self.remove_layer(probability=delete_layer_rate)
        
        self.fix_connections()
    
    def get_layer_sizes(self):
        """Returns a list with sizes of each layer of the network."""
        return [len(layer) for layer in self.layers]