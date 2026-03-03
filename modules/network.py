import random
import numpy as np
from .layer import Layer

class Network:
    def __init__(self, layer_sizes):
        self.layers = []

        # We build layers (hidden have Leaky ReLU, last one is linear)
        for i in range(len(layer_sizes) - 1):
            activation = ((lambda x: x) if i == len(layer_sizes) - 2 else lambda x: np.where(x > 0, x, x * 0.01))
            self.layers.append(Layer(activation=activation, input_size=layer_sizes[i], output_size=layer_sizes[i + 1]))

    # def connect_layers(self):
    #     """For each neuron in every layer of network it assigns correct number of **random** weights."""
    #     prev_size = self.input_size

    #     for layer in self.layers:
    #         for neuron in layer:
    #             neuron.weights = [random.uniform(-1, 1) for _ in range(prev_size)]
    #         prev_size = len(layer)

    # def fix_connections(self):
    #     """Ensures there are the correct number of weights for each neuron in each layer"""
    #     prev_size = self.input_size

    #     for layer in self.layers:
    #         for neuron in layer:
    #             # So that there aren't too many weights
    #             neuron.weights = neuron.weights[:prev_size] 

    #             if len(neuron.weights) < prev_size:
    #                 neuron.weights += [random.uniform(-1, 1) for _ in range(prev_size - len(neuron.weights))]
    #         prev_size = len(layer)

    def add_layer(self, probability:float = 0.0):
        """Add layer to random place in network based on the probability given."""

        if random.random() >= probability or len(self.layers) < 1:
            return
            
        # Where to insert new layer; inserting pushes everything after it 1 index up
        insert_index = random.randint(0, len(self.layers) - 1)
        new_size = random.randint(2, 6)

        # sizes
        prev_output_size = self.layers[insert_index].weights.shape[1]

        if insert_index + 1 < len(self.layers):
            next_output_size = self.layers[insert_index + 1].weights.shape[1]
        else:
            next_output_size = None  # inserting at end

        # create new layer
        new_layer = Layer(
            activation=self.get_activation(insert_index),
            input_size=prev_output_size,
            output_size=new_size
        )

        # insert
        self.layers.insert(insert_index + 1, new_layer)

        # fix next layer input size
        if next_output_size is not None:
            self.layers[insert_index + 2].weights = np.random.uniform(
                -1, 1, (new_size, next_output_size)
            )

    def remove_layer(self, probability:float = 0.0):
        """Remove layer from random place in network based on the probability given. We only do that if there are enough hidden layers"""

        if random.random() >= probability or len(self.layers) < 4:
            return

        remove_index = random.randint(1, len(self.layers) - 2)

        prev_layer = self.layers[remove_index - 1]
        next_layer = self.layers[remove_index + 1]

        prev_output_size = prev_layer.weights.shape[1]
        next_output_size = next_layer.weights.shape[1]

        # reconnect
        next_layer.weights = np.random.uniform(
            -1, 1, (prev_output_size, next_output_size)
        )

        self.layers.pop(remove_index)

    def add_neuron(self, probability = 0.0):
        if len(self.layers) <= 2 or random.random() >= probability:
            return
 
        layer_index = random.randint(1, len(self.layers) - 2)
        layer = self.layers[layer_index]
        input_size, output_size = layer.weights.shape

        new_column = np.random.uniform(-1, 1, (input_size, 1))
        layer.weights = np.hstack([layer.weights, new_column])
        layer.biases = np.append(layer.biases, random.uniform(-1, 1))

        next_layer = self.layers[layer_index + 1]
        next_output = next_layer.weights.shape[1]

        new_row = np.random.uniform(-1, 1, (1, next_output))
        next_layer.weights = np.vstack([next_layer.weights, new_row])

        # layer.append(Layer(biases=random.uniform(-1, 1), activation=self.get_activation(layer_index), weights=[random.uniform(-1, 1) for _ in range(len(self.layers[layer_index - 1]))]))

    def remove_neuron(self, probability = 0.0):
        # If there are no hidden layers or we didn't get chosen we exit
        if len(self.layers) <= 2 or random.random() >= probability:
            return

        layer_index = random.randint(1, len(self.layers) - 2)
        layer = self.layers[layer_index]

        if layer.weights.shape[1] <= 1:
            return

        neuron_index = random.randrange(layer.weights.shape[1])

        # Remove neuron
        layer.weights = np.delete(layer.weights, neuron_index, axis=1)
        layer.biases = np.delete(layer.biases, neuron_index)

        # Update next layer
        next_layer = self.layers[layer_index + 1]

        if next_layer.weights.shape[0] <= 1:
            return

        next_layer.weights = np.delete(next_layer.weights, neuron_index, axis=0)


    def predict(self, inputs):
        values = np.array(inputs)

        for layer in self.layers:
            values = layer.forward(values)
        return values

    def evaluate(self, dataset, uses_log_scaling:bool = False):
        X, y = dataset
        X = np.array(X)
        y = np.array(y)
        max_log_price, min_log_price = 20, -5
        
        predictions = self.predict(X)

        if uses_log_scaling:
            errors = np.abs(predictions - y)

            prediction_clamped = np.clip(predictions, min_log_price, max_log_price)
            target_clamped = np.clip(y, min_log_price, max_log_price)

            raw_errors = np.abs(np.expm1(prediction_clamped) - np.expm1(target_clamped))
            return errors.mean(), raw_errors.mean()
        return np.abs(predictions - y).mean(), None
        
        
        # errors = []
        # raw_errors = []

        # for inputs, target in zip(X, y):
        #     prediction = self.predict(inputs)

        #     if uses_log_scaling:
        #         errors.append(abs(prediction - target))
        #         prediction_clamped = np.clip(prediction, -5, MAX_LOG_PRICE)
        #         target_clamped  = np.clip(target, -5, MAX_LOG_PRICE)
        #         raw_errors.append(abs(np.expm1(prediction_clamped) - np.expm1(target_clamped)))

        # MAE = np.mean(errors)
        # raw_MAE = np.mean(raw_errors)

        # return MAE, raw_MAE

    def get_genes(self):
        genes = []
        for layer in self.layers:
            genes.append({
                "weights": layer.weights.tolist(), 
                "biases": layer.biases.tolist()
            })
        return genes

    def set_genes(self, genes):
        for layer, gene in zip(self.layers, genes):
            layer.weights = np.array(gene["weights"], dtype=np.float32)  # overwrite weights
            layer.biases = np.array(gene["biases"], dtype=np.float32)     # overwrite bias

    def get_activation(self, index, total_layers = 0):
        """Returns the correct activation function based on the rules we set."""
        if index == len(self.layers) - 1:
            return lambda x:x # Linear for the output
        else:
            return lambda x: np.where(x > 0, x, x* 0.01) # Leaky ReLU for the hidden layers (NOT ReLU)
        
    def mutate_genes(self, mutation_rate = 0.1, mutation_strength = 0.1, new_neuron_rate = 0.0, delete_neuron_rate = 0.0, new_layer_rate = 0.0, delete_layer_rate = 0.0, mutate_topology:bool = False):
        """Mutate a given set of genes (weights and biases)."""
        for layer in self.layers:
            # We create a mask (so that we don't update all values, 
            # but approximately the percentage of all, given by mutation_rate)
            # If the random value is below treshold, it is set to True (1), otherwise False (0)
            weights_mask = np.random.rand(*layer.weights.shape) < mutation_rate
            biases_mask = np.random.rand(*layer.biases.shape) < mutation_rate

            # We update the weights by adding original matrix and mutation changes matrix together
            # Mutation changes matrix is matrix of random values (if they are to be updated) and zeroes (if they are to stay the same)
            layer.weights += weights_mask * np.random.uniform(-mutation_strength, mutation_strength, layer.weights.shape)
            layer.biases += biases_mask * np.random.uniform(-mutation_strength, mutation_strength, layer.biases.shape)
            
            # for neuron in layer:
            #     for i in range(len(neuron.weights)):
            #         # Mutate and clamp weights
            #         if random.random() < mutation_rate:
            #             neuron.weights[i] += (random.gauss(0, mutation_strength) 
            #                                   if use_gaussian_dist 
            #                                   else random.uniform(-mutation_strength, mutation_strength))
            #         neuron.weights[i] = max(min(neuron.weights[i], 20.0), -20.0)

            #     # Mutate and clamp bias
            #     if random.random() < mutation_rate:
            #         neuron.bias += (random.gauss(0, mutation_strength) 
            #                         if use_gaussian_dist 
            #                         else random.uniform(-mutation_strength, mutation_strength))
            #     neuron.bias = max(min(neuron.bias, 5.0), -5.0)

        # Optionally we add/remove neurons/layers
        if mutate_topology:
            self.add_neuron(probability=new_neuron_rate)
            self.remove_neuron(probability=delete_neuron_rate)
            self.add_layer(probability=new_layer_rate)
            self.remove_layer(probability=delete_layer_rate)
        
        # self.fix_connections()
    
    def get_layer_sizes(self):
        """Returns a list with sizes of each layer of the network."""
        return [layer.weights.shape[1] for layer in self.layers]