import random
import numpy as np
from .layer import Layer
from utils.utils import resize_matrix, resize_vector, linear, leaky_relu

class Network:
    def __init__(self, layer_sizes):
        self.layers = []

        # We build layers (hidden have Leaky ReLU, last one is linear)
        for i in range(len(layer_sizes) - 1):
            activation = self.get_activation(i == len(layer_sizes) - 2)
            self.layers.append(Layer(activation=activation, input_size=layer_sizes[i], output_size=layer_sizes[i + 1]))

    def add_layer(self, probability:float = 0.0, max_neurons = 128, max_layers_count = 8, min_layer_size = 2):
        """Add layer to random place in network based on the probability given."""
        # Don't add layer if the probability isn't selected,
        # if there is already max number of layers
        # or if new layer would add too many neurons
        if random.random() >= probability:
            return

        if len(self.layers) >= max_layers_count:
            return
        
        new_size = random.randint(min_layer_size, 6)
        if self.get_total_neurons() + new_size >= max_neurons:
            return
            
        # Where to insert new layer; inserting pushes everything after it 1 index up
        insert_index = random.randint(0, len(self.layers) - 2)
        prev_output_size = self.layers[insert_index].weights.shape[1]

        # create new layer
        new_layer = Layer(
            activation=self.get_activation(is_ouput=False),
            input_size=prev_output_size,
            output_size=new_size
        )

        self.layers.insert(insert_index + 1, new_layer)

    def remove_layer(self, min_layers_count:int = 4, probability:float = 0.0, min_layer_size=2):
        """Remove layer from random place in network based on the probability given. We only do that if there are enough hidden layers"""
        # Don't remove a layer if probability isn't selected or if there are no hidden layers
        if random.random() >= probability or len(self.layers) <= min_layers_count:
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

    def add_neuron(self, probability = 0.0, max_neurons = 128, max_layer_size = 40):
        # Don't add neurons if there are no hidden layers,
        # if probability isn't selected,
        # if there are already max number of neurons
        # or if the layer is alreaady max size
        if len(self.layers) <= 2:
            return
        
        if random.random() >= probability:
            return
        
        if self.get_total_neurons() >= max_neurons:
            return
        
        layer_index = random.randint(1, len(self.layers) - 2)
        layer = self.layers[layer_index]
        
        if layer.weights.shape[1] >= max_layer_size:
            return
 
        input_size, output_size = layer.weights.shape

        new_column = np.random.uniform(-1, 1, (input_size, 1))
        layer.weights = np.hstack([layer.weights, new_column])
        layer.biases = np.append(layer.biases, random.uniform(-1, 1))

        next_layer = self.layers[layer_index + 1]
        next_output = next_layer.weights.shape[1]

        new_row = np.random.uniform(-1, 1, (1, next_output))
        next_layer.weights = np.vstack([next_layer.weights, new_row])

    def remove_neuron(self, probability = 0.0, min_layer_size = 2):
        if random.random() >= probability:
            return
        
        layer_index = random.randint(1, len(self.layers) - 2)
        layer = self.layers[layer_index]

        # If there are no hidden layers exit
        if layer.weights.shape[1] <= min_layer_size:
            return

        neuron_index = random.randrange(layer.weights.shape[1])

        # Update next layer
        next_layer = self.layers[layer_index + 1]

        if next_layer.weights.shape[0] <= min_layer_size:
            return

        # Remove neuron
        layer.weights = np.delete(layer.weights, neuron_index, axis=1)
        layer.biases = np.delete(layer.biases, neuron_index)

        next_layer.weights = np.delete(next_layer.weights, neuron_index, axis=0)

    def repair_network(self, min_layer_size = 2):
        """
        Ensures every layer's input size matches
        the previous layer's output size.
        """
        if len(self.layers) <= 2:
            return

        for i in range(1, len(self.layers) - 1):
            prev_output = self.layers[i - 1].weights.shape[1]
            # current_output = max(self.layers[i].weights.shape[1], min_layer_size)
            layer = self.layers[i]

            current_input, current_output = layer.weights.shape
            target_output = max(current_output, min_layer_size)

            if current_input != prev_output or current_output != target_output:
                layer.weights = resize_matrix(matrix=layer.weights, new_shape=(prev_output, target_output))
            if len(layer.biases) != target_output:
                layer.biases = resize_vector(vector=layer.biases, new_size=target_output)

        output_layer = self.layers[-1]
        prev_output = self.layers[-2].weights.shape[1]

        if output_layer.weights.shape != (prev_output, 1):
            output_layer.weights = resize_matrix(matrix=output_layer.weights, new_shape=(prev_output, 1))
        if len(output_layer.biases) != 1:
            output_layer.biases = resize_vector(vector=output_layer.biases, new_size=1)

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

    def get_genes(self):
        genes = []
        for layer in self.layers:
            genes.append({
                "weights": layer.weights.tolist(), 
                "biases": layer.biases.tolist()
            })
        return genes

    def set_genes(self, genes, min_layer_size):
        for layer, gene in zip(self.layers, genes):
            layer.weights = np.array(gene["weights"], dtype=np.float32)  # overwrite weights
            layer.biases = np.array(gene["biases"], dtype=np.float32)     # overwrite bias
        self.repair_network(min_layer_size=min_layer_size)

    def get_activation(self, is_ouput:bool = False):
        """Returns the correct activation function based on the rules we set."""
        return linear if is_ouput else leaky_relu
        
    def mutate_genes(self, mutation_rate = 0.1, mutation_strength = 0.1, new_neuron_rate = 0.0, delete_neuron_rate = 0.0, new_layer_rate = 0.0, delete_layer_rate = 0.0, mutate_topology:bool = False, min_layers_count = 4, min_layer_size = 2, max_layer_size = 40, max_neurons = 128, max_layers_count = 8):
        """Mutate a given set of genes (weights and biases)."""
        for layer in self.layers:
            # Create a mask (so that we don't update all values, 
            # but approximately the percentage of all, given by mutation_rate)
            # If the random value is below treshold, it is set to True (1), otherwise False (0)
            weights_mask = np.random.rand(*layer.weights.shape) < mutation_rate
            biases_mask = np.random.rand(*layer.biases.shape) < mutation_rate

            # Get indicies of elements which have 1 as their value
            weights_indicies = np.where(weights_mask)
            biases_indicies = np.where(biases_mask)
            
            # Update only elements that should be changed
            if weights_indicies[0].shape[0] > 0:
                layer.weights[weights_indicies] += np.random.uniform(-mutation_strength, mutation_strength, weights_indicies[0].shape[0])

            if biases_indicies[0].shape[0] > 0:
                layer.biases[biases_indicies] += np.random.uniform(-mutation_strength, mutation_strength, biases_indicies[0].shape[0])

        # Optionally we add/remove neurons/layers
        if mutate_topology:
            self.add_layer(probability=new_layer_rate, max_neurons=max_neurons, max_layers_count=max_layers_count, min_layer_size=min_layer_size)
            self.remove_layer(probability=delete_layer_rate, min_layers_count=min_layers_count, min_layer_size=min_layer_size)
            self.add_neuron(probability=new_neuron_rate, max_layer_size=max_layer_size, max_neurons=max_neurons)
            self.remove_neuron(probability=delete_neuron_rate, min_layer_size=min_layer_size)
            self.repair_network(min_layer_size=min_layer_size)

        for layer in self.layers:
            layer.weights = np.ascontiguousarray(layer.weights)
            layer.biases = np.ascontiguousarray(layer.biases)
    
    def get_layer_sizes(self):
        """Returns a list with sizes of each layer of the network."""
        return [layer.weights.shape[1] for layer in self.layers]

    def get_total_neurons(self):
        return sum(layer.weights.shape[1] for layer in self.layers)
    
    def clone(self):
        net = Network.__new__(Network)
        net.layers = []

        for layer in self.layers:
            new_layer = Layer.__new__(Layer)

            new_layer.weights = np.ascontiguousarray(layer.weights.copy())
            new_layer.biases = np.ascontiguousarray(layer.biases.copy())
            new_layer.activation = layer.activation

            net.layers.append(new_layer)

        return net
