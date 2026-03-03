import numpy as np

class Layer:
    def __init__(self, activation, input_size:int, output_size:int):
        self.weights = np.random.uniform(-1, 1, (input_size, output_size))
        self.biases = np.random.uniform(-1, 1, output_size)
        self.activation = activation if activation else (lambda x: x) # Leaky ReLU for hidden layers and linear for output layer
    
    def forward(self, inputs):
        value = np.dot(inputs, self.weights) + self.biases
        return self.activation(value)