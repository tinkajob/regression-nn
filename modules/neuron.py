class Neuron:
    def __init__(self, bias:float, activation, weights:list):
        # Each neuron gets it's own unique id
        self.bias = bias
        self.activation = activation if activation else (lambda x: x) # Should be lambda x: max(0, x) for hidden layers and lambda x: x
        self.weights = weights
    
    def forward(self, inputs):
        value = self.bias
        for i in range(len(inputs)):
            value += inputs[i] * self.weights[i]
        return self.activation(value)