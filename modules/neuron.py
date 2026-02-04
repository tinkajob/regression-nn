class Neuron:
    id_counter = 0

    def __init__(self, bias:float, activation, weights = None):
        # Each neuron gets it's own unique id
        self.id = Neuron.id_counter
        Neuron.id_counter += 1
        
        self.weights = {} if weights is None else dict(weights)
        self.bias = bias
        self.activation = activation if activation else (lambda x: x) # Should be lambda x: max(0, x) for hidden layers and lambda x: x

    def forward(self, inputs_by_id):
        s = self.bias
        for input_id, w in self.weights.items():
            s += w * inputs_by_id.get(input_id, 0.0)
        return self.activation(s)



