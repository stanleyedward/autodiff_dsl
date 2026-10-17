import numpy as np
import random
from .Value import Value

class Neuron:
    def __init__(self, n_inputs):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Value(random.uniform(-1, 1))
        
    def __call__(self, x):
        return self.forward(x)
    def forward(self, x):
        #w * x + b
        act = sum(wi* xi for wi, xi in zip(self.w, x)) + self.b
        output = act.tanh()
        return output

    def parameters(self):
        return self.w + [self.b]
    
class Layer:
    def __init__(self, n_input, n_output):
        self.neurons = [Neuron(n_inputs=n_input) for _ in range(n_output)]
    def __call__(self, x):
        return self.forward(x)
    def forward(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs 
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
    
class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
    def __call__(self, x):
        return self.forward(x)
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]