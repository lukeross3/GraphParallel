import numpy as np
import time

# Generic Node class
class Node():
    def __init__(self):
        self.input_dependencies = []
        self.output_dependencies = []
        self.output = None
        
    def forward(self, out):
        self.output = out
        return out

    def reset(self):
        self.output = None

# Input node (for sending input to subsequent nodes)
class Input(Node):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return super().forward(x)

# Add list of numpy arrays
class Add(Node):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = np.zeros_like(x[0])
        for i in range(len(x)):
            out = out + x[i]
        return super().forward(out)

# Flatten multi-dimensional numpy array
class Flatten(Node):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return super().forward(x.flatten())

# Linear layer (matrix multiply, no bias)
class Linear(Node):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights

    def forward(self, x):
        return super().forward(np.matmul(self.weights, x))

# 1D Convolutional layer
class Conv1d(Node):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights

    def forward(self, x):
        return super().forward(np.convolve(self.weights, x, mode='same'))

# Arbitrary number of inputs, lasts for variable amount of time
class Timing(Node):
    def __init__(self, input_shape=(300,1), t=0.0001):
        super().__init__()
        self.input_shape = input_shape
        self.t = t

    def forward(self, x):
        time.sleep(self.t)
        return super().forward(np.ones(self.input_shape))