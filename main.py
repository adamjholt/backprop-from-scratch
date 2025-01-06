# ───────────────────────────────────────────────────────────────────────────────
# 1. Initialize Network
# ───────────────────────────────────────────────────────────────────────────────
#
# Each neuron has a set of weights that need to be maintained. One weight for each
# input connection and an additional weight for the bias. We will need to store
# additional properties for a neuron during training, therefore we will use a
# dictionary to represent each neuron and store properties by names such as
# 'weights' for the weights.
#
# A neural network is organized into layers. We will organize layers as arrays of
# dictionaries and treat the whole network as an array of layers.

from random import random, seed

# %%


def initialize_network(n_inputs, n_hidden, n_outputs):
    """Initialize a neural network with random weights.

    Args:
        n_inputs (int): Number of inputs to the network.
        n_hidden (int): Number of neurons in the hidden layer.
        n_outputs (int): Number of neurons in the output layer.

    Returns:
        list: A list of layers, where each layer is a list of neuron
            dictionaries with initialized weights (including bias).
    """
    network = []
    hidden_layer = [
        {"weights": [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)
    ]
    output_layer = [
        {"weights": [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)
    ]
    network.append(hidden_layer)
    network.append(output_layer)
    return network


# %%

seed(1)
network = initialize_network(2, 1, 2)
for layer in network:
    print(layer)


# %%

# ───────────────────────────────────────────────────────────────────────────────
# 2. Forward Propagate
# ───────────────────────────────────────────────────────────────────────────────
#
# We can calculate an output from a neural network by propagating an input signal
# through each layer until the output layer outputs its values.
#
# We can break forward propagation down into three parts:
#
#     1. Neuron Activation
#     2. Neuron Transfer
#     3. Forward Propagation

# ──────────────────────────────────────
# 2.1 Neuron Activation
# ──────────────────────────────────────
#
# The first step is to calculate the activation of one neuron given an input.
# Neuron activation is calculated as the weighted sum of the inputs – much like
# linear regression.

# %%

# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation

# %%

# ──────────────────────────────────────
# 2.2 Neuron Transfer
# ──────────────────────────────────────
#
# Once a neuron is activated, we need to transfer the activation to see what the
# neuron output actually is. We can transfer an activation function using the
# sigmoid function as follows:

# %%

from math import exp

# Transfer a neuron activation
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))

# %%

# ──────────────────────────────────────
# 2.3 Forward Propagation
# ──────────────────────────────────────
#
# Forward propagating an input is straightforward. We work through each layer of
# our network calculating the outputs for each neuron. All of the outputs from
# one layer become inputs to the neurons on the next layer.

# %%

# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

# %%

# test forward propagation
network = [
        [{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
        [{'weights': [0.2550690257394217, 0.49543508709194095]}, {'weights': [0.4494910647887381, 0.651592972722763]}]
]
row = [1, 0, None]
output = forward_propagate(network, row)
print(output)

# %%
