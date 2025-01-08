# ──────────────────────────────────────────────────────────────────────────────
# 1. Initialize Network
# ──────────────────────────────────────────────────────────────────────────────
#
# Each neuron has a set of weights that need to be maintained. One weight for each
# input connection and an additional weight for the bias. We will need to store
# additional properties for a neuron during training, therefore we will use a
# dictionary to represent each neuron and store properties by names such as
# 'weights' for the weights.
#
# A neural network is organized into layers. We will organize layers as arrays of
# dictionaries and treat the whole network as an array of layers.

# %%

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

# ──────────────────────────────────────────────────────────────────────────────
# 2. Forward Propagate
# ──────────────────────────────────────────────────────────────────────────────
#
# We can calculate an output from a neural network by propagating an input signal
# through each layer until the output layer outputs its values.
#
# We can break forward propagation down into three parts:
#
#     1. Neuron Activation
#     2. Neuron Transfer
#     3. Forward Propagation
#
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


# ──────────────────────────────────────────────────────────────────────────────
# 3. Back Propagate Error
# ──────────────────────────────────────────────────────────────────────────────
#
#
# The backpropagation algorithm is named for the way in which weights are 
# trained. Error is calculated between the expected outputs and the outputs
# forward propagated from the network. These errors are then propagated backward
# through the network from the output layer to the hidden layer, assigning blame
# for the error and updating weights as they go.
#
# We can break it down into:
#
#    1. Transfer Derivative
#    2. Error Backpropagation
#
# ──────────────────────────────────────
# 3.1 Transfer Derivative
# ──────────────────────────────────────
#
# We are using the sigmoid transfer function, the derivative of which can be
# calculated as follows:

# %%

def transfer_derivative(output):
    return output * (1.0 - output)

# %%

# ──────────────────────────────────────
# 3.2. Error Backpropagation
# ──────────────────────────────────────
#
# The first step is to calculate the error for each output neuron, this will
# give us our error signal (input) to propagate backwards through the network.
# The error signal for a neuron in the hidden layer is calculated as the
# weighted error of each neuron in the output layer. Think of the error
# traveling back along the weights of the output layer to the neurons in the
# hidden layer. The back-propagated error signal is accumulated and then used
# to determine the error for the neuron in the hidden layer.

# %%

def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(neuron['output'] - expected[j])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# %%

# test backpropagation of error
network = [
    [{'output': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
    [{'output': 0.6213859615555266, 'weights': [0.2550690257394217, 0.49543508709194095]},
     {'output': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763]}]
]
expected = [0, 1]
backward_propagate_error(network, expected)
for layer in network:
    print(layer)

# %%


# ──────────────────────────────────────────────────────────────────────────────
# 4. Train Network
# ──────────────────────────────────────────────────────────────────────────────
# 
# The network is trained using stochastic gradient descent.
# 
# This involves multiple iterations of exposing a training dataset to the
# network and for each row of data forward propagating the inputs,
# backpropagating the error and updating the network weights.
#
# This is broken down into two sections:
#
#     1. Update Weights
#     2. Train Network
#
# ──────────────────────────────────────
# 4.1 Update Weights
# ──────────────────────────────────────

# Once errors are calculated for each neuron in the network via the back
# propagation method above, they can be used to update weights. Network weights
# are updated as follows:
#
# `weight = weight - learning_rate * error * input`
#
# Where weight is a given weight, learning_rate is a parameter that you must
# specify, error is the error calculated by the backpropagation procedure for
# the neuron and input is the input value that caused the error.

# %%

# Update network weights with error
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
            for neuron in network[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j] -= l_rate * neuron['delta'] * inputs[j]
                neuron['weights'][-1] -= l_rate * neuron['delta']

# %%

# ──────────────────────────────────────
# 4.2. Train Network
# ──────────────────────────────────────
#
# The network is updated using stochastic gradient descent. This involves first
# looping for a fixed number of epochs and within each epoch updating the
# network for each row in the training dataset.
#
# Below is a function that implements the training of an already initialized
# neural network with a given training dataset, learning rate, fixed number of
# epochs and an expected number of output values.
#
# The expected number of output values is used to transform class values in the
# training data into a one hot encoding. That is a binary vector with one column
# for each class value to match the output of the network. This is required to
# calculate the error for the output layer.

# %%

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

# %%


# Test training backprop algorithm
seed(1)
dataset = [[2.7810836,2.550537003,0],
           [1.465489372,2.362125076,0],
           [3.396561688,4.400293529,0],
           [1.38807019,1.850220317,0],
           [3.06407232,3.005305973,0],
           [7.627531214,2.759262235,1],
           [5.332441248,2.088626775,1],
           [6.922596716,1.77106367,1],
           [8.675418651,-0.242068655,1],
           [7.673756466,3.508563011,1]]
n_inputs = len(dataset[0]) - 1
n_outputs = len(set([row[-1] for row in dataset]))
network = initialize_network(n_inputs, 2, n_outputs)
train_network(network, dataset, 0.5, 20, n_outputs)
for layer in network:
    print(layer)

# %%


# ──────────────────────────────────────────────────────────────────────────────
# 5. Predict
# ──────────────────────────────────────────────────────────────────────────────
#
# We have already seen how to forward-propagate an input pattern to get an
# output. This is all we need to do to make a prediction. We can use the output
# values themselves directly as the probability of a pattern belonging to each
# output class.

# %%

# Make a prediction with a network
def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))

# %%

# Test making predictions with the network
dataset = [[2.7810836,2.550537003,0],
           [1.465489372,2.362125076,0],
           [3.396561688,4.400293529,0],
           [1.38807019,1.850220317,0],
           [3.06407232,3.005305973,0],
           [7.627531214,2.759262235,1],
           [5.332441248,2.088626775,1],
           [6.922596716,1.77106367,1],
           [8.675418651,-0.242068655,1],
           [7.673756466,3.508563011,1]]
network = [
    [{'weights': [-1.482313569067226, 1.8308790073202204, 1.078381922048799]},
     {'weights': [0.23244990332399884, 0.3621998343835864, 0.40289821191094327]}],
    [{'weights': [2.5001872433501404, 0.7887233511355132, -1.1026649757805829]},
     {'weights': [-2.429350576245497, 0.8357651039198697, 1.0699217181280656]}]
]
for row in dataset:
    prediction = predict(network, row)
    print('Expected=%d, Got=%d' % (row[-1], prediction))

# %%



