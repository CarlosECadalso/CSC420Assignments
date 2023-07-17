######################################
# Assignement 2 for CSC420
# MNIST clasification example
# Author: Jun Gao
######################################
import math
import numpy as np


def cross_entropy_loss_function(prediction, label):
    # TODO: compute the cross entropy loss function between the prediction and ground truth label.
    # prediction: the output of a neural network after softmax. It can be an Nxd matrix,
    # where N is the number of samples, and d is the number of different categories
    # label: The ground truth labels, it can be a vector with length N, and each element in this vector stores
    # the ground truth category for each sample.
    # Note: we take the average among N different samples to get the final loss.
    truth_label = np.array(label)
    nn_pred = np.array(prediction)
    softmax_of_pred = softmax(nn_pred)

    row, col = nn_pred.shape

    loss_final = 0
    loss_at_sample = np.zeros((row, col))

    for k in range(col):
        loss_at_sample[:, k] = truth_label * np.log(softmax_of_pred[:, k])
        loss_final -= loss_at_sample[k]

    return loss_final


def sigmoid(x):
    # TODO: compute the softmax with the input x: y = 1 / (1 + exp(-x))
    sigmoid_return = 1 / (1 + math.e**(-x))
    return sigmoid_return


def d_sigmoid(x):
    # TODO: compute the softmax with the input x: y = 1 / (1 + exp(-x))
    sigmoid_return = (math.e**(-x)) / ((1 + math.e**(-x))**2)
    return sigmoid_return


def softmax(x):
    # TODO: compute the softmax function with input x.
    #  Suppose x is Nxd matrix, and we do softmax across the last dimention of it.
    #  For each row of this matrix, we compute x_{j, i} = exp(x_{j, i}) / \sum_{k=1}^d exp(x_{j, k})
    row, col = x.shape

    softmax_top = np.exp(x)
    softmax_bottom = np.sum(softmax_top)

    softmax_o = softmax_top / softmax_bottom

    return softmax_o


class OneLayerNN():
    def __init__(self, num_input_unit, num_output_unit):
        # TODO: Random Initliaize the weight matrixs for a one-layer MLP.
        #  the number of units in each layer is specified in the arguments
        # Note: We recommend using np.random.randn() to initialize the weight matrix:
        # zero mean and the variance equals to 1 and initialize the bias matrix as full zero using np.zeros()
        self.weights = np.random.randn(num_input_unit, num_output_unit)
        self.bias = np.zeros((1, num_output_unit))

    def forward(self, input_x):
        # TODO: Compute the output of this neural network with the given input.
        # Suppose input_x is an Nxd matrix, where N is the number of samples and d is the number of dimension
        # for each sample.
        # Compute output: z = softmax (input_x * W_1 + b_1), where W_1, b_1 are weights, biases for this layer
        # Note: If we only have one layer in the whole model and we want to use it to do classification,
        #       then we directly apply softmax **without** using sigmoid (or relu) activation
        softmax_input = np.dot(input_x, self.weights)
        softmax_input = softmax_input + self.bias
        forward_output = softmax(softmax_input)

        return forward_output

    def backpropagation_with_gradient_descent(self, loss, learning_rate, input_x, label):
        # TODO: given the computed loss (a scalar value),
        #  compute the gradient from loss into the weight matrix and running gradient descent
        # Note that you may need to store some intermediate value when you do forward pass,
        # such that you don't need to recompute them
        # Suggestions: you need to first write down the math for the gradient, then implement it to compute the gradient
        output_zn = self.forward(input_x)
        forward_output = sigmoid(output_zn)

        dl_do = np.zeros(forward_output.shape)

        for n in range(forward_output.shape[1]):
            dl_do[:, n] = forward_output[:, n] - label

        w_star = loss
        do_dz = d_sigmoid(output_zn)
        dz_dw = output_zn

        dl_dw = sum(dl_do * do_dz * dz_dw)

        new_weights = self.weights - (learning_rate * dl_dw)
        new_bias = self.bias - (learning_rate * w_star)

        self.weights = new_weights
        self.bias = new_bias


# [Bonus points] This is not necessary for this assignment
class TwoLayerNN():
    def ___init__(self, num_input_unit, num_hidden_unit, num_output_unit):
        # TODO: Random Initliaize the weight matrixs for a two-layer MLP with sigmoid activation,
        #  the number of units in each layer is specified in the arguments
        # Note: We recommend using np.random.randn() to initialize the weight matrix:
        # zero mean and the variance equals to 1 and initialize the bias matrix as full zero using np.zeros()
        pass

    def forward(self, input_x):
        # TODO: Compute the output of this neural network with the given input.
        # Suppose input_x is Nxd matrix,
        # where N is the number of samples and d is the number of dimension for each sample.
        # Compute: first layer: z = sigmoid (input_x * W_1 + b_1) # W_1, b_1 are weights, biases for the first layer
        # Compute: second layer: o = softmax (z * W_2 + b_2) # W_2, b_2 are weights, biases for the second layer
        pass

    def backpropagation_with_gradient_descent(self, loss, learning_rate):
        # TODO: given the computed loss (a scalar value),
        #  compute the gradient from loss into the weight matrix and running gradient descent
        # Note that you may need to store some intermidiate value when you do forward pass,
        # such that you don't need to recompute them
        # Suggestions: you need to first write down the math for the gradient, then implement it to compute the gradient
        pass
