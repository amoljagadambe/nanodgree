import numpy as np

"""
Soft max activation function

This function that takes as input a list of numbers, and returns
the list of values given by the softmax function.
"""
output_values_from_linear_equation = [1.0, 2.0, 4.0, 3.5, 2.5]


def softmax(output_list):
    exp_list = np.exp(output_list)
    sum_exp_list = sum(exp_list)
    result = []
    for i in exp_list:
        result.append(i * 1.0 / sum_exp_list)
    return result


print(softmax(output_values_from_linear_equation))

"""
Second Method to calculate softmax
"""


def softmax_using_np(output_list):
    exp_list = np.exp(output_list)
    return np.divide(exp_list, exp_list.sum())


print(softmax_using_np(output_values_from_linear_equation))

"""
sigmoid function

With the help of Sigmoid activation function, we are able to reduce the loss during the time 
of training because it eliminates the gradient problem in machine learning model while training.
"""


def sigmoid(value):
    return 1 / (1 + np.exp(-value))


print(list(map(sigmoid, output_values_from_linear_equation)))
