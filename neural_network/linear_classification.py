"""
the linear algorithm to separate the following data
"""

import numpy as np
import pandas as pd
import os

# Setting the random seed, feel free to change it and see different solutions.
np.random.seed(42)

# Input and Output data path
cwd = os.getcwd()
data_file_path = os.path.join(cwd, 'data_files/input_data.csv')


def step_function(t):
    if t >= 0:
        return 1
    return 0


def read_csv(file_path):
    input_df = pd.read_csv(file_path, delimiter=',')
    for row in input_df.values:
        yield row[:2], row[-1]


def prediction(inputs, wights, b):
    return step_function(np.matmul(inputs, wights) + b)


def perceptrons_step(file_path, W, b, learn_rate=0.01):
    for input_data, label in read_csv(file_path):
        y_pred = prediction(input_data, W, b)
        if y_pred != label:
            if y_pred == 1:  # subtract
                W[0] = W[0] - input_data[0] * learn_rate
                W[1] = W[1] - input_data[1] * learn_rate
                b = b - learn_rate
            else:
                W[0] = W[0] + input_data[0] * learn_rate
                W[1] = W[1] + input_data[1] * learn_rate
                b = b + learn_rate
    return W, b


def train_perceptrons_algorithm(file_path, learn_rate=0.01, num_epochs=50):
    # These are the solution lines that get plotted below.
    boundary_lines = []
    # Weights and bias
    weights = np.array(np.random.rand(2, 1))
    bias = 0.73199394
    for i in range(num_epochs):
        # In each epoch, we apply the perceptrons step.
        W, b = perceptrons_step(file_path, weights, bias, learn_rate)
        boundary_lines.append((-W[0] / W[1], -b / W[1]))
    return boundary_lines


if __name__ == "__main__":
    print(train_perceptrons_algorithm(data_file_path))
