"""
Cross Entropy

This will help us to find the better model which gives high probability to the correctly classified labels

Below example shows the model which predict the classes as possible_class have less cross_entropy
means it is good fit (i.e less the cross entropy better the model)
example 1) has only two classes 0 and 1
"""
import numpy as np

possible_classes = [1, 1, 0]
possible_probabilities = [0.8, 0.7, 0.1]


def cross_entropy(possible_outcomes, probabilities):
    Y = np.float_(possible_outcomes)
    P = np.float_(probabilities)
    return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))


print(cross_entropy(possible_classes, possible_probabilities))


# calculate cross entropy
def multi_cross_entropy(p, q):
    return -sum([p[i] * np.log(q[i]) for i in range(len(p))])
