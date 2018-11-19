import numpy as np
from matplotlib import pyplot as plt
import random

from gradient_descent import GradientDescent

# Mean Squared Error gradient descent
mean_squared_descent = GradientDescent(10,
                                       lambda loss, sample_size: np.sum(loss ** 2) / (2 * sample_size),
                                       lambda x_transposed, loss, sample_size: np.dot(x_transposed, loss) / sample_size)


def generate_sample_data(sample_size, bias, variance):
    x = np.zeros(shape=(sample_size, 2))
    y = np.zeros(shape=sample_size)
    # basically a straight line
    for i in range(0, sample_size):
        # bias feature
        x[i][0] = 1
        x[i][1] = i
        # our target variable
        y[i] = (i + bias) + random.uniform(0, 1) * variance
    return x, y


def test_gradient_descent(sample_size, bias, variance, gradient_descent=mean_squared_descent):
    x, y = generate_sample_data(sample_size, bias, variance)
    m, n = np.shape(x)
    theta = np.ones(n)

    theta = gradient_descent.run_gradient_descent(x, y, theta, 0.0006, sample_size, 10000, True)
    print (theta)


test_gradient_descent(100, 25, 25)
