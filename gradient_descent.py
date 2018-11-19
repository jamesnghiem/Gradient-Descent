import numpy as np
from matplotlib import pyplot as plt


class GradientDescent:
    """
    parameters for gradient descent:
    @param precision - step size to terminate the algorithm on
    @param cost_function - lambda cost function to minimize
    @param gradient - lambda gradient function to determine the average gradient

    note that gradient must be consistent with cost_function, as it is the partial derivative
    """

    def __init__(self, precision, cost_function, gradient):
        self.precision = precision
        self.gradient = gradient
        self.cost_function = cost_function

    """
        @param x - sample data for the features
        @param y - sample data for the output
        @param theta - weights for the hypothesis
        @param learning_rate - learning rate (alpha)
        @param sample_size - number of samples (n)
        @param max_iterations - maximum number of iterations to run before halting the descent
        @param graphOn - whether gradient descent graph should be displayed (graph only supports 2d figures)
    """
    def run_gradient_descent(self, x, y, theta, learning_rate, sample_size, max_iterations, graphOn = True):
        # Draw the initial graph with the provided sample data
        _, n = np.shape(x)
        enable_graph = graphOn and n < 3
        if enable_graph:
            figure = plt.figure()
            ax = figure.add_subplot(111)
            ax.plot(x[:, 1], y, "x")
            ln, = ax.plot(x[:, 1], [np.sum(elems) for elems in x * theta], "-r")

            plt.ion()
            figure.show()

        x_transposed = x.transpose()

        for i in range(0, max_iterations):
            hypothesis = np.dot(x, theta)
            loss = hypothesis - y

            if enable_graph and i % 250 == 0:
                # redraw the graph every 250 iterations
                ln.remove()
                ln, = ax.plot(x[:, 1], [np.sum(elems) for elems in x * theta], "-r")
                figure.canvas.draw()

            # Average cost for the current sample data, using the provided cost function
            cost = self.cost_function(loss, sample_size)
            print ("Current Iteration: %d | Cost: %f" % (i, cost))
            # Once minimum cost has been passed, we can terminate and end early
            if cost <= self.precision:
                break

            # Average gradient for the current sample data
            gradient = self.gradient(x_transposed, loss, sample_size)
            theta = theta - learning_rate * gradient

        # draw the final graph
        if enable_graph:
            ln.remove()
            ax.plot(x[:, 1], [np.sum(elems) for elems in x * theta], "-r")
            figure.canvas.draw()
            plt.ioff()
            plt.show()

        return theta
