import numpy as np

# def MSE(predictions: np.ndarray, targets: np.ndarray) -> float:
#     """ Compute the Mean Squared Error (MSE) between predictions and targets.

#     The Mean Squared Error is a measure of the average squared difference
#     between predicted and actual values. It's a popular metric for regression tasks.

#     Formula:
#     MSE = (1/n) * Σ (predictions - targets)^2

#     where:
#     - n is the number of samples
#     - Σ denotes the sum
#     - predictions are the predicted values by the model
#     - targets are the true values
#     TODO implement this function. This function is expected to be implemented without the use of loops.

#     """
#     pass


def MSE(predictions: np.ndarray, targets: np.ndarray) -> float:
    return (1 / len(targets)) * np.sum(np.power(targets - predictions, 2))


def R_Square(predictions: np.ndarray, targets: np.ndarray):
    return 1 - len(targets) * MSE(predictions, targets) / np.sum(np.power(targets - targets.mean(), 2))
