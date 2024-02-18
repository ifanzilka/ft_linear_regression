import numpy as np


def data_spliter(x: np.ndarray, y: np.ndarray, proportion: float) -> tuple:
    """
    Shuffles and splits the dataset (given by x and y) into a training and a test set,
    while respecting the given proportion of examples to be kept in the training set.
    Args:
        x: has to be an numpy.array, a matrix of shape m * n.
        y: has to be an numpy.array, a vector of shape m * 1.
        proportion: has to be a float, the proportion of the dataset that will be assigned to the training set.
    Return:
        (x_train, x_test, y_train, y_test) as a tuple of numpy.array
        None if x or y is an empty numpy.array.
        None if x and y do not share compatible shapes.
        None if x, y or proportion is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        return None
    if not isinstance(proportion, float):
        return None
    if proportion <= 0 or proportion >= 1:
        return None
    
    randomize = np.arange(x.shape[0])
    np.random.shuffle(randomize)
    x_shuffle = x[randomize]
    y_shuffle = y[randomize]
    length = x.shape[0]
    return (
        *np.split(x_shuffle, [int(proportion * length)]),
        *np.split(y_shuffle, [int(proportion * length)]),
    )


def r2score_elem_ssr(y, y_hat):
    a = y - y_hat
    return (a ** 2)


def r2score_elem_sst(y):
    a = y - np.mean(y)
    return (a ** 2)

def r2score_(y, y_hat):
    """
    Description:
    Calculate the R2score between the predicted output and the output.
    Args:
    y: has to be a numpy.array, a vector of dimension m * 1.
    y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
    r2score: has to be a float.
    None if there is a matching dimension problem.
    Raises:
    This function should not raise any Exceptions.
    """
    if len(y) != len(y_hat):
        return None
    ssr = np.sum(r2score_elem_ssr(y, y_hat))
    sst = np.sum(r2score_elem_sst(y))
    return 1 - (ssr / sst)