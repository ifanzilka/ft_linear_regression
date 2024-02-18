import numpy as np
import time
import sys

def ft_progress(list):
    start = time.time()
    mval = max(list)
    length = len(str(max(list)))
    eta = 0
    barsize = 40
    for i in list:
        per = i/mval*100
        bar = int(i/mval*barsize)
        t = time.time() - start
        if not per == 0:
            eta = t/per*100
        if per < 40:
            color = "\033[31m"
        elif per < 75:
            color = "\033[33m"
        else:
            color = "\033[32m"
        sys.stdout.write('\r')
        sys.stdout.write("ETA: %.2fs [%3d%%] |%s%-*.*s%s| %*d/%d | elapsed time %.2fs" % (eta, per, color, barsize, barsize, "█"*bar, "\033[0m", length, i, mval, t))
        sys.stdout.flush()
        yield i

class MyLinearRegression():
    """
    Description:
        My personnal linear regression class to fit like a boss.
    """

    def __init__(self,  thetas:np.array, alpha=0.001, max_iter=1000, normalize= 'n'):
        
        if isinstance(thetas, list):
            thetas = np.array(thetas)        
        if not isinstance(thetas, np.ndarray):
            return None
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas
        self.not_normalized = None
        self.normalize = normalize

    def gradient(self, x:np.array, y:np.array, theta:np.array):
        """
        Computes a gradient vector from three non-empty numpy.ndarray, without any for-loop.
        The three arrays must have the compatible dimensions.
        Args:
            x: has to be an numpy.ndarray, a matrix of dimension m * n.
            y: has to be an numpy.ndarray, a vector of dimension m * 1.
            theta: has to be an numpy.ndarray, a vector (n + 1) * 1.
        Returns:
            The gradient as a numpy.ndarray, a vector of dimensions n * 1, containg the result ofthe formula for all j.
            None if x, y, or theta are empty numpy.ndarray.
            None if x, y and theta do not have compatible dimensions.
            None if x or y or theta are not of the expected type objects.
        Raises:
        This function should not raise any Exception.
        """
        if not isinstance(y, np.ndarray) or not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
            return None

        y = y.reshape(-1,1)
        theta = theta.reshape(-1,1)
        X = np.hstack((np.ones((x.shape[0], 1)), x))
        Y_pred = X.dot(theta)
        return ((X.T.dot(Y_pred - y)) / len(x)).reshape(-1)


    def fit_(self, x:np.array, y:np.array):
        """
        Description:
            Fits the model to the training dataset contained in x and y.
        Args:
            x: has to be a numpy.ndarray, a matrix of dimension m * n: (number of trainingexamples, number of features).
            y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of trainingexamples, 1).
            theta: has to be a numpy.ndarray, a vector of dimension (n + 1) * 1: (number offeatures + 1, 1).
            alpha: has to be a float, the learning ratemax_iter: has to be an int, the number of iterations done during the gradientdescent
        Returns:
            new_theta: numpy.ndarray, a vector of dimension (number of features + 1, 1).
            None if there is a matching dimension problem.None if any of the parameter is not of the expected type object.
        Raises:
            This function should not raise any Exception.
        """
        if not isinstance(y, np.ndarray) or not isinstance(x, np.ndarray):
            return None
        
        if self.normalize == 'y':
            x_target, y_target = self.normalize_(x, y)
        else:
            x_target = x
            y_target = y


        for _ in range(self.max_iter):
            grad = self.gradient(x_target, y_target, self.thetas).reshape(len(self.thetas), 1).astype(np.float64) ## (4,1)

            self.thetas = self.thetas.reshape(len(self.thetas), 1) - self.alpha * grad
            self.thetas = self.thetas.astype(np.float64)
            #print(f"thetas new SHAPE: {self.thetas.shape}")
            #print("\n")
            #break
        print("\n")
        if self.normalize == 'y':
            print('before denormalization', self.thetas.flatten())
            before = self.thetas
            self.thetas = self.denormalize_theta(self.thetas, x, y)
            print('after denormalization', self.thetas.flatten())
            self.not_normalized = before
            return self.thetas, before

        return self.thetas
        

    def predict_(self, x:np.array):
        """
        Computes the prediction vector y_hat from two non-empty numpy.ndarray.
        Args:
            x: has to be an numpy.ndarray, a matrix of dimension m * n.
            theta: has to be an numpy.ndarray, a vector of dimension (n + 1) * 1.
        Returns:
            y_hat as a numpy.ndarray, a vector of dimension m * 1.
            None if x or theta are empty numpy.ndarray.
            None if x or theta dimensions are not matching.
            None if x or theta are not of the expected type objects.
        Raises:This function should not raise any Exception.
        """
        if not isinstance(x, np.ndarray):
            return None
        if (x.shape[1] + 1 != self.thetas.shape[0]):
            return None
        X = np.hstack((np.ones((x.shape[0], 1)), x))
        if self.normalize == 'y':
            return X.dot(self.not_normalized)
        return X.dot(self.thetas)

    def mse_elem(self, y, y_hat):
        a = y_hat - y
        return (a ** 2)

    def mse_(self, y, y_hat):
        """
        Description:
        Calculate the MSE between the predicted output and the real output.
        Args:
        y: has to be a numpy.array, a vector of dimension m * 1.
        y_hat: has to be a numpy.array, a vector of dimension m * 1.
        Returns:
        mse: has to be a float.
        None if there is a matching dimension problem.
        Raises:
        This function should not raise any Exceptions.
        """
        if y.shape != y_hat.shape:
            return None
        return np.mean(np.square(y_hat - y))


    # def mse_(self, x:np.array, y:np.array):
    #     """
    #     Description:
    #     Calculate the MSE between the predicted output and the real output.
    #     Args:
    #         y: has to be a numpy.ndarray, a vector of dimension m * 1. 
    #         y_hat: has to be a numpy.ndarray, a vector of dimension m * 1.
    #     Returns:
    #         mse: has to be a float.
    #         None if there is a matching dimension problem.
    #     """
    #     if not isinstance(y, np.ndarray) or not isinstance(x, np.ndarray):
    #         return None
    #     y_hat = self.predict_(x) 
    #     y_hat = y_hat.reshape(-1, 1)
    #     return (np.power(y - y_hat,2)/ len(y)).sum()

    def cost_elem_(self, x:np.array, y:np.array):
        """
        Description:Calculates all the elements (y_pred - y)ˆ2 of the cost function.
        Args:
            x: has to be an numpy.ndarray, a vector.
            y: has to be an numpy.ndarray, a vector.
        Returns:
            J_elem: numpy.ndarray, a vector of dimension (number of the training examples,1).
            None if there is a dimension matching problem between X, Y or theta.None if any argument is not of the expected type.
        Raises:
            This function should not raise any Exception.
        """
        
        if not isinstance(y, np.ndarray) or not isinstance(x, np.ndarray):
            return None
        if y.shape[0] != x.shape[0]:
            return None
        Y_pred = self.predict_(x)
        return (Y_pred - y) * (Y_pred - y)

    def cost_(self, x:np.array, y:np.array):
        """
        Description:Calculates the value of cost function.
        Args:
            y: has to be an numpy.ndarray, a vector.
            y_hat: has to be an numpy.ndarray, a vector.
        Returns:
            J_value : has to be a float.
            None if there is a dimension matching problem between X, Y or theta.
            None if any argument is not of the expected type.
        Raises:This function should not raise any Exception.
        """
        if not isinstance(y, np.ndarray) or not isinstance(y, np.ndarray):
            return None
        return np.sum(self.cost_elem_(x, y)) / (2 * len(y))


    def normalize_(self, x, y):
        """
        Normalize the feature matrix x by subtracting the mean and dividing by the standard deviation.
        Args:
        x: numpy.array, a matrix of shape m * n (number of training examples, number of features).
        Returns:
        x_normalized: numpy.array, a matrix of shape m * n with normalized feature values.
        None if x is an empty numpy.array.
        """
        if x.size == 0:
            return None

        if x.size == 0 or y.size == 0:
            return None

        X_normalized = self.minmax_(x)
        y_normalized = self.minmax_(y)

        return X_normalized, y_normalized
    
    def minmax_(self, x):
        """Computes the normalized version of a non-empty numpy.ndarray using the min-max standardization.
        Args:
        x: has to be an numpy.ndarray, a vector.
        Returns:
        x’ as a numpy.ndarray.
        None if x is a non-empty numpy.ndarray or not a numpy.ndarray.
        Raises:
        This function shouldn’t raise any Exception.
        """

        min_x = np.min(x)
        max_x = np.max(x)

        if min_x == max_x:
            return x

        return (x - min_x) / (max_x - min_x)

    def normalize_theta(self, x, y):
        if self.thetas.size == 0 or x.size == 0 or y.size == 0:
            return None

        x_mean, x_std = np.mean(x), np.std(x)
        y_mean, y_std = np.mean(y), np.std(y)

        if x_std == 0:
            return None

        normalized_theta = self.thetas.copy()
        normalized_theta[0] = self.thetas[0] - \
            (self.thetas[1] * y_std / x_std) * x_mean + y_mean
        normalized_theta[1] = self.thetas[1] * y_std / x_std

        return normalized_theta

    def denormalize_theta(self, theta, x, y):
        x_min = np.min(x)
        x_max = np.max(x)
        y_min = np.min(y)
        y_max = np.max(y)

        theta_denorm = np.zeros_like(theta)
        theta_denorm[0] = theta[0] * (y_max - y_min) + y_min
        theta_denorm[1:] = theta[1:] * (y_max - y_min) / (x_max - x_min)

        return theta_denorm