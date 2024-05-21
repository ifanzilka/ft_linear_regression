from easydict import EasyDict
from utils.enums import TrainType

import numpy as np
import math

cfg = EasyDict()

# Path to the dataframe
cfg.dataframe_path = './ft_datasets/linear_regression_dataset.csv'

# cfg.base_functions contains callable functions to transform input features.
# E.g., for polynomial regression: [lambda x: x, lambda x: x**2]
# TODO You should populate this list with suitable functions based on the requirements.

cfg.base_functions = [lambda x: x, lambda x: x**2]

# список lambda функций

cfg.base_functions_1 = [lambda x, i=i: np.power(x, i) for i in range(21)]
cfg.base_functions_2 = [lambda x, i=i: np.power(x, i) for i in range(101)]
cfg.base_functions_3 = [lambda x, i=i: np.power(x, i) for i in range(301)]

cfg.base_functions_4 = [lambda x, i=i: math.sin(x) for i in range(21)]
cfg.base_functions_5 = [lambda x, i=i: math.sin(x ** i) for i in range(11)]
cfg.base_functions_6 = [lambda x, i=i: math.sin(x ** i) for i in range(51)]

cfg.base_functions_7 = [lambda x: -math.log(1 + x) for i in range(21)]
cfg.base_functions_8 = [lambda x, i=i: -math.log(1 + x ** i) for i in range(11)]
cfg.base_functions_9 = [lambda x, i=i: -math.log(1 + x ** i) for i in range(51)]

cfg.base_functions_10 = [lambda x: math.exp(x) for i in range(21)]
cfg.base_functions_11 = [lambda x, i=i: math.exp(x ** i) for i in range(11)]
cfg.base_functions_12 = [lambda x, i=i: math.exp(x ** i) for i in range(51)]


# cfg.base_functions = [cfg.base_functions_1, cfg.base_functions_2, cfg.base_functions_3, cfg.base_functions_4,
#                       cfg.base_functions_5, cfg.base_functions_6, cfg.base_functions_7, cfg.base_functions_8,
#                       cfg.base_functions_9, cfg.base_functions_10, cfg.base_functions_11, cfg.base_functions_12]

cfg.base_functions = [cfg.base_functions_2]

#cfg.base_functions = [cfg.base_functions_2, cfg.base_functions_4, cfg.base_functions_7, cfg.base_functions_10]



cfg.train_set_percent = 0.8
cfg.valid_set_percent = 0.1

# Specifies the type of training algorithm to be used
cfg.train_type = TrainType.gradient_descent  #normal_equation

# how many times the algorithm will process the entire dataset for gradient descent algorithm
cfg.epoch = 1000

cfg.learning_rate = [0.1, 0.001, 0.0001]
cfg.reg_coefficient = [0, 0.1, 0.2, 0.4]
cfg.normalize = [True, False]
cfg.epoch = [100,1000]

cfg.exp_name = 'First'
cfg.env_path = '' # Путь до файла .env где будет храниться api_token.
cfg.api_token = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1NTBhZWU4ZS0wODJiLTQ0MTctOTE3OC00MTA2OTBmMTljNTUifQ=='
cfg.project_name = "ifanzilka/Linear-Regression-test"

