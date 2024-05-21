# TODO:
#  1. Load the dataset using pandas' read_csv function.
#  2. Split the dataset into training, validation, and test sets. Consider using a split ratio such as 80:10:10 or another appropriate ratio.
#  3. Initialize the Linear Regression model using the provided `LinearRegression` class
#  4. Train the model using the training data.
#  5. Evaluate the trained model on the validation set,train set, test set. You might consider metrics like Mean Squared Error (MSE) for evaluation.
#  6. Plot the model's predictions against the actual values from the validation set using the `Visualisation` class.
import numpy as np
from configs.linear_regression_cfg import cfg

from models.linear_regression_model import LinearRegression
from ft_datasets.linear_regression_dataset import LinRegDataset
from utils.metrics import MSE
from utils.visualisation import Visualisation

from itertools import product


def search_for_random_hyperparameters(smallest_max_degree: int = 5,
                                      biggest_max_degree: int = 200,
                                      smallest_reg_coeff: float = 0.0,
                                      biggest_reg_coeff: float = 5.0,
                                      number_of_hyperparameter_sets: int = 100):
    return np.random.randint(low=smallest_max_degree, high=biggest_max_degree + 1, size=number_of_hyperparameter_sets), \
        np.random.uniform(low=smallest_reg_coeff, high=biggest_reg_coeff, size=number_of_hyperparameter_sets)



def start_train_validate_test(model, dataset):
    

    epoch = 20000
    print('---------------------------- TRAIN ----------------------------')
    model.train(dataset.inputs_train, dataset.targets_train, epoch)

    print('---------------------------- VALID ----------------------------')
    validate = model(dataset.inputs_valid)

   
    mse = MSE(validate,  dataset.targets_valid)
    loss = model.calculate_cost_function(model._plan_matrix(dataset.inputs_valid), dataset.targets_valid)

    if model.neptune_logger is not None:
        model.neptune_logger.log_final_val_mse(mse)
        model.neptune_logger.log_final_val_loss(loss)

        model.neptune_logger.run.stop()

        print('Actual: ', dataset.targets_valid[:5])
        print('Predicted: ', validate[:5])



    print('---------------------------- TEST ----------------------------')
    test = model(dataset.inputs_test)



def parameter_search(model, dataset, param_grid, scoring=MSE):
    """
    Функция для поиска лучших параметров модели.

    :param model: модель машинного обучения
    :param dataset: датасет для обучения
    :param param_grid: словарь с параметрами для перебора
    :param scoring: функция оценки качества модели
    :return: словарь с лучшими параметрами и значением метрики
    """
    X_train,  y_train = dataset.inputs_train, dataset.targets_train
    
    X_valid, y_valid = dataset.inputs_valid, dataset.targets_valid
    
    best_score = float('inf')
    best_params = {}
    best_model = None
    
    # Создание списка всех комбинаций параметров
    keys, values = zip(*param_grid.items())
    exp_count = 0
    for params in product(*values):
        kwargs = dict(zip(keys, params))
        #print(kwargs)
        test_model = model(**kwargs, experiment_name = f"exp_{exp_count}")
        test_model.fit(X_train, y_train, 1000)
        
        predictions = test_model.predict(X_valid)
        score = scoring(y_valid, predictions)
        loss = test_model.calculate_cost_function(test_model._plan_matrix(X_valid), y_valid)

        if test_model.neptune_logger is not None:
            test_model.neptune_logger.log_final_val_mse(score)
            test_model.neptune_logger.log_final_val_loss(loss)

            test_model.neptune_logger.run.stop()

            print('Actual: ',y_valid[:5])
            print('Predicted: ', predictions[:5])


        print(f"Score: {score}")
        #break
        
        # Поиск лучшей комбинации параметров
        if score < best_score:
            best_score = score
            best_params = kwargs
            best_model = test_model
        exp_count += 1
    
    return {'best_params': best_params, 'best_score': best_score, 'best_model': best_model}


if __name__ == "__main__":
    dataset = LinRegDataset(cfg)
    #model  = LinearRegression(cfg.base_functions, 0.001, 0.2, "First")
    #start_train_validate_test(model, dataset)
    param_grid = {
        'base_functions':cfg.base_functions,
        'learning_rate':cfg.learning_rate,
        'reg_coefficient':cfg.reg_coefficient#,
        #'epoch':cfg.epoch#,
        #'normalize': [True, False],
    }

    best_param = parameter_search(LinearRegression, dataset, param_grid)
    print(f"Best param: {best_param}")