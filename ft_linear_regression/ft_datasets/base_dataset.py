import numpy as np
from abc import ABC, abstractmethod


class BaseDataset(ABC):

    def __init__(self,train_set_percent,valid_set_percent):
        self.train_set_percent = train_set_percent
        self.valid_set_percent = valid_set_percent

    @property
    @abstractmethod
    def targets(self):
        # targets variables
        pass

    @property
    @abstractmethod
    def inputs(self):
        # inputs variables
        pass


    def _divide_into_sets(self):
        # define self.inputs_train, self.targets_train, self.inputs_valid, self.targets_valid,
        #  self.inputs_test, self.targets_test
        inputs_length = len(self.inputs)
        indexes = np.random.permutation(inputs_length)

        train_set_size = int(self.train_set_percent * inputs_length)
        valid_set_size = int(self.valid_set_percent * inputs_length)

        randomized_inputs = self.inputs[indexes]
        randomized_targets = self.targets[indexes]

        self.inputs_train = randomized_inputs[:train_set_size]
        self.inputs_valid = randomized_inputs[train_set_size:train_set_size + valid_set_size]
        self.inputs_test = randomized_inputs[train_set_size + valid_set_size:]

        self.targets_train = randomized_targets[:train_set_size]
        self.targets_valid = randomized_targets[train_set_size:train_set_size + valid_set_size]
        self.targets_test = randomized_targets[train_set_size + valid_set_size:]
