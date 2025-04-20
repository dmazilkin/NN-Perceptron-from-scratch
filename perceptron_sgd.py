import numpy as np

class Perceptron:
    def __init__(self, X, Y, lr: int = 0.01, max_epochs: int = 3000):
        """
        Initialize the perceptron with train dataset

        :param X: train features
        :param Y: train targets
        """

        self._bias = np.random.randn(1)
        self._weights = np.transpose(np.random.randn(1, 2))
        self._learning_rate = lr
        self._errors = list()
        self._epochs = 0
        self._max_epochs = max_epochs
        self._features = X
        self._target = Y

    def train(self):
        while not self._is_condition_satisfied():
            self._start_epoch()
            self._epochs += 1

    def _start_epoch(self):
        dataset = np.random.permutation(np.concatenate((self._features, self._target.reshape(-1, 1)), axis=1))
        epoch_error = 0
        for sample in dataset:
            target = sample[-1]
            features = sample[:2]
            prediction = self.predict(features)
            error = self._calc_error(target, prediction)
            epoch_error += abs(error)
            self._update_weights(error, features)
            self._update_bias(error)
        self._errors.append(epoch_error)

    def predict(self, X):
        return self._get_activation_func(np.dot(X, self._weights) + self._bias)

    def _is_condition_satisfied(self):
        return (self._epochs >= self._max_epochs) or (self._errors[-1] == 0 if len(self._errors) > 0 else False)

    def _update_weights(self, error, features):
        self._weights = self._weights + self._learning_rate * error * features.reshape((-1, 1))

    def _update_bias(self, error):
        self._bias = self._bias + self._learning_rate * error * 1

    def _calc_error(self, target, prediction):
        error = target - prediction
        return error

    def _get_activation_func(self, prediction):
        return 1 if prediction > 0 else 0