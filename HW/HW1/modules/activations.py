import numpy as np
import scipy.special
from .base import Module
import scipy

class ReLU(Module):
    """
    Applies element-wise ReLU function
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        return np.where(input > 0, input, 0)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        return (input > 0) * grad_output


class Sigmoid(Module):
    """
    Applies element-wise sigmoid function
    """
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        return self.sigmoid(input)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        s = self.sigmoid(input)
        return s * (1 - s) * grad_output


class Softmax(Module):
    """
    Applies Softmax operator over the last dimension
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        exp_input = np.exp(input - np.max(input, axis=1, keepdims=True))
        return exp_input / exp_input.sum(axis=1, keepdims=True)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        output = self.compute_output(input)
        
        batch_size = input.shape[0]
        grad_input = np.zeros_like(input)
        
        for i in range(batch_size):     # Я не знаю как обойтись без цикла
            s = output[i]
            J = np.diag(s) - np.outer(s, s)
            grad_input[i] = np.dot(grad_output[i], J)
        
        return grad_input


class LogSoftmax(Module):
    """
    Applies LogSoftmax operator over the last dimension
    """
    def compute_output(self, input: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        return scipy.special.log_softmax(input, axis=1)

    def compute_grad_input(self, input: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        output = np.exp(scipy.special.log_softmax(input, axis=1))
        grad_input = grad_output - output * np.sum(grad_output, axis=1, keepdims=True)
        
        return grad_input
