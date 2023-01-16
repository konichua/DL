import numpy as np

class Sgd:
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        return weight_tensor - self.learning_rate * gradient_tensor

class SgdWithMomentum:
    def __init__(self, learning_rate: float, momentum_rate: float):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v_prev = 0
        self.v_cur = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.v_cur = self.momentum_rate * self.v_prev - self.learning_rate * gradient_tensor
        self.v_prev = self.v_cur
        return weight_tensor + self.v_cur

class Adam:
    def __init__(self, learning_rate: float, mu: float, rho: float):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.iter = 1
        self.v_prev = 0
        self.v_cur = 0
        self.r_prev = 0
        self.r_cur = 0
    
    def calculate_update(self, weight_tensor, gradient_tensor):
        self.v_cur = self.mu * self.v_prev + (1 - self.mu) * gradient_tensor
        self.r_cur = self.rho * self.r_prev + (1 - self.rho) * gradient_tensor ** 2
        v_corr = self.v_cur / (1 - self.mu ** self.iter)
        r_corr = self.r_cur / (1 - self.rho ** self.iter)
        self.v_prev = self.v_cur
        self.r_prev = self.r_cur
        self.iter += 1
        return weight_tensor - self.learning_rate * v_corr / (np.sqrt(r_corr) + np.finfo(float).eps)
