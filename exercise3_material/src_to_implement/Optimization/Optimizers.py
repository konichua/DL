import numpy as np

class Optimizer: # + regularization deriv
    def __init__(self):
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer

class Sgd(Optimizer):
    def __init__(self, learning_rate: float):
        super().__init__()
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        basic_update = weight_tensor - self.learning_rate * gradient_tensor
        if self.regularizer:
            return basic_update - self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
        return basic_update

class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate: float, momentum_rate: float):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v_prev = 0
        self.v_cur = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.v_cur = self.momentum_rate * self.v_prev - self.learning_rate * gradient_tensor
        self.v_prev = self.v_cur
        if self.regularizer:
            return weight_tensor + self.v_cur - self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
        return weight_tensor + self.v_cur

class Adam(Optimizer):
    def __init__(self, learning_rate: float, mu: float, rho: float):
        super().__init__()
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
        basic_update = weight_tensor - self.learning_rate * v_corr / (np.sqrt(r_corr) + np.finfo(float).eps)
        if self.regularizer:
            return basic_update - self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
        return basic_update
