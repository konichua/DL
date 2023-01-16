import numpy as np

class BaseLayer:
    def __init__(self):
        self.trainable = False
        self.weights = None
        self._optimizer = None
    