# Your New Python File
import math
import numpy as np

class AbsValMaxOne():
    def __init__(self):
        self._counter = 0
        self._max = 1.0
        self._min = -1.0
    
    @staticmethod
    def clipped_increment(curr, inc, val_min, val_max):
        return np.clip(curr + inc, a_min=val_min, a_max=val_max)
    
    def operation(self, val):
        if math.isnan(val):
            return self._counter
        
        self._counter = self.clipped_increment(self._counter, val, self._min, self._max)
        
        return self._counter
        
    def __call__(self, val):
        print(val)
        return self.operation(val)