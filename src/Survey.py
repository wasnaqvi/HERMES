from .Model import Model
import numpy as np
class Survey(Model):
    
    def __init__(self,version,data:list):
        x, y_obs, y_err_low, y_err_high = zip(*data)
        super().__init__(version,x,y_obs,y_err_low,y_err_high)
        self.surveys = []
        self.plots = []
    
    @staticmethod
    def compute_leverage(arr):
        return lambda arr: float(np.sum((arr - np.mean(arr))**2))