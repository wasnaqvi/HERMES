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
        '''
        Nic and Ben Leverage
        '''
        return lambda arr: float(np.sum((arr - np.mean(arr))**2))
    
    def run_survey_samples(self,n_samples=5, survey_sizes:list=[]):
        '''
        the goal is to run the model here(super.build_model for each survey in the survey sizes list)
        should return the dataframe for all the survey sizes
        
        '''
        
        pass
    
    def plot_survey_results(self):
        '''
        plot prior vs truth vs posterior samples plots for each survey size.
        You inherit the plot_posterior from the Bayesian linear Model use that
        
        '''
        pass
    
    def leverage_analysis(self):
        '''
        leverage analysis for each survey size.
        '''
        pass
    
    def generate_report(self):
        '''
        generate a report for the survey results.
        '''
        pass