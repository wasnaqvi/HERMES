from .Model import Model
from .data import HermesData
import numpy as np
import pandas as pd

from collections.abc import Iterable

class Survey():
    
    def __init__(self, dataset: HermesData, idx: Iterable[int], survey_id: int | None = None):
        """
        Parameters
        ----------
        dataset : HermesData
            The full HERMES dataset.
        idx : iterable of int / pd.Index
            Row indices in dataset.df that belong to this survey.
        survey_id : 
            for bookkeeping.
        """
        self._dataset = dataset
        self._idx = pd.Index(idx)
        self.survey_id = survey_id
        
    @staticmethod
    def compute_leverage(arr):
        '''
        Nic and Ben Leverage
        eh dont need this here
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
        goes in analysis.py
        plot prior vs truth vs posterior samples plots for each survey size.
        You inherit the plot_posterior from the Bayesian linear Model use that
        
        '''
        pass
    
    def leverage_analysis(self):
        '''
        leverage analysis for each survey size.
        Nah. Dont need this here.
        '''
        pass
    
