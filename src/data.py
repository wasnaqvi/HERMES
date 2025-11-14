import pandas as pd
import numpy as np
from dataclasses import dataclass


def compute_leverage(arr: np.ndarray) -> float:
    """L = sqrt(sum_i (y_i - mean(y))^2)."""
    arr = np.asarray(arr, float)
    m = np.isfinite(arr)
    arr = arr[m]
    if arr.size < 2:
        return 0.0
    return float(np.sqrt(np.sum((arr - arr.mean())**2)))


@dataclass
class HermesData:
    '''
    Handles synthetic HERMES data from MCS etc.
    '''
    df: pd.DataFrame
    
    @classmethod
    def from_csv(cls, filepath: str) -> 'HermesData':
        df = pd.read_csv(filepath)
        return cls(
            df=df
        )
    
    @property
    def logM(self) -> pd.Series:
        return self.df['logM']

    @property
    def met(self) -> pd.Series:
        return self.df['log(X_H2O)']
    
    def uncertainty_upper(self):
        # can also use np.full_like if needed
        return self.df['log(X_H2O)_err_high']
    def uncertainty_lower(self):
         # can also use np.full_like if needed
        return self.df['log(X_H2O)_err_low']
    
    def subset_by_mass(self, min_mass: float, max_mass: float) -> 'HermesData':
        mask = (self.df['logM'] >= min_mass) & (self.df['logM'] <= max_mass)
        return HermesData(df=self.df[mask].reset_index(drop=True))

        
    




