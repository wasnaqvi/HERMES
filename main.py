from src import Model
from src import Survey
import numpy as np
import pandas as pd

dummy_model=Model(version="0.0.1",x=np.linspace(0,10,10),y_obs=np.random.normal(5,2,10),
                  y_err_low=np.random.uniform(0.1,0.5,10),y_err_high=np.random.uniform(0.1,0.5,10))
dummy_model.build_model(n_samples=500,random_seed=42,centered=True)

# clean the MCS into arrays here. GET data in the shape that I need for the array class.
data=pd.read_csv('../results/Ariel_MCS_Known_202')
'''
Get hermes_data here. 
Get HM_scatter, HM_survey in here.

Make 
'''