from src import Model
from src import Survey
import numpy as np

dummy_model=Model(version="0.0.1",x=np.linspace(0,10,10),y_obs=np.random.normal(5,2,10),
                  y_err_low=np.random.uniform(0.1,0.5,10),y_err_high=np.random.uniform(0.1,0.5,10))
dummy_model.build_model(n_samples=500,random_seed=42,centered=True)