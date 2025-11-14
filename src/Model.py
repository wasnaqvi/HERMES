import numpy as np
import arviz as az
import pymc as pm
import matplotlib.pyplot as plt
class Model:
    '''
    Introducing HERMES: HiERarchical bayesian Models for Exoplanet Science. 
    
    Only uncertainties are in Y. 2D uncertainties, covariance, and multiple predictor model coming soonish!!
    '''
    def __init__(self,version,x,y_obs,y_err_low,y_err_high):
        self.version = version
        self.x = x
        self.y_obs = y_obs
        self.y_err_low = y_err_low
        self.y_err_high = y_err_high


        
    def build_model(self,n_samples,random_seed=42, centered=True):
        '''
        2D Bayesian Linear Model with pymc.
        '''
        if centered:
            x = np.asarray(self.x)
            y_obs = np.asarray(self.y_obs)
            
            meas_sigma = 0.5 * (np.abs(self.y_err_low) + np.abs(self.y_err_high))
            x_mean = x.mean()
            x_centered = x - x_mean

            # ptp is the peak to peak function. delta x and delta y.
            delta_x = np.ptp(x_centered)
            delta_y = np.ptp(y_obs)

            with pm.Model() as model:
                    alpha = pm.Normal('alpha', mu=y_obs.mean(), sigma=y_obs.std())
                    beta  = pm.Normal('beta',  mu=0.0, sigma=(delta_y/np.abs(delta_x)))
                    epsilon = pm.HalfNormal('epsilon', sigma=y_obs.std() or 1.0)

                    mu = alpha + beta * x_centered
                    noise= pm.Normal('noise', sigma=epsilon, shape=len(x))

                    y = pm.Normal('y', mu=mu, sigma=meas_sigma, observed=y_obs)

                    idata = pm.sample(n_samples, tune=1000, target_accept=0.9,
                          return_inferencedata=True, random_seed=random_seed)
                    graph = pm.model_to_graphviz(model)
            
            return idata, graph, f"Ran {self.y_obs} vs {self.x} with centered model"
            
        else: 
                with pm.Model() as model:
                    # rewrite priors. Weakly informative but adjustable.
                        alpha = pm.Normal('alpha', mu=np.mean(y_obs), sigma=np.std(y_obs) / np.sqrt(len(y_obs)))
                        beta  = pm.Normal('beta', mu=0, sigma=(np.max(y_obs) - np.min(y_obs)) / (np.max(x) - np.min(x)))
                        epsilon = pm.HalfNormal('epsilon', sigma= np.std(y_obs) / np.sqrt(len(y_obs)))
                        noise= pm.Normal('noise',sigma= epsilon, shape=len(x))
                        mu = alpha + beta * x + epsilon
        
                    # Observed sigma 
                        obs_sigma = pm.math.sqrt(meas_sigma**2 + sigma**2)
                        y = pm.Normal('y', mu=mu, sigma=obs_sigma, observed=y_obs)
        
                        idata = pm.sample(2000, tune=1000, target_accept=0.9, return_inferencedata=True, random_seed=random_seed)
                        graph=pm.model_to_graphviz(model)
                
                return idata, graph, f"Ran {self.y_obs} vs {self.x} with non-centered model"
        
        
    def multi_dimensional_model(self):
        '''
        3D Bayesian Linear Model with pymc.
        Goal is to have stellar metallicity + uncertainties added from the Ariel List of Targets
        '''
        pass
    
   
    