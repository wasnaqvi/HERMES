 def plot_posterior():
        '''
        prior vs truth vs posterior samples plots
        '''
        
        pass
    
def summary(self, idata):
        '''
        Summary of the model fit using ArviZ
        '''
        return az.summary(idata, hdi_prob=0.95)
    
def generate_report(self):
        '''
        Generate a report with all the plots and Model details
        return a pdf file 
        
        
        '''
        pass