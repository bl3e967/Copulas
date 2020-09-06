import numpy as np 
from scipy.stats import norm

# It seems you cannot have just function definitions that can be imported
# into other scripts in QC. So define everything in classes as staticmethods. 

class DataCleaningFuncs(): 
    
    @staticmethod 
    def np_remove_nan(x): 
        return x[np.logical_not(np.isnan(x).any(axis=1))]

    @staticmethod 
    def np_remove_inf(x):
        return x[np.logical_not(np.isinf(x).any(axis=1))]
    
    @staticmethod 
    def np_remove_inf_1D(x):
        return x[~np.isinf(x)]
        
    @staticmethod 
    def np_remove_nan_1D(x):
        return x[~np.isinf(x)]
    
class SamplingFuncs():
    
    @staticmethod 
    def sample_from_bivariate(x_domain, y_domain, weights, n_samples):
        x_domain = x_domain.ravel()
        y_domain = y_domain.ravel()
        weights = np.nan_to_num(weights.ravel()) # account for nans
        
        population = np.array([x_domain, y_domain]).T
        print(population)
        return np.array(random.choices(population, weights, k=n_samples))


class CorrelationFuncs():
    
    @staticmethod 
    def correlation_above_thresh(corr_df, thresh, logger=None): 
        '''
        Args:
            corr_df: pandas dataframe object containing correlation values
        '''
            
        # change from df to series to be able to filter based on threshold
        corr_sr = (corr_df.where(np.triu(np.ones(corr_df.shape), k=1).astype(np.bool))
                     .stack()
                     .sort_values(ascending=False))
        
        filtered_corr = corr_sr[corr_sr > thresh]
        corr_per_pair_dict = dict(zip(filtered_corr.index, filtered_corr.values))
        
        return corr_per_pair_dict

class Transformations():
    '''
        Functions for data transformations. 
    '''
    
    @staticmethod 
    def gaussian_transform(rng): 
        '''
        Utility function for transforming values in range [0,1]
        to [-inf, inf] using Gaussian inverse cdf. 
        
        Values outside the range [0,1] will raise an exception. 
        
        Args: 
            rng: values in [0,1]
        '''
        transformed = norm.ppf(rng)
        transf_inf_removed = DataCleaningFuncs.np_remove_inf_1D(transformed)
        return transf_inf_removed