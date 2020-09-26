import random 
import numpy as np 
from scipy.stats import norm
from dataclasses import dataclass 

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

class MICalculator():
    
    @staticmethod 
    def get_range_gaussian_transformation(val=None, width=None): 
        '''
            Using this is recommended for copulas modelling as this 
            has a higher sample density in the extremes which is the 
            region of interest for a copula. 

            Args: 
                val: The range upper limit. If None, defaults to 1. 
                width: The interval for the range. If not specified, 
                raises a ValueError. 
            Returns: 
                Numpy array within range [0,val] with spacing = width
        '''
        # std dev vals 
        _MINRNG = -4
        _MAXRNG_default = 4
        # take account for None and inf values
        cond = ((val is None) or (norm.ppf(val) > _MAXRNG_default))
        _MAXRNG = _MAXRNG_default if cond else norm.ppf(val)
        linrng = np.arange(_MINRNG, _MAXRNG, width)
        return norm.cdf(linrng)
    
    @staticmethod 
    def integrate(yval, xval): 
        '''Wrapper for numpy numerical integration via trapezoidal rule'''
        return np.trapz(yval, xval)
        
    def __call__(self, u:float, v:float, model, delta:float): 
        '''
            Calculate the mispricing index for a given u,v value
            Args: 
                u: Scalar bounded in range [0,1]. 
                v: Scalar bounded in range [0,1]. 
                model: The copula model
                delta: width for numerical integration. Larger value 
                leads to faster computation for less accuracy
                
            Returns:
                C_uv/Z_uv: C(U|V) / Z where Z is the normalisation constant and 
                C(U|V) is the non-normalised conditonal copula function.
                C_vu/Z_vu: C(V|U) / Z where Z is the normalisation constant and 
                C(V|U) is the non-normalised conditonal copula function.
        '''
        _FULLRNG = self.get_range_gaussian_transformation(width=delta)
        _URNG = self.get_range_gaussian_transformation(u, width=delta)
        _VRNG = self.get_range_gaussian_transformation(v, width=delta)
        
        # c(u,v=v') & c(u=u',v)
        c_uv, c_vu = model.ev(_URNG, v), model.ev(u, _VRNG)
        
        # copula values for normalisation const
        z_uv, z_vu = model.ev(_FULLRNG, v), model.ev(u, _FULLRNG)
        
        # non-normalised conditional copula values
        C_uv, C_vu = self.integrate(c_uv, _URNG), self.integrate(c_vu, _VRNG)
        
        # normalisation constants 
        Z_uv, Z_vu = self.integrate(z_uv, _FULLRNG), self.integrate(z_vu, _FULLRNG)
        
        # MI values 
        return C_uv/Z_uv, C_vu/Z_vu