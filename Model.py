import time 
import config 
import Containers
import numpy as np
import timestamp 
import statsmodels.api as sm
from UtilFuncs import DataCleaningFuncs
from scipy.stats import norm
from statsmodels.distributions.empirical_distribution import ECDF

class QCBase():
    def __init__(self):
        if not self.qc: 
            raise ValueError(f"No logger has been registered for class {self.__class__.__name__}")
            
    @classmethod
    def register_QC(cls, algo=None):
        cls.qc = algo
        
    def add_timestamp(self, msg):
        rmsg = f"[{timestamp.now()}] >> {msg}"
        return rmsg
    
    def debug(self, msg):
        self.qc.Debug(self.add_timestamp(msg))
    
    def log(self, msg):
        self.qc.Log(self.add_timestamp(msg))
    
    def log_and_debug(self, msg):
        nmsg = self.add_timestamp(msg)
        self.qc.Log(nmsg)
        self.qc.Debug(nmsg)
        
    def timed_exec(self, func, *args, **kwargs): 
        t = time.time()
        ret = func(*args, **kwargs)
        elapsed = time.time() - t
        return ret, elapsed
        

class BivariateNonParametricCopula(QCBase):
    KDE_PARAMS = {
        "var_type" : 'cc', 
        'bw' : 'cv_ml'
    }
    
    KDEKernel = sm.nonparametric.KDEMultivariate
    
    def __init__(self, returns, auto_fit=True):
        '''
        Non-parametric bivairate copula model. 
        Args: 
            returns: Historical returns price series for an asset. np.ndarray. Should be of 
            dimensions (N,2) where where N is the number of history samples and 2 is the number of assets.
            auto_fit: Run the model fitting pipeline upon initialisation if True. Else, manually trigger the 
            self.fit() function. 
            
        '''
        super().__init__()
        
        # check input for shape - raise ValueError if shape is incorrect
        self._validate_data(returns)
        
        # unpack data
        self.pair = tuple(returns.columns)
        self.returns = returns
        
        # run the model fitting pipeline if told to do so
        if auto_fit: 
            self.fit()
        
    
    def _validate_data(self, data): 
        x, y = data.shape
        if y > 2: 
            raise ValueError(f"Shape of input data is {data.shape}. Expected shape is (N,2) where N is number of"\
            + " history samples and 2 is the number of assets.")
        return None 
        
    def fit_marginal_dist(self, ticker):
        return ECDF(self.returns[ticker])
    
    @staticmethod 
    def gaussian_transform(marginal_values): 
        '''
        Utility function for transforming marginal values in range [0,1]
        to [-inf, inf] using Gaussian inverse cdf. 
        '''
        transformed = norm.ppf(marginal_values)
        transf_inf_removed = DataCleaningFuncs.np_remove_inf_1D(transformed)
        return transf_inf_removed 
        
    def get_marginal_values(self, ticker):
        # first model the marginal distributions using ECDF
        ecdf = self.fit_marginal_dist(ticker)
        returns = self.returns[ticker]
        bounded_marginal_values = ecdf(returns)
        
        # marginals domain is bounded between [0,1]. 
        # Use Gaussian transform so that our domain is [-inf, inf]
        unbounded_marginal_values = self.gaussian_transform(bounded_marginal_values)
        
        return unbounded_marginal_values
        
    def fit(self):
        x, y = self.pair 
        x_marginal = self.get_marginal_values(x)
        y_marginal = self.get_marginal_values(y)
        
        self.debug(x_marginal.shape)
        self.debug(y_marginal.shape)
        
        # # fit the KDE model to the unbounded values
        # self.log_and_debug("Fitting KDE model")
        # model, elapsed = self.timed_exec(kernel, data=marginals, **self.KDE_PARAMS)
        # self.log_and_debug(f"KDE Model fit in {elapsed} seconds")