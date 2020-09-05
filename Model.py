import time
import scipy
import config 
import Containers
import numpy as np
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
        rmsg = f"[{self.qc.Time}] >> {msg}"
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
    
    INTERPOLATION_PARAMS = {
        "kx" : 1, 
        "ky" : 1
    }
    
    KDEKernel = sm.nonparametric.KDEMultivariate
    InterpModel = scipy.interpolate.RectBivariateSpline
    
    def __init__(self, returns, pair:tuple, auto_fit=True):
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
        self.pair = pair
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
    
    @staticmethod 
    def gaussian_transform(marginal_values): 
        '''
        Utility function for transforming marginal values in range [0,1]
        to [-inf, inf] using Gaussian inverse cdf. 
        '''
        transformed = norm.ppf(marginal_values)
        transf_inf_removed = DataCleaningFuncs.np_remove_inf_1D(transformed)
        return transf_inf_removed 
        
    def get_marginal_values(self, data):
        # first model the marginal distributions using ECDF
        ecdf = ECDF(data)
        bounded_marginal_values = ecdf(data)
        
        # marginals domain is bounded between [0,1]. 
        # Use Gaussian transform so that our domain is [-inf, inf]
        unbounded_marginal_values = self.gaussian_transform(bounded_marginal_values)
        
        return unbounded_marginal_values
    
    def get_marginal_val(self, data): 
        ecdf = ECDF(data) # initialise the ECDF object
        bounded_marginal_val = ecdf(data) # transform data to marginal values
        return bounded_marginal_val
        
    def fit(self):
        '''
        Fit KDEModel.
        
        Get values from marginal distributions for each asset. 
        Transform these marginal values using gaussian transformation. 
        Fit the model to the transformed values. 
        '''
        
        fn_stack_data = lambda x,y: np.vstack((x,y)).T # Transpose to get (N,2) shape
        
        x, y = self.returns[:,0], self.returns[:,1]
        
        # [0,1]^2 domain
        x_marginal_U = self.get_marginal_val(x)
        y_marginal_U = self.get_marginal_val(y)
        
        # R^2 domain
        x_marginal_R = self.gaussian_transform(x_marginal_U)
        y_marginal_R = self.gaussian_transform(y_marginal_U)
        marginals_R = fn_stack_data(x_marginal_R, y_marginal_R) 
        
        # fit the KDE model to the unbounded values
        self.log_and_debug(f"Fitting KDE model for {self.pair}")
        model, elapsed = self.timed_exec(self.KDEKernel, data=marginals_R, **self.KDE_PARAMS)
        self.log_and_debug(f"KDE Model fit in {elapsed} seconds")
        
        # --- fit the interpolation model on the KDE Model ---
        
        # Generate mesh grid of values in R^2 domain
        _min = -3; _max = 3; w = 0.01
        x_valuesR = np.arange(_min, _max, w)
        y_valuesR = np.arange(_min, _max, w)
        mesh_xR, mesh_yR = np.meshgrid(x_valuesR, y_valuesR)
        rng = fn_stack_data(mesh_xR.ravel(), mesh_yR.ravel())
        z_valuesR = model.pdf(data_predict=rng)
        
        # transform back to [0,1]^2 domain
        z_valuesU = np.array(z_valuesR / (norm.pdf(mesh_xR.ravel()) * norm.pdf(mesh_xR.ravel())))
        
        # Fit the interpolation model using the meshgrid 
        
    def mispricing_index(self, p1, p2):
        pass