import time
import scipy
import config 
import Containers
import numpy as np
from scipy import integrate
from scipy.stats import norm
import statsmodels.api as sm
from UtilFuncs import DataCleaningFuncs, Transformations, MICalculator
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
    
    # Use degree 1 for interpolation as larger degrees can overshoot to lead to negative pdf values
    # do not enforce bbox as it causes the interpolation to overshoot into negative values _BBOX = [0,1,0,1]
    INTERPOLATION_PARAMS = {
        "kx" : 1, 
        "ky" : 1
    }
    
    KDEKernel = sm.nonparametric.KDEMultivariate
    InterpModel = scipy.interpolate.RectBivariateSpline
            
    def __init__(self, x:tuple, y:tuple, auto_fit=True):
        '''
        Non-parametric bivairate copula model. 
        Args: 
            x: tuple, containing (Symbol:string, data:numpy.array)
            y: tuple, containing (Symbol:string, data:numpy.array)
            auto_fit: Run the model fitting pipeline upon initialisation if True. Else, manually trigger the 
            self.fit() function. 
        '''
        super().__init__()
        self.model = None 
        self.ecdf = None 
        self._fit_status = False 
        self._ecdf_dict = {}
        self._mispricing_index = MICalculator()
        
        # unpack data
        self.x_sym, self.y_sym = x[0], y[0]
        self.x_data, self.y_data = x[1], y[1]
        
        if auto_fit:
            self.fit()
        
    @staticmethod 
    def gaussian_transform(marginal_values): 
        '''
        Utility function for transforming marginal values in range [0,1]
        to [-inf, inf] using Gaussian inverse cdf. 
        '''
        transformed = norm.ppf(marginal_values)
        transf_inf_removed = DataCleaningFuncs.np_remove_inf_1D(transformed)
        return transf_inf_removed 
    
    def fit_ecdf(self, data):
        return ECDF(data)
        
    def fit(self):
        '''
        Fit KDEModel.
        
        Get values from marginal distributions for each asset. 
        Transform these marginal values using gaussian transformation. 
        Fit the model to the transformed values. 
        '''
        self._fit_status = False 
        
        fn_stack_data = lambda x,y: np.vstack((x,y)).T # Transpose to get (N,2) shape
        
        # TODO: Remove nan from returns so that we don't need to do data cleaning later
        
        # [0,1]^2 domain
        self._ecdf_dict[self.x_sym] = self.fit_ecdf(self.x_data)
        self._ecdf_dict[self.y_sym] = self.fit_ecdf(self.y_data) 
        x_marginal_U = self._ecdf_dict[self.x_sym](self.x_data)
        y_marginal_U = self._ecdf_dict[self.y_sym](self.y_data)
        
        # R^2 domain
        x_marginal_R = Transformations.gaussian_transform(x_marginal_U)
        y_marginal_R = Transformations.gaussian_transform(y_marginal_U)
        marginals_R = fn_stack_data(x_marginal_R, y_marginal_R) 
        
        # fit the KDE model to the unbounded values - around 17 seconds
        # self.log_and_debug(f"Fitting KDE model for {self.pair}")
        model, elapsed = self.timed_exec(self.KDEKernel, data=marginals_R, **self.KDE_PARAMS)
        # self.log_and_debug(f"KDE Model fit in {elapsed} seconds")
        
        # --- fit the interpolation model on the KDE Model ---
        
        # Generate mesh grid of values in R^2 domain
        _min = -3; _max = 3; w = 0.05
        x_valuesR = np.arange(_min, _max, w)
        y_valuesR = np.arange(_min, _max, w)
        mesh_xR, mesh_yR = np.meshgrid(x_valuesR, y_valuesR)
        rng = fn_stack_data(mesh_xR.ravel(), mesh_yR.ravel())
        # TODO: This part is the bottleneck - need to multiprocess here
        z_valuesR = model.pdf(data_predict=rng)
        
        # transform everything back to [0,1]^2 domain
        z_valuesU = np.array(z_valuesR / (norm.pdf(mesh_xR.ravel()) * norm.pdf(mesh_xR.ravel())))
        x_valuesU = norm.cdf(x_valuesR)
        y_valuesU = norm.cdf(y_valuesR)
        
        # Fit the interpolation model using the meshgrid 
        mesh_zU = z_valuesU.reshape(mesh_xR.shape)
        self.model = self.InterpModel(x_valuesU, y_valuesU, mesh_zU, **self.INTERPOLATION_PARAMS)
        
        # update internal status to indicate fit method was successful
        self._fit_status = True
        
        return 
    
    # TODO: Need to enforce u mapping to instrument x and v mapping to instrument y
    # it would be terrible if we had marginal value for instrument y being fed into u and vice versa
    # Merge the below two methods together so that calculating the mispricing index involves passing
    # price and correponding identifier info. 
    def price_to_marginal(self, symbol, price):
        '''
        Returns marginal probability value from ECDF distribution. 
        '''
        return self._ecdf_dict[symbol](price)

    def mispricing_index(self, u, v, delta=0.001):
        '''
        Calculate mispricing index C(u|v) by integrating 
        the pdf c(u,V) over u in range [0,1] for some value of V=v.
        
        Args:
            u: r.v. over which to integrate the copula. The integrand is the copula density function. 
            v: conditional r.v
            delta: grid width for numerical integration
            
        Returns:
            MI: mispricing index value
        '''
        return self._mispricing_index(u=u, v=v, 
                                      model=self.model, 
                                      delta=delta)

class ModelFactory(): 
    @staticmethod
    def get_model(x:tuple, y:tuple): 
        model = BivariateNonParametricCopula(x, y, auto_fit=True)
        return model