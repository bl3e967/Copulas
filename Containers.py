from collections import namedtuple 

def unstack_df(df, column, level=0):
    return df[column].unstack(level=level)
    
# Your New Python File
class Data():
    
    def __init__(self, df, ohlcv):
        '''
        Args:
            df: pandas dataframe of historical data
            ohlcv: string specifying whether to use open, close, high, low, volume
            for the timeseries
        '''
        self.raw = df
        self._ohlcv = ohlcv
        self._timeseries = unstack_df(df,ohlcv) # unstack so columns are instrument tickers
    
    def __len__(self):
        return len(self._timeseries)
        
    @property
    def index(self):
        return self._timeseries.index
    
    @property
    def returns(self):
        return self._timeseries.diff()
    
    def get_correlations(self, type="kendall"):
        return self.returns.corr(method=type)
        
    
class ModelContainer():
    '''
    Container object for storing copula models according to the instrument pair 
    tickers. 
    Expects to receive a tuple of size two containing the ticker
    for the instruments being modelled, and its corresponding
    model. 
    '''
    def __init__(self):
        self._container_dict = {}
    
    def __getitem__(self, key:tuple): 
        std_key = self._enforce_key_structure(key)
        return self._container_dict[std_key]
        
    def __setitem__(self, key:tuple, model): 
        std_key = self._enforce_key_structure(key)
        self._container_dict[std_key] = model
     
    def _enforce_key_structure(self, pair:tuple): 
        return tuple(sorted(pair))
        
    def append(self, pair:tuple, model):
        key = self._enforce_key_structure(pair)
        self._container_dict(key, model)
    
    def update(self, new_dict): 
        self._container_dict.update(new_dict)