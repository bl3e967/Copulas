_YEAR = 365

class ModelParameters(): 
    LOOKBACK = timedelta(5*_YEAR) # 5 years 
    RESOLUTION = Resolution.Daily
    OHLCV = 'close'
    CORRELATION_THRESHOLD = 0.7
    CORRELATION_METHOD = "kendall"