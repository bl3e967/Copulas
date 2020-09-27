_YEAR = 365

class ModelParameters(): 
    LOOKBACK = timedelta(5*_YEAR) # 5 years 
    RESOLUTION = Resolution.Daily
    OHLCV = 'close'
    CORRELATION_THRESHOLD = 0.7
    CORRELATION_METHOD = "kendall"
    FIT_MULTIPROCESS = True

class TradeParameters():
    MI_UPPER_THRESH = 0.9
    MI_LOWER_THRESH = 1 - MI_UPPER_THRESH