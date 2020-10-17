_YEAR = 365

class PortfolioParameters():
    # starting funds
    INIT_FUNDS = 100000

    # max proportion of funds that can be used. 
    MAX_ACCOUNT_RISK = 0.01

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