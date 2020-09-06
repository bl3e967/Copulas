import time
import Model 
import config 
import itertools
import Containers 
import multiprocessing as mp 
from UtilFuncs import CorrelationFuncs

class CopulasAlgorithm(QCAlgorithm):

    def Initialize(self):
        # Set Start Date so that backtest has 5+ years of data
        self.SetStartDate(2015, 1, 1)

        # No need to set End Date as the final submission will be tested
        # up until the review date

        # Set $1m Strategy Cash to trade significant AUM
        self.SetCash(100000)

        # Use the Alpha Streams Brokerage Model, developed in conjunction with
        # funds to model their actual fees, costs, etc.
        # Please do not add any additional reality modelling, such as Slippage, Fees, Buying Power, etc.
        self.SetBrokerageModel(AlphaStreamsBrokerageModel())
        
        # Define symbols used for Manual Universe Selection
        self.symbols = self.load_symbols()
        
        # Use Manual Universe Selection 
        self.SetUniverseSelection(ManualUniverseSelectionModel(self.symbols))
        
        # Initialise Universe Settings
        self._set_universe_settings()
        
        # set the model type we are going to use
        self.Model = Model.BivariateNonParametricCopula
        
        # initialise loggers
        self.initialise_loggers()
        
        # Schedule model initialisation - Every week on Saturday at midnight
        self.Schedule.On(self.DateRules.Every(DayOfWeek.Friday),
                         self.TimeRules.At(12,0),
                         self.fit_model)
        
    def log_and_debug(self, msg):
        self.Log(msg)
        self.Debug(msg)
    
    def initialise_loggers(self):
        '''
        We use a classmethod for any custom class objects that requires logging
        within the QC algorithm framework to store the algorithm instance as a 
        class variable
        '''
        self.Model.register_QC(algo=self)
    
    def _set_universe_settings(self): 
        '''
            Setup for Universe
        '''
        self.UniverseSettings.Resolution = config.ModelParameters.RESOLUTION
        # self.UniverseSettings.Leverage =  
        # self.UniverseSettings.FillForward =  
        # self.UniverseSettings.MinimumTimeInUniverse =  
        # self.UniverseSettings.ExtendedMarketHours =  
    
    def _get_historical_data(self, symbols, lookback, resolution, ohlcv): 
        '''
        Get historical data for our algorithm
        '''
        df = self.History(symbols, lookback, resolution)
        return Containers.Data(df, ohlcv=ohlcv)

    def load_symbols(self): 
        '''
            Load pre-defined sets into the instruments universe
            
            Returns: 
                Tickers: list of tickers used in this universe 
        '''
        # Symbols list
        symbols = []
        
        # SPY ETFs
        self.SP500_Tickers = ["SPY","XLK", "VGT", "IYW", "IGV"]
        
        for ticker in self.SP500_Tickers: 
            
            # Add the tickers to algorithm
            symbol = self.AddEquity(ticker, self.UniverseSettings.Resolution)
            
            # record symbols used 
            symbols.append(self.Symbol(ticker))
        
        return symbols
    
    def fit_model(self): 
        '''
        Return a CopulaModel object for each asset data
        '''
        
        # initialise dictionary for containing copula models
        self.copulas = Containers.ModelContainer()
        
        Data = self._get_historical_data(self.symbols, 
                                         config.ModelParameters.LOOKBACK, 
                                         config.ModelParameters.RESOLUTION,
                                         config.ModelParameters.OHLCV)
        self.Debug(f"Received {len(Data)} datapoints ranging from {Data.index[0]} to {Data.index[-1]}")
        self.Log("Received {} datapoints".format(len(Data)))
        
        # Find the pairs that satisfy our correlation criteria
        # Criteria: Kendall Tau correlation > threshold 
        corr = Data.get_correlations()
        pairs_dict = CorrelationFuncs.correlation_above_thresh(corr, config.ModelParameters.CORRELATION_THRESHOLD)
        
        # fit Copula model for each pair - TODO: Multiprocess this part
        if not config.ModelParameters.FIT_MULTIPROCESS: 
            for pair in pairs_dict.keys():
                returns = Data.returns[list(pair)].to_numpy()
                t = time.time()
                copula_model = Model.BivariateNonParametricCopula(returns, pair)
                elapsed = time.time() - t
                self.copulas[pair] = copula_model
        else:
            # allocate cpu resource 
            num_workers = len(pairs_dict) if mp.cpu_count() > len(pairs_dict) else mp.cpu_count()
            self.Debug(f"Using {num_workers} processses to fit {len(pairs_dict)} models")
            
            # fit models
            with mp.Pool(processes=num_workers) as pool:
                pairs_sorted = sorted(pairs_dict.keys())
                args_list = [(Data.returns[list(pair)].to_numpy(), pair) for pair in pairs_sorted]
                t = time.time()
                models_list = pool.starmap(Model.BivariateNonParametricCopula, args_list)
                elapsed = time.time() - t 
             
            # add to container
            self.copulas.update(dict(zip(pairs_sorted, models_list)))
        
        self.log_and_debug(f"{elapsed}s taken to fit {len(pairs_dict)} models")
            
        return None 
        
    def OnData(self, data):
        '''OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.
            Arguments:
                data: Slice object keyed by symbol containing the stock data
        '''

        # if not self.Portfolio.Invested:
        #   self.SetHoldings("SPY", 1)