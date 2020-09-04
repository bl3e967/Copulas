import itertools
import Model 
import config 
import Containers 
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
        
        # Initialise our model
        self.initialise_model()
        
    def initialise_loggers(self):
        '''
        We use a classmethod for any custom class objects that requires logging
        within the QC algorithm framework to store the algorithm instance as a 
        class variable
        '''
        self.Model.register_QC(algo=self)
    
    def initialise_model(self): 
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
        
        # fit Copula model for each pair
        for pair in pairs_dict.keys():
            returns = Data.returns[list(pair)]
            copula_model = Model.BivariateNonParametricCopula(returns)
            self.copulas[pair] = copula_model
    
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
        
    def OnData(self, data):
        '''OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.
            Arguments:
                data: Slice object keyed by symbol containing the stock data
        '''

        # if not self.Portfolio.Invested:
        #   self.SetHoldings("SPY", 1)