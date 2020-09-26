import time
import Model 
import config 
import random 
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
        
        # Initialise member variable for copulas container
        self.copulas = None 
        self.pairs_dict = None
        
        # initialise loggers
        self.initialise_loggers()
        
        # Schedule model initialisation - Every week on Saturday at midnight
        self.Schedule.On(self.DateRules.Every(DayOfWeek.Saturday),
                         self.TimeRules.At(0,0),
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
        self.SP500_Tickers = ["XLK", "IYW"] # ["SPY","XLK", "VGT", "IYW", "IGV"]
        
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
        
        # fit the model
        model_generator = Model.ModelFactory() 
        if not config.ModelParameters.FIT_MULTIPROCESS: 
            for pair in pairs_dict.keys():
                data_packet1 = Data.get_returns(pair[0])
                data_packet2 = Data.get_returns(pair[1])
                
                t = time.time()
                copula_model = model_generator.get_model(data_packet1, data_packet2)
                elapsed = time.time() - t
                self.log_and_debug(f"{elapsed}s taken to fit model for {pair} on dataframe of shape {Data.shape}")
                self.copulas[pair] = copula_model
        else:
            # allocate cpu resource 
            num_workers = len(pairs_dict) if mp.cpu_count() > len(pairs_dict) else mp.cpu_count()
            self.Debug(f"Using {num_workers} processses to fit {len(pairs_dict)} models")
            
            # dispatch workers from pool
            async_results = []
            with mp.Pool() as pool: 
                for pair in pairs_dict.keys():
                    data_packet1 = Data.get_returns(pair[0])
                    data_packet2 = Data.get_returns(pair[1])
                    copula_model_getter = pool.apply_async(func=model_generator.get_model, 
                                                           args=(data_packet1, data_packet2))
                    async_results.append((pair, copula_model_getter))
            
            res = async_results[0]
            pair, copula_model_getter = res
            start = time.time()
            while not copula_model_getter.ready():
                time.sleep(5)
                elapsed = time.time() - start
                self.log_and_debug(f"Elapsed time: {elapsed}s")
                if elapsed > 60: 
                    self.log_and_debug(f"Elapsed time more than 60s")
                
            
            # wait for pool to return results
            t = time.time()
            for res in async_results:
                pair, copula_model_getter = res
                copula_model = copula_model_getter.get()
                self.copulas[pair] = copula_model
            elapsed = time.time() - t   

        self.log_and_debug(f"{elapsed}s taken to fit {len(pairs_dict)} models")
        

    def OnData(self, data):
        '''OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.
            Arguments:
                data: Slice object keyed by symbol containing the stock data
        '''
        if not self.copulas: 
            return
        else: 
            for pair in self.copulas.keys(): 
                sym1, sym2 = pair
                close1 = data.Bars[sym1].Close
                close2 = data.Bars[sym2].Close

                u = self.copulas[pair].price_to_marginal(price=close1, symbol=sym1)
                v = self.copulas[pair].price_to_marginal(price=close2, symbol=sym2)
                
                grid_width = 0.001
                mi_u_given_v, mi_v_given_u = self.copulas[pair].mispricing_index(u,v,grid_width)

                logmsg = f'''For {pair} at time {self.UtcTime}: 
                {sym1} close price = {close1}
                {sym1} marginal value = {u}
                {sym1} mispricing index = {mi_u_given_v}
                
                {sym2} close price = {close2}
                {sym2} marginal value = {v}
                {sym2} mispricing_index = {mi_v_given_u}
                '''
                self.log_and_debug(logmsg)


        
        # call mispricing_index method for debugging
        # width = 1e-3
        # n = 25
        # for i in range(n):
        #     u,v = random.uniform(0,1), random.uniform(0,1)
        #     mi = self.copulas.mispricing_index(u,v,width) 
            
        #     # mi is a tuple containing ( C(u|v), C(v|u) )
        #     self.log_and_debug(f"u: {u}, v: {v}, MI: {mi}")