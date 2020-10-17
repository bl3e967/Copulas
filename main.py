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
        self.SetCash(config.PortfolioParams.INIT_FUNDS)

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
        
        # Intialise dictionary to contain rolling windows for each pair
        self.last_prices = {}
        
        # initialise loggers
        self.initialise_loggers()
        
        # Schedule model initialisation - Every week on Saturday at midnight
        self.Schedule.On(self.DateRules.Every(DayOfWeek.Saturday),
                         self.TimeRules.At(0,0),
                         self.initialise_models)
        
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
        
    def initialise_models(self): 
        '''
        Return a CopulaModel object for each asset data
        '''
        
        # initialise dictionary for containing copula models
        self.copulas = Containers.ModelContainer()
        
        Data = self._get_historical_data(self.symbols, 
                                         config.ModelParameters.LOOKBACK, 
                                         config.ModelParameters.RESOLUTION,
                                         config.ModelParameters.OHLCV)
        self.log_and_debug(f"Received {len(Data)} datapoints ranging from {Data.index[0]} to {Data.index[-1]}")
              
        # Find the pairs that satisfy our correlation criteria
        # Criteria: Kendall Tau correlation > threshold 
        corr = Data.get_correlations()
        pairs_dict = CorrelationFuncs.correlation_above_thresh(corr, config.ModelParameters.CORRELATION_THRESHOLD)
        
        # fit the model
        model_generator = Model.ModelFactory() 
        if not config.ModelParameters.FIT_MULTIPROCESS: 
            t = time.time()
            for pair in pairs_dict.keys():
                data_packet1 = Data.get_returns(pair[0])
                data_packet2 = Data.get_returns(pair[1])
                copula_model = model_generator.get_model(data_packet1, data_packet2)
                self.copulas[pair] = copula_model
            elapsed = time.time() - t
            self.log_and_debug(f"{elapsed}s taken to fit {len(pairs_dict)} models")
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
                    get_proc = pool.apply_async(func=model_generator.get_model, 
                                                args=(data_packet1, data_packet2))
                    async_results.append((pair, get_proc))
            
                # wait for pool to return results
                t = time.time()
                for res in async_results:
                    pair, model_getter = res
                        copula_model = model_getter.get()
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
        
        # daily trades only
        if self.Time.day == self.day:
            return
        
        for pair in self.copulas.keys():
            sym1, sym2 = pair
            close1 = data.Bars[sym1].Close
            close2 = data.Bars[sym2].Close
            
            # TODO: We could have a rolling window of prices from which we calculate the returns
            # From this we calculate the ECDF and copula. Using some distance measure from the 
            # current and new distribution, we can quantify how much the distribution has changed 
            # and therefore judge the optimal frequency of refitting needed to keep the model up-to-date.
            # Use just a dict to retain the last price for now. 
            
            CALC_MI = True
            # handle the case when we first encounter this pair
            try: 
                last_close1 = self.last_prices[sym1]
                last_close2 = self.last_prices[sym2]
            except KeyError:
                self.last_prices[sym1] = close1 
                self.last_prices[sym2] = close2 
                CALC_MI = False
            
            if CALC_MI:
                d1 = close1 - last_close1
                u = self.copulas[pair].price_to_marginal(price=d1, symbol=sym1)
            
                d2 = close2 - last_close2
                v = self.copulas[pair].price_to_marginal(price=d2, symbol=sym2)
            
                grid_width = 0.001
                mi_u_v, mi_v_u = self.copulas[pair].mispricing_index(u,v,grid_width)
                
                # u over-priced and v under-priced
                u_overpriced = mi_u_v > config.TradeParameters.MI_UPPER_THRESH
                v_underpriced = mi_v_u < config.TradeParameters.MI_LOWER_THRESH
                
                # u under-priced and v over-priced
                u_underpriced = mi_u_v < config.TradeParameters.MI_LOWER_THRESH
                v_overpriced = mi_v_u > config.TradeParameters.MI_UPPER_THRESH
                
                if u_overpriced and v_underpriced: 
                    self.Debug(f"{self.Time}: Sell u and Buy v - C(U|V) : {mi_u_v}, C(V|U) : {mi_v_u}")

                    # ratio for equivalent exposure to each leg of pair - price(sym1) / price(sym2)
                    p_ratio = close1 / close2 
                    u_quantity = self.CalculateOrderQuantity(sym1, self.max_account_risk)
                    v_quantity = -p_ratio * u_quantity 

                    self.MarketOrder(sym1, u_quantity, self._ASYNC_ORDER)
                    self.MarketOrder(sym2, v_quantity, self._ASYNC_ORDER)

                elif u_underpriced and v_overpriced: 
                    self.Debug(f"{self.Time}: Buy u and Sell v - C(U|V) : {mi_u_v}, C(V|U) : {mi_v_u}")

                    p_ratio = close1 / close2
                    u_quantity = -self.CalculateOrderQuantity(sym1, self.max_account_risk)
                    v_quantity = -p_ratio * u_quantity

                    self.MarketOrder(sym1, u_quantity, self._ASYNC_ORDER)
                    self.MarketOrder(sym2, v_quantity, self._ASYNC_ORDER)
            
            self.day = self.Time.day