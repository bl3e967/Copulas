import time
import Model 
import config 
import random 
import itertools
import Containers 
import pandas as pd 
import multiprocessing as mp 
from datetime import datetime, timedelta
from UtilFuncs import CorrelationFuncs
from sklearn import linear_model 

_TICKERS = ["SPY","XLK", "VGT", "IYW", "IGV"]
MI_THRESHOLD = 0.9

class Parameter():
    def __init__(self,
                resolution,
                start_year, 
                start_month=1, 
                start_day=1,
                n_years=0,
                n_months=0, 
                n_days=0): 
        
        self.Resolution = resolution
        
        end_year = start_year + n_years
        end_month = start_month + n_months
        end_day = start_day + n_days 
        self.Start = datetime(start_year,start_month,start_day,9,30,0)
        self.End = datetime(end_year, end_month, end_day,16,30,0)

class NotebookMain():
    '''
    Simulates the main function used in QC Algorithm. 
    '''
    CORRELATION_THRESHOLD = 0.7
    
    def __init__(self, qb:"QuantBook"):
        # set the model type we are going to use
        self.Model = Model.BivariateNonParametricCopula
        self.Model.register_QC(qb) # WARNING: this will throw if we start logging inside models. 
        self.qb = qb 
    
    def _get_historical_data(self, param:"Parameter"):
        PARAM = param 

        # Specify list of correlated tickers for S&P 500 
        tickers = _TICKERS

        # register tickers to quantbook
        for ticker in tickers: 
            self.qb.AddEquity(ticker)

        print("Loading historical data...")
        # get historical prices
        history = self.qb.History(tickers, PARAM.Start, PARAM.End, PARAM.Resolution)

        print("Preparing historical data...")

        # create data object
        close = Containers.Data(history, "close")
        print("Done")
        
        return close

    def initialise_models(self, params): 
        '''
        Return a CopulaModel object for each asset data
        '''
        self.PARAM = params 

        # initialise dictionary for containing copula models
        self.copulas = Containers.ModelContainer()

        self.history = self._get_historical_data(self.PARAM)

        # Find the pairs that satisfy our correlation criteria
        # Criteria: Kendall Tau correlation > threshold 
        corr = self.history.get_correlations()
        pairs_dict = CorrelationFuncs.correlation_above_thresh(corr, self.CORRELATION_THRESHOLD)

        # fit the model
        print("Fitting model")
        print(f"Multiprocessing: {config.ModelParameters.FIT_MULTIPROCESS}")
        model_generator = Model.ModelFactory() 
        if not config.ModelParameters.FIT_MULTIPROCESS: 
            t = time.time()
            for pair in pairs_dict.keys():
                data_packet1 = self.history.get_returns(pair[0])
                data_packet2 = self.history.get_returns(pair[1])
                copula_model = model_generator.get_model(data_packet1, data_packet2)
                self.copulas[pair] = copula_model
            elapsed = time.time() - t
            print(f"{elapsed}s taken to fit {len(pairs_dict)} models")
        else:
            # allocate cpu resource 
            num_workers = len(pairs_dict) if mp.cpu_count() > len(pairs_dict) else mp.cpu_count()
            print(f"Using {num_workers} processses to fit {len(pairs_dict)} models")

            # dispatch workers from pool
            async_results = []
            with mp.Pool() as pool: 
                for pair in pairs_dict.keys():
                    data_packet1 = self.history.get_returns(pair[0])
                    data_packet2 = self.history.get_returns(pair[1])
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

        print(f"{elapsed}s taken to fit {len(pairs_dict)} models")
    
    def get_scaling_factor(self, pair):
        sym1, sym2 = pair
        ret1 = self.history.returns[sym1].to_numpy().reshape(-1,1)
        ret2 = self.history.returns[sym2].to_numpy().reshape(-1,1)
        
        reg_model = linear_model.LinearRegression().fit(ret1, ret2)
        
        plt.scatter(ret1, ret2)
        return reg_model.coef_[0][0]
        
        
def long_or_short(x):
    if x > MI_THRESHOLD:
        return -1
    elif x < 1 - MI_THRESHOLD:
        return 1
    else:
        return 0



class Backtest():
    def __init__(self, qb:"QuantBook", main:NotebookMain):
        self.qb = qb 
        self.main = main
        self.data = None
        self.timeseries = None 
        
    def load_backtest_data(self, n_days, ohlcv="close"):
        '''This will load data after the end of the training data for the specified number of days'''
        print("Loading test data")
        history = self.qb.History(_TICKERS, self.main.PARAM.End, self.main.PARAM.End + timedelta(days=n_days))
        self.data = Containers.Data(history, "close")
        print("Done")
    
    def get_all_pairs(self):
        '''
        Returns the instrument pairs that were used for fitting the copulas model
        '''
        return self.main.copulas.keys()
    
    def _calculate_marginals(self, pair):
        '''
        Calculates the marginals using a specific Copulas model for the specified 
        pair of instruments. Marginals stored as member var. 
        '''
        sym1, sym2 = pair
        # calculate marginals series
        u = self.main.copulas[pair].price_to_marginal(price=self.data.returns[sym1], symbol=sym1)
        v = self.main.copulas[pair].price_to_marginal(price=self.data.returns[sym2], symbol=sym1)
        
        self.marginals = pd.DataFrame({"u":u, "v":v}, index=self.data.returns.index)
        
    def _find_trading_points(self, pair):
        '''
        Finds the points of entry for a trade, based on looking for price movements that exceed
        the MI threshold. 
        '''
        grid_width = 0.01
        mi_calculator = self.main.copulas[pair].mispricing_index
        mi_func = lambda x: mi_calculator(x["u"], x["v"], grid_width)
        
        t = time.time()
        self.marginals[["MI_u", "MI_v"]] = pd.DataFrame(self.marginals.apply(mi_func, axis=1).tolist(), index=self.data.index[1:])
        elapsed = time.time() - t
        print(f"elapsed:{elapsed}s")
        
        # generate trading signals
        self.marginals[["trade_u", "trade_v"]] = self.marginals[["MI_u", "MI_v"]].applymap(long_or_short)
        # don't trade if only one leg has fired
        self.marginals.loc[np.abs(self.marginals["trade_u"]) + np.abs(self.marginals["trade_v"])==1] = 0 
    
    def get_backtest_timeseries(self, pair):
        
        # update marginals dataset
        self._calculate_marginals(pair)
        self._find_trading_points(pair)
        
        # join marginals dataset to original to get full dataset
        prices = self.data._timeseries[list(pair)]
        prices.join(self.marginals)
        
        self.timeseries = prices
        
        return self.timeseries