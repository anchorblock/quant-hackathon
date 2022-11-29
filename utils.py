from zipline.api import order, record, symbol
from zipline import run_algorithm
from zipline.utils.run_algo import load_extensions
from zipline.data import bundles
from zipline.data.data_portal import DataPortal
from zipline.utils.calendar_utils import get_calendar
from datetime import datetime, timedelta
import warnings
import os
import pandas as pd
import numpy as np

def get_symbols(datadir):
    coin_list = []
    files = os.listdir(datadir)
    for i in range(len(files)):
        coin_list.append(files[i][:-4])
    return coin_list

load_extensions(
    default=True,
    extensions=[],
    strict=True,
    environ=os.environ,
)

class dailyBars:
    """This class contains the methods to retrieve data from the ingested bundle for the given timeframe.

    Args:
            coins (list, optional): A list of coins for which the price data will be returned. Defaults to ['BTC'].
            bundle (str, optional): Name of the bundle from where to extract data. Defaults to 'cryptocompare_daily'.
            calendar (exchange_calendars object): The exchange calendar object that the bundle follows.
    """
    def __init__(self, calendar, coins=['BTC'], bundle='cryptocompare_daily'):
        self.coins = coins
        self.bundle = bundle
        self.calendar = calendar
        self.extensions = load_extensions(
                                        default=True,
                                        extensions=[],
                                        strict=True,
                                        environ=os.environ,
                                        )
        self.bundle_data = bundles.load(self.bundle)
        self.data = DataPortal(
                        self.bundle_data.asset_finder,
                        trading_calendar=self.calendar,
                        first_trading_day=self.bundle_data.equity_daily_bar_reader.first_trading_day,
                        equity_minute_reader=None,
                        equity_daily_reader=self.bundle_data.equity_daily_bar_reader,
                        adjustment_reader=self.bundle_data.adjustment_reader,
                        )
        self.sids = self.bundle_data.asset_finder.sids
        self.assets = self.bundle_data.asset_finder.lookup_symbols(self.coins, as_of_date=None)
        self.pca = None
        self.num_pc = None
        self.features = None
        self.fct = None
        self.fct_ret = None
        self.fct_exp = None
        self.km = None
        self.corr = None
        self.cov = None
        self.eig_val = None
        self.eig_vec = None
        self.fct_rl = None
        self.zscores = None

    def ohlcvData(self, start, end, value_list=['close']):    
        """Generates a dataframe containing OHLCV (as per value_list) data of coins for a given timeframe.

        Args:
            start (str): Start date of price data. Format "%Y-%m-%d". Example '2022-04-05'.
            end (str): End date of price data; format "%Y-%m-%d", example '2022-07-20'.
            value_list (list, optional): A list of values to be returned. Example ['open', 'high', 'low', 'close', 'volume']. Defaults to ['close'].
        Returns:
            pd.DataFrame: A dataframe containing the daily OHLCV values of the sysmbols for a given timeframe.
        """
        warnings.filterwarnings("ignore")

        tmp = {}
    
        for value in value_list:
            df_value = self.data.get_history_window(self.assets,
                            end_dt = pd.Timestamp(end, tz='utc'),
                            bar_count = len(self.calendar.sessions_in_range(start,end)),
                            frequency = '1d',
                            field = value,
                            data_frequency = 'daily'
                            )
            df_value.columns = [eq.symbol for eq in df_value.columns]

            tmp[value] = pd.DataFrame(df_value, index=df_value.index)
            df = pd.concat(tmp, axis=1)
        return df

    def avgPriceData(self, start, end):
        """Calculate daily average price from OHLC data and return a dataframe containing prices of coins for a given timeframe.

        Args:
            start (str): Start date of price data. Format "%Y-%m-%d". Example '2022-04-05'.
            end (str): End date of price data; format "%Y-%m-%d", example '2022-07-20'.

        Returns:
            pd.DataFrame: A dataframe containing the daily prices of the sysmbols for a given timeframe.
        """
        warnings.filterwarnings("ignore")

        df = self.ohlcvData(start, end, value_list=['open', 'high', 'low', 'close'])
        return pd.concat([df['open'],df['high'],df['low'],df['close']]).groupby(level=0).mean()

    def pctReturn(self, start, end, periods=1):
        """Computes the percent changes for the given period.

        Args:
            start (str): Start date of price data. Format "%Y-%m-%d". Example '2022-04-05'.
            end (str): End date of price data; format "%Y-%m-%d", example '2022-07-20'.
            periods (int, optional): Number of periods to compute percent change for. Defaults to 1.

        Returns:
            pd.DataFrame: A dataframe containing the percent changes of the coins for the given periods.

        Note:
            The returned dataframe will ommit the first n=periods rows as they will have NaN values in them.
        """
        warnings.filterwarnings("ignore")
        
        price = self.data.get_history_window(self.assets,
                                        end_dt = pd.Timestamp(end, tz='utc'),
                                        bar_count = len(self.calendar.sessions_in_range(start,end))+periods,
                                        frequency = '1d',
                                        field = 'close',
                                        data_frequency = 'daily'
                                        )
        price.columns = [eq.symbol for eq in price.columns]

        self.pct_ret = price.pct_change(periods = periods)[periods:]
        return self.pct_ret