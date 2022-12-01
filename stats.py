import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy import stats

def rolling_sharpe(ret):
    return np.multiply(np.divide(ret.mean(), ret.std()), np.sqrt(252))

def get_rolling_stats(result, r_window):
    result["rolling_sharpe"] = result["portfolio_value"].pct_change().rolling(r_window).apply(rolling_sharpe)
    result["rolling_vol"] = result["portfolio_value"].pct_change().rolling(r_window).std()
    w, pp, result["drawdowns"] = get_drawdowns(result)
    stats = [math.log(result.rolling_sharpe.mean())/result.rolling_sharpe.std(), result.portfolio_value.mean(), result.rolling_vol.mean()]
    return result, stats

def get_drawdowns(result):
    return_series = result.portfolio_value.pct_change()
    wealth_index = (1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return wealth_index, previous_peaks, drawdowns

def show_rolling_stats(result, r_window):
    result, stats = get_rolling_stats(result, r_window)
    fig, ax = plt.subplots(1, 3, figsize=(18,4))
    result['rolling_sharpe'].plot(ax = ax[0], title='Rolling sharpe')
    ax[0].axhline(0,color='red',ls='--')
    #result['rolling_vol'].plot(ax = ax[1], title='Rolling volatility')
    result['drawdowns'].plot(ax = ax[1], title='Drawdowns')
    result['portfolio_value'].plot(ax = ax[2], title='Portfolio value')
    print(f"Starting portfolio value: {result.portfolio_value[0]}")
    print(f"Final portfolio value: {result.portfolio_value[-1]}")
    print(f"Average sharpe: {result.rolling_sharpe.mean()}")
    print(f"Max drawdown: {result.drawdowns.min()}")
    plt.show()
    return result