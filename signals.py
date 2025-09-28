import ta.momentum
import pandas as pd
from optuna import trial


def rsi_signals(data:pd.DataFrame, rsi_window: int, rsi_lower: int, rsi_upper: int):

    rsi_indicator = ta.momentum.RSIIndicator(data.Close, window=rsi_window)
    data['rsi'] = rsi_indicator.rsi()

    rsi = rsi_indicator.rsi()
    buy_signals = rsi < rsi_lower
    sell_signals = rsi > rsi_upper

    return buy_signals, sell_signals


    # buy_sig, sell_sig = rsi_signals(data, rsi_window=rsi_window, rsi_lower=, rsi_upper=rsi_upper, rsi_ind=rsi_ind)

