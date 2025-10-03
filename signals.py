import ta.momentum, ta.trend, ta.volatility
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

def macd_signals(data: pd.DataFrame, fast: int, slow: int, signal: int):
    """MACD crossover signals.
    Buy cuando MACD cruza por arriba de la Signal; sell cuando cruza por abajo.
    Devuelve (buy_signals, sell_signals) alineados con `data`.
    """
    # Garantiza relación válida
    if slow <= fast:
        slow = fast + 1

    macd_ind = ta.trend.MACD(close=data.Close, window_fast=fast, window_slow=slow, window_sign=signal)
    data['macd'] = macd_ind.macd()
    data['macd_signal'] = macd_ind.macd_signal()

    macd = data['macd']
    macd_sig = data['macd_signal']

    prev_macd = macd.shift(1)
    prev_sig = macd_sig.shift(1)

    buy_cross = (prev_macd <= prev_sig) & (macd > macd_sig)   # cruce alcista
    sell_cross = (prev_macd >= prev_sig) & (macd < macd_sig)  # cruce bajista
    return buy_cross.fillna(False), sell_cross.fillna(False)

def bbands_signals(data: pd.DataFrame, window: int, n_std: float):
    bb = ta.volatility.BollingerBands(data.Close, window=window, window_dev=n_std)
    lower, upper = bb.bollinger_lband(), bb.bollinger_hband()

    buy = data['Close'] < lower
    sell = data['Close'] > upper
    return buy.fillna(False), sell.fillna(False)