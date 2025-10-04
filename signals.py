import ta.momentum, ta.trend, ta.volatility
import pandas as pd
from optuna import trial


def rsi_signals(data:pd.DataFrame, rsi_window: int, rsi_lower: int, rsi_upper: int):

    data = data.copy()
    rsi_indicator = ta.momentum.RSIIndicator(data.Close, window=rsi_window)
    data['rsi'] = rsi_indicator.rsi()

    rsi = rsi_indicator.rsi()
    buy_signals = rsi < rsi_lower
    sell_signals = rsi > rsi_upper

    return buy_signals, sell_signals

def macd_signals(data: pd.DataFrame, fast: int, slow: int, signal: int):
    """MACD crossover signals.
    Buy cuando MACD cruza por arriba de la Signal; sell cuando cruza por abajo.
    Devuelve (buy_signals, sell_signals) alineados con `data`.
    """
    # Garantiza relación válida
    data = data.copy()
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

def bbands_signals(data: pd.DataFrame, window: int, n_std: int):
    bb = ta.volatility.BollingerBands(data.Close, window=window, window_dev=n_std)
    lower, upper = bb.bollinger_lband(), bb.bollinger_hband()

    buy = data['Close'] < lower
    sell = data['Close'] > upper
    return buy.fillna(False), sell.fillna(False)


def obv_signals(data: pd.DataFrame, window: int = 20):
    """
    OBV (On-Balance Volume) signals.
    Buy cuando OBV cruza por arriba de su media móvil.
    Sell cuando OBV cruza por abajo.
    """
    data = data.copy()
    # Calcular OBV acumulado
    obv = ( (data['Close'] > data['Close'].shift(1)) * data['Volume BTC']
          - (data['Close'] < data['Close'].shift(1)) * data['Volume BTC'] ).cumsum()
    data['obv'] = obv

    # Media móvil del OBV
    obv_ma = obv.rolling(window=window).mean()
    data['obv_ma'] = obv_ma

    # Señales por cruce
    prev_obv = obv.shift(1)
    prev_ma = obv_ma.shift(1)

    buy_obv = (prev_obv <= prev_ma) & (obv > obv_ma)     # cruce alcista
    sell_obv = (prev_obv >= prev_ma) & (obv < obv_ma)    # cruce bajista

    return buy_obv.fillna(False), sell_obv.fillna(False)

def adx_signals(data: pd.DataFrame, window: int , threshold: float):
    """
    ADX + DI cruces con filtro de fuerza de tendencia.
    Buy: +DI cruza arriba de -DI y ADX >= threshold.
    Sell: -DI cruza arriba de +DI y ADX >= threshold.
    """
    data = data.copy()
    adx_ind = ta.trend.ADXIndicator(
        high=data['High'], low=data['Low'], close=data['Close'], window=window
    )
    data['adx'] = adx_ind.adx()
    data['plus_di'] = adx_ind.adx_pos()
    data['minus_di'] = adx_ind.adx_neg()

    prev_plus = data['plus_di'].shift(1)
    prev_minus = data['minus_di'].shift(1)

    buy_adx = (prev_plus <= prev_minus) & (data['plus_di'] > data['minus_di']) & (data['adx'] >= threshold)
    sell_adx = (prev_plus >= prev_minus) & (data['plus_di'] < data['minus_di']) & (data['adx'] >= threshold)

    return (buy_adx.fillna(False), sell_adx.fillna(False))


def atr_breakout_signals(data: pd.DataFrame, atr_window: int, atr_mult: float):
    """
    Señal basada en ruptura de rango con filtro ATR.
    Buy: Close > (High rolling máximo - ATR * multiplicador)
    Sell: Close < (Low rolling mínimo + ATR * multiplicador)
    """
    data = data.copy()

    # Calcular ATR
    atr = ta.volatility.AverageTrueRange(
        high=data['High'], low=data['Low'], close=data['Close'], window=atr_window
    ).average_true_range()

    # Rolling high y low recientes
    rolling_high = data['High'].rolling(window=atr_window).max()
    rolling_low = data['Low'].rolling(window=atr_window).min()

    # Señales de ruptura
    buy_atr = data['Close'] > (rolling_high - atr * atr_mult)
    sell_atr = data['Close'] < (rolling_low + atr * atr_mult)

    return buy_atr.fillna(False), sell_atr.fillna(False)