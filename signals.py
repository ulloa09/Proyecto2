import ta.momentum, ta.trend, ta.volatility
import pandas as pd
from optuna import trial

# --- Propósito general del archivo ---
# Este archivo contiene funciones para generar señales de compra y venta basadas en diferentes indicadores técnicos.
# Cada función calcula un indicador específico y determina puntos de entrada y salida en el mercado según reglas definidas.
# Las señales se devuelven como series booleanas alineadas con los datos de precios de entrada.

def rsi_signals(data:pd.DataFrame, rsi_window: int, rsi_lower: int, rsi_upper: int):
    # --- Indicador: RSI (Relative Strength Index) ---
    # --- Funcionamiento ---
    # Calcula el RSI usando una ventana temporal definida (rsi_window).
    # Genera señales de compra cuando el RSI está por debajo del umbral inferior (rsi_lower),
    # indicando condiciones de sobreventa.
    # Genera señales de venta cuando el RSI está por encima del umbral superior (rsi_upper),
    # indicando condiciones de sobrecompra.
    # --- Parámetros ---
    # data: DataFrame con datos de precios, debe contener columna 'Close'.
    # rsi_window: ventana para cálculo del RSI.
    # rsi_lower: umbral inferior para señal de compra.
    # rsi_upper: umbral superior para señal de venta.
    # --- Retorno ---
    # Tupla de dos Series booleanas (buy_signals, sell_signals) alineadas con data.

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
    # --- Indicador: MACD (Moving Average Convergence Divergence) ---
    # --- Funcionamiento ---
    # Calcula las medias móviles exponenciales rápidas y lentas y la línea de señal.
    # Genera señal de compra cuando la línea MACD cruza hacia arriba la línea de señal (crossover alcista).
    # Genera señal de venta cuando la línea MACD cruza hacia abajo la línea de señal (crossover bajista).
    # --- Parámetros ---
    # data: DataFrame con datos de precios, debe contener columna 'Close'.
    # fast: ventana para media móvil rápida.
    # slow: ventana para media móvil lenta (debe ser mayor que fast).
    # signal: ventana para la línea de señal.
    # --- Retorno ---
    # Tupla de dos Series booleanas (buy_signals, sell_signals) alineadas con data.

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
    # --- Indicador: Bandas de Bollinger ---
    # --- Funcionamiento ---
    # Calcula las bandas superior e inferior basadas en la media móvil y desviaciones estándar.
    # Señal de compra cuando el precio cierra por debajo de la banda inferior (posible sobreventa).
    # Señal de venta cuando el precio cierra por encima de la banda superior (posible sobrecompra).
    # --- Parámetros ---
    # data: DataFrame con datos de precios, debe contener columna 'Close'.
    # window: tamaño de la ventana para la media móvil.
    # n_std: número de desviaciones estándar para las bandas.
    # --- Retorno ---
    # Tupla de dos Series booleanas (buy, sell) alineadas con data.

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
    # --- Indicador: OBV (On-Balance Volume) ---
    # --- Funcionamiento ---
    # Calcula el OBV acumulado basado en cambios de precio y volumen.
    # Calcula una media móvil del OBV para suavizar la señal.
    # Señal de compra cuando OBV cruza hacia arriba su media móvil (indicación de entrada de volumen positivo).
    # Señal de venta cuando OBV cruza hacia abajo su media móvil.
    # --- Parámetros ---
    # data: DataFrame con datos de precios y volumen, debe contener 'Close' y 'Volume BTC'.
    # window: tamaño de la ventana para la media móvil del OBV.
    # --- Retorno ---
    # Tupla de dos Series booleanas (buy_obv, sell_obv) alineadas con data.

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
    # --- Indicador: ADX (Average Directional Index) y DI (Directional Indicators) ---
    # --- Funcionamiento ---
    # Calcula ADX para medir fuerza de tendencia y +DI/-DI para dirección.
    # Señal de compra cuando +DI cruza hacia arriba de -DI y ADX supera el umbral (tendencia fuerte alcista).
    # Señal de venta cuando -DI cruza hacia arriba de +DI y ADX supera el umbral (tendencia fuerte bajista).
    # --- Parámetros ---
    # data: DataFrame con datos de precios, debe contener 'High', 'Low', 'Close'.
    # window: ventana para cálculo del ADX y DI.
    # threshold: valor mínimo de ADX para confirmar fuerza de tendencia.
    # --- Retorno ---
    # Tupla de dos Series booleanas (buy_adx, sell_adx) alineadas con data.

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
    # --- Indicador: ATR (Average True Range) y ruptura de rango ---
    # --- Funcionamiento ---
    # Calcula ATR para medir volatilidad.
    # Calcula máximos y mínimos recientes del precio en una ventana móvil.
    # Señal de compra cuando el precio cierra por encima del máximo reciente menos un múltiplo del ATR (ruptura al alza).
    # Señal de venta cuando el precio cierra por debajo del mínimo reciente más un múltiplo del ATR (ruptura a la baja).
    # --- Parámetros ---
    # data: DataFrame con datos de precios, debe contener 'High', 'Low', 'Close'.
    # atr_window: ventana para cálculo del ATR y máximos/mínimos móviles.
    # atr_mult: multiplicador del ATR para definir zona de ruptura.
    # --- Retorno ---
    # Tupla de dos Series booleanas (buy_atr, sell_atr) alineadas con data.

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