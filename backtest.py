# Función backtesting
import numpy as np
import pandas as pd
import ta

from signals import rsi_signals, macd_signals, bbands_signals
from metrics import annualized_sharpe, annualized_calmar, annualized_sortino, win_rate
from models import Operation, get_portfolio_value


def backtest(data, trial, params=None) -> float:
    data = data.copy()
    data['Datetime'] = pd.to_datetime(data['timestamp'], unit= 'ms', errors='coerce')
    data.set_index('Datetime')

    if trial is not None:
        # --- cuando Optuna optimiza ---
        rsi_window = trial.suggest_int('rsi_window', 5, 50)
        rsi_lower = trial.suggest_int('rsi_lower', 5, 35)
        rsi_upper = trial.suggest_int('rsi_upper', 65, 95)
        stop_loss = trial.suggest_float('stop_loss', 0.01, 0.15)
        take_profit = trial.suggest_float('take_profit', 0.01, 0.15)
        macd_fast = trial.suggest_int('macd_fast', 8, 20)
        macd_slow = trial.suggest_int('macd_slow', 21, 50)  # debe ser > fast
        macd_signal = trial.suggest_int('macd_signal', 5, 20)
        bb_window = trial.suggest_int('bb_window', 10, 50)
        bb_std = trial.suggest_float('bb_std', 1.5, 3.5)
        n_shares = trial.suggest_int('n_shares', 1, 30)
    elif params is not None:
        # --- cuando se usa con best_params ---
        rsi_window = params['rsi_window']
        rsi_lower = params['rsi_lower']
        rsi_upper = params['rsi_upper']
        macd_fast = params['macd_fast']
        macd_slow = params['macd_slow']
        macd_signal = params['macd_signal']
        bb_window = params['bb_window']
        bb_std = params['bb_std']
        stop_loss = params['stop_loss']
        take_profit = params['take_profit']
        n_shares = params['n_shares']
    else:
        raise ValueError("Debes pasar un trial de Optuna o un diccionario params.")

    buy_rsi, sell_rsi = rsi_signals(data, rsi_window=rsi_window, rsi_lower=rsi_lower, rsi_upper=rsi_upper)
    buy_macd, sell_macd = macd_signals(data, fast=macd_fast, slow=macd_slow, signal=macd_signal)
    buy_bbands, sell_bbands = bbands_signals(data, bb_window, bb_std)

    # Juntamos señales en un DataFrame para contar cuántas se activan
    buy_df = pd.concat([buy_rsi, buy_macd, buy_bbands], axis=1)
    sell_df = pd.concat([sell_rsi, sell_macd, sell_bbands], axis=1)

    # Condición: al menos 2 señales activas
    buy_signal = (buy_df.sum(axis=1) >= 2)
    sell_signal = (sell_df.sum(axis=1) >= 2)

    rsi_indicator = ta.momentum.RSIIndicator(data.Close, window=rsi_window)
    data['rsi'] = rsi_indicator.rsi()

    historic = data.dropna()
    historic['buy_signal'] = buy_signal
    historic['sell_signal'] = sell_signal

    COM = 0.125 / 100
    SL = stop_loss
    TP = take_profit
    BORROW_RATE = 0.25 / 100

    cash = 1_000_000

    active_long_positions: list[Operation] = []

    portfolio_value = [cash]

    for i, row in historic.iterrows():

        # Close positions
        for position in active_long_positions[:]:  # Iterate over a copy of the list
            if row.Close > position.take_profit or row.Close < position.stop_loss:
                # Close the position
                cash += row.Close * position.n_shares * (1 - COM)
                # Remove the position from active positions
                active_long_positions.remove(position)
                continue

        # Buy
        # Check signal
        if not row.buy_signal:
            portfolio_value.append(get_portfolio_value(
                cash, active_long_positions, [], row.Close, n_shares
            ))
            continue

        # Do we have enough cash?
        if cash < row.Close * n_shares * (1 + COM):
            portfolio_value.append(get_portfolio_value(
                cash, active_long_positions, [], row.Close, n_shares
            ))
            continue

        # Discount the cost
        cash -= row.Close * n_shares * (1 + COM)
        # Save the operation as active position
        active_long_positions.append(Operation(
            time=row.Datetime,
            price=row.Close,
            stop_loss=row.Close * (1 - SL),
            take_profit=row.Close * (1 + TP),
            n_shares=n_shares,
            type='LONG'
        ))

        # This only works for long positions
        portfolio_value.append(get_portfolio_value(
            cash, active_long_positions, [], row.Close, n_shares
        ))

    cash += row.Close * len(active_long_positions) * n_shares * (1 - COM)
    active_long_positions = []

    df = pd.DataFrame()
    df['value'] = portfolio_value
    df['rets'] = df.value.pct_change()
    df.dropna(inplace=True)

    mean_t = df.rets.mean()
    std_t = df.rets.std()
    values_port = df['value']
    sharpe_anual = annualized_sharpe(mean=mean_t, std=std_t)
    calmar = annualized_calmar(mean=mean_t, values=values_port)
    sortino = annualized_sortino(mean_t, df['rets'])
    wr = win_rate(df['rets'])

    results = pd.DataFrame()
    results['Portfolio'] = df['value'].tail(1)
    results['Sharpe'] = sharpe_anual
    results['Calmar'] = calmar
    results['Sortino'] = sortino
    results['Win Rate'] = wr

    if params is None:
        return calmar
    else:
        return calmar, values_port, results
