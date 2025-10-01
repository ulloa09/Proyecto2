# Función backtesting
import numpy as np
import pandas as pd
import ta

from signals import rsi_signals
from metrics import annualized_sharpe, annualized_calmar
from models import Operation, get_portfolio_value


def backtest(data, trial, params=None) -> float:
    data = data.copy()
    data['Datetime'] = pd.to_datetime(data['timestamp'], unit= 'ms')

    if trial is not None:
        # --- cuando Optuna está optimizando ---
        rsi_window = trial.suggest_int('rsi_window', 5, 50)
        rsi_lower = trial.suggest_int('rsi_lower', 5, 35)
        rsi_upper = trial.suggest_int('rsi_upper', 65, 95)
        stop_loss = trial.suggest_float('stop_loss', 0.01, 0.15)
        take_profit = trial.suggest_float('take_profit', 0.01, 0.15)
        n_shares = trial.suggest_int('n_shares', 50, 500)
    elif params is not None:
        # --- cuando re-ejecutas con best_params ---
        rsi_window = params['rsi_window']
        rsi_lower = params['rsi_lower']
        rsi_upper = params['rsi_upper']
        stop_loss = params['stop_loss']
        take_profit = params['take_profit']
        n_shares = params['n_shares']
    else:
        raise ValueError("Debes pasar un trial de Optuna o un diccionario params.")

    buy_sig, sell_sig = rsi_signals(data, rsi_window=rsi_window, rsi_lower=rsi_lower, rsi_upper=rsi_upper)

    rsi_indicator = ta.momentum.RSIIndicator(data.Close, window=rsi_window)
    data['rsi'] = rsi_indicator.rsi()

    historic = data.dropna()

    historic['buy_signal'] = buy_sig
    historic['sell_signal'] = sell_sig

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

    if params is None:
        return calmar
    else:
        return calmar, values_port
