import numpy as np
import pandas as pd

def annualized_sharpe(mean: float, std: float) -> float:

    annual_rets = (mean * 8760)
    annual_std = std * np.sqrt(8760)

    return annual_rets / annual_std if annual_rets > 0 else 0

def maximum_drawdown(values: pd.Series) -> float:
    roll_max = values.cummax()
    max_drawdown = (roll_max - values) / roll_max
    return max_drawdown.max()

def annualized_calmar(mean, values) -> float:
    annual_rets = (mean * 8760)
    max_drawdown = maximum_drawdown(values)
    return annual_rets / max_drawdown if max_drawdown != 0 else 0

def downside_deviation(rets) -> float:
    negative_rets = rets[rets < 0]
    return ((negative_rets ** 2).mean()) ** 0.5

def annualized_sortino(mean: float, rets) -> float:
    annual_rets = (mean * 8760)
    annual_std_down = downside_deviation(rets) * np.sqrt(8760)
    return annual_rets / annual_std_down if annual_rets > 0 else 0