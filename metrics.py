import numpy as np


def annualized_sharpe(mean: float, std: float) -> float:

    annual_rets = mean * (60 / 5) * (6.5) * (252)
    annual_std = std * np.sqrt((60 / 5) * (6.5) * (252))

    return annual_rets / annual_std if annual_rets > 0 else 0
