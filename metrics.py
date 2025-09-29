import numpy as np


def annualized_sharpe(mean: float, std: float) -> float:

    annual_rets = (mean * 8760)
    annual_std = std * np.sqrt(8760)

    return annual_rets / annual_std if annual_rets > 0 else 0
