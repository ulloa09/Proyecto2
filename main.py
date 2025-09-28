# Entrypoint
import optuna
import pandas as pd

from backtest import backtest
from utils import triangle
from models import Operation



def main():
    data = pd.read_csv("Binance_BTCUSDT_1h.csv") ### CARGA DE DATOS
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: backtest(trial=trial, data=data), n_trials=10)
    print("Best Parameters:")
    print(study.best_params)
    print("Best Value:")
    print(study.best_value)


if __name__ == "__main__":
    main()








