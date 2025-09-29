# Entrypoint
import optuna
import pandas as pd

from backtest import backtest
from utils import triangle
from models import Operation
from split import split_dfs



def main():

    data = pd.read_csv("Binance_BTCUSDT_1h.csv") ### CARGA DE DATOS
    train_df, test_df, validation_df = split_dfs(data=pd.read_csv("Binance_BTCUSDT_1h.csv"),
                                                 train=60, test=20, validation=20)
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: backtest(trial=trial, data=train_df), n_trials=50)
    print("Best Parameters:")
    print(study.best_params)
    print("Best Value:")
    print(study.best_value)


if __name__ == "__main__":
    main()








