# Entrypoint
import optuna
import pandas as pd
import matplotlib.pyplot as plt

from backtest import backtest
from utils import triangle
from models import Operation
from split import split_dfs



def main():

    data = pd.read_csv("Binance_BTCUSDT_1h.csv").dropna() ### CARGA DE DATOS
    plt.plot(data['Close'], color="darkgray")
    plt.title("Close Bitcoin (24/7)")
    plt.grid()
    plt.show()

    train_df, test_df, validation_df = split_dfs(data=pd.read_csv("Binance_BTCUSDT_1h.csv"),
                                                 train=60, test=20, validation=20)
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: backtest(trial=trial, data=train_df, params=None), n_trials=50)
    best_parameters = study.best_params
    best_value = study.best_value
    print("Best Parameters:")
    print(best_parameters)
    print("Best Value:")
    print(best_value)

    # Re-ejecutar backtest con los mejores parámetros
    metric, curve = backtest(trial=None, data=train_df, params=best_parameters)

    # Graficar evolución del portafolio
    plt.figure(figsize=(12,6))
    plt.plot(curve)
    plt.title("Evolución del Portafolio (Mejor estrategia)")
    plt.xlabel("Fecha")
    plt.ylabel("Valor del Portafolio")
    #plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()








