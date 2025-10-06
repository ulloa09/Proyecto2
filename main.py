# Entrypoint
import optuna
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from backtest import backtest
from results import show_results
from split import split_dfs
from walk_forward_objective import walk_forward_objective


def main():

    n = 100

    data = pd.read_csv("Binance_BTCUSDT_1h.csv").dropna() ### CARGA DE DATOS
    data = data.sort_values("timestamp").reset_index(drop=True)

    train_df, test_df, validation_df = split_dfs(data=data,
                                                 train=60, test=20, validation=20)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    pbar = tqdm(total=n, desc="Optuna optimization", ncols=80)
    for _ in range(n):
        study.optimize(lambda trial: walk_forward_objective(trial=trial, data=train_df, n_splits=3),
                       n_trials=1, catch=(Exception,), n_jobs=-1)
        pbar.update(1)
    pbar.close()
    best_parameters = study.best_params
    best_value = study.best_value
    print("Best Parameters:")
    print(best_parameters)
    print("Best Value:")
    print(best_value)
    # Re-ejecutar backtest con los mejores parámetros en:
    # TRAIN
    metric_train, curve_train, results_train = backtest(trial=None, data=train_df, params=best_parameters)

    # TEST
    metric_test, curve_test, results_test = backtest(trial=None, data=test_df, params=best_parameters)

    #VALIDATION
    metric_validation, curve_validation, results_validation = backtest(trial=None, data=validation_df, params=best_parameters)

    # Mostrar resultados
    show_results(curve_train, curve_test, curve_validation, train_df, test_df, validation_df)

    #### GRÁFICAS
    # Reindexar cada curva para que inicien donde terminó la anterior
    curve_train = curve_train.reset_index(drop=True)
    curve_test = curve_test.reset_index(drop=True)
    curve_validation = curve_validation.reset_index(drop=True)
    curve_test.index = curve_test.index + len(curve_train)
    curve_validation.index = curve_validation.index + len(curve_train) + len(curve_test)

    # Graficar evolución del portafolio
    plt.figure(figsize=(12,6))
    plt.plot(curve_train.index, curve_train.values, label="Train" ,color='red', linewidth=2)
    plt.plot(curve_test.index, curve_test.values, label="Test", color='blue', linewidth=2)
    plt.plot(curve_validation.index,curve_validation.values,  label="Validation", color='green', linewidth=2)
    plt.title("Evolución del Portafolio (Mejor estrategia)")
    plt.xlabel("Fecha")
    plt.ylabel("Valor del Portafolio")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(results_train.head())
    print(results_test.head())
    print(results_validation.head())


if __name__ == "__main__":
    main()





