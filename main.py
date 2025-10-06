#######################################################################
# --- Entrypoint y configuración de módulos principales ---
# Este bloque importa las bibliotecas y módulos necesarios para el
# funcionamiento del script principal, incluyendo manejo de datos,
# visualización, optimización y utilidades propias del proyecto.
#######################################################################
import optuna
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from backtest import backtest
from comparacion import compare_btc_vs_portfolio
from results import show_results
from split import split_dfs
from walk_forward_objective import walk_forward_objective


#######################################################################
# --- Función principal ---
# Esta función ejecuta el flujo completo del proceso:
# 1. Carga y preprocesamiento de los datos históricos.
# 2. División de los datos en conjuntos de entrenamiento, prueba y validación.
# 3. Optimización de hiperparámetros mediante Optuna con validación walk-forward.
# 4. Evaluación de la estrategia óptima en los tres conjuntos de datos.
# 5. Visualización de resultados y gráficas de evolución del portafolio.
#######################################################################
def main():

    # --- Definición del número de iteraciones para la optimización ---
    n = 500

    # --- Carga y preprocesamiento de datos históricos ---
    data = pd.read_csv("Binance_BTCUSDT_1h.csv").dropna()
    data = data.sort_values("timestamp").reset_index(drop=True)

    # --- División de los datos en conjuntos de entrenamiento, prueba y validación ---
    train_df, test_df, validation_df = split_dfs(data=data,
                                                 train=60, test=20, validation=20)

    # --- Configuración y ejecución de la optimización con Optuna ---
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

    # --- Ejecución de backtest con los mejores parámetros en cada conjunto ---
    # TRAIN
    metric_train, curve_train, results_train = backtest(trial=None, data=train_df, params=best_parameters)
    # TEST
    metric_test, curve_test, results_test = backtest(trial=None, data=test_df, params=best_parameters)
    # VALIDATION
    metric_validation, curve_validation, results_validation = backtest(trial=None, data=validation_df, params=best_parameters)

    # --- Visualización de resultados generales ---
    show_results(curve_train, curve_test, curve_validation, train_df, test_df, validation_df)

    # --- Reindexación de curvas y visualización gráfica de la evolución del portafolio ---
    curve_train = curve_train.reset_index(drop=True)
    curve_test = curve_test.reset_index(drop=True)
    curve_validation = curve_validation.reset_index(drop=True)
    curve_test.index = curve_test.index + len(curve_train)
    curve_validation.index = curve_validation.index + len(curve_train) + len(curve_test)

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

    # --- Impresión de los primeros resultados de cada conjunto ---
    print(results_train.head())
    print(results_test.head())
    print(results_validation.head())

    compare_btc_vs_portfolio(curve_train, curve_test, curve_validation, data_path='Binance_BTCUSDT_1h.csv')

#######################################################################
# --- Ejecución del script principal ---
# Este bloque asegura que la función main() se ejecute únicamente si el
# script es ejecutado directamente y no importado como módulo.
#######################################################################
if __name__ == "__main__":
    main()





