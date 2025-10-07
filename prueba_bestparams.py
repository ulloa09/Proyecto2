import pandas as pd
import matplotlib.pyplot as plt

from backtest import backtest
from comparacion import compare_btc_vs_portfolio
from results import show_results
from split import split_dfs

# --- Función principal para re-ejecutar el backtest con los mejores parámetros obtenidos ---
def best():
    # --- Carga de datos ---
    data = pd.read_csv("Binance_BTCUSDT_1h.csv").dropna()  ### CARGA DE DATOS
    data = data.sort_values("timestamp").reset_index(drop=True)

    # --- División del dataset en conjuntos de entrenamiento, prueba y validación ---
    train_df, test_df, validation_df = split_dfs(data=pd.read_csv("Binance_BTCUSDT_1h.csv"),
                                                 train=60, test=20, validation=20)
    # --- Definición de los mejores parámetros obtenidos en la optimización ---
    best_parameters = {'stop_loss': 0.045762288469242886, 'take_profit': 0.14755127286023728, 'rsi_window': 12, 'rsi_lower': 29,
     'rsi_upper': 75, 'macd_fast': 8, 'macd_slow': 40, 'macd_signal': 17, 'bb_window': 36, 'bb_std': 3,
     'obv_window': 38, 'atr_window': 10, 'atr_mult': 1.0527979122714386, 'adx_window': 22, 'adx_tresh': 22,
     'n_shares': 4.768467501024193}
    # {'stop_loss': 0.04998883807787347, 'take_profit': 0.12285313073332124, 'rsi_window': 29, 'rsi_lower': 26, 'rsi_upper': 74, 'macd_fast': 11, 'macd_slow': 25, 'macd_signal': 11, 'bb_window': 20, 'bb_std': 3, 'obv_window': 20, 'atr_window': 12, 'atr_mult': 2.1481660625542567, 'adx_window': 29, 'adx_tresh': 29, 'n_shares': 4.842593653355739}
    # --- Ejecución del backtest con los mejores parámetros en el conjunto de entrenamiento ---
    metric_train, curve_train, results_train = backtest(trial=None, data=train_df, params=best_parameters)

    # --- Ejecución del backtest con los mejores parámetros en el conjunto de prueba ---
    metric_test, curve_test, results_test = backtest(trial=None, data=test_df, params=best_parameters)

    # --- Ejecución del backtest con los mejores parámetros en el conjunto de validación ---
    metric_validation, curve_validation, results_validation = backtest(trial=None, data=validation_df,
                                                                       params=best_parameters)
    # --- Visualización de resultados generales ---
    show_results(train_df, test_df, validation_df, params = best_parameters)

    # --- Graficación de la evolución del portafolio para cada conjunto ---
    # Reindexar cada curva para que inicien donde terminó la anterior
    curve_train = curve_train.reset_index(drop=True)
    curve_test = curve_test.reset_index(drop=True)
    curve_validation = curve_validation.reset_index(drop=True)
    curve_test.index = curve_test.index + len(curve_train)
    curve_validation.index = curve_validation.index + len(curve_train) + len(curve_test)

    # Graficar evolución del portafolio
    plt.figure(figsize=(12, 6))
    plt.plot(curve_train.index, curve_train.values, label="Train", color='red', linewidth=2)
    plt.plot(curve_test.index, curve_test.values, label="Test", color='blue', linewidth=2)
    plt.plot(curve_validation.index, curve_validation.values, label="Validation", color='green', linewidth=2)
    plt.title("Evolución del Portafolio (Mejor estrategia)")
    plt.xlabel("Fecha")
    plt.ylabel("Valor del Portafolio")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Presentación de los primeros registros de los resultados obtenidos en cada conjunto ---
    print(results_train.head())
    print(results_test.head())
    print(results_validation.head())

    compare_btc_vs_portfolio(curve_train, curve_test, curve_validation, data_path='Binance_BTCUSDT_1h.csv')


# --- Ejecución del script ---
if __name__ == "__main__":
    best()