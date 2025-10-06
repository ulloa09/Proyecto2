import pandas as pd
import matplotlib.pyplot as plt

def compare_btc_vs_portfolio(curve_train, curve_test, curve_validation, data_path="Binance_BTCUSDT_1h.csv"):
    """
    Compara el rendimiento del portafolio (train, test, validation) contra una estrategia Buy & Hold de BTC.
    Puede llamarse directamente desde main.py o prueba_bestparams.py sin modificar su estructura.

    Parámetros
    ----------
    curve_train, curve_test, curve_validation : pd.Series
        Curvas del valor del portafolio obtenidas en cada fase.
    data_path : str
        Ruta del dataset con precios históricos de BTC (por defecto "Binance_BTCUSDT_1h.csv").
    """

    # --- Carga de precios de BTC ---
    data = pd.read_csv(data_path).sort_values("timestamp").reset_index(drop=True)
    btc_prices = data["Close"]

    # --- Concatenar curvas del portafolio ---
    curve_train = curve_train.reset_index(drop=True)
    curve_test = curve_test.reset_index(drop=True)
    curve_validation = curve_validation.reset_index(drop=True)
    curve_test.index = curve_test.index + len(curve_train)
    curve_validation.index = curve_validation.index + len(curve_train) + len(curve_test)
    curve_total = pd.concat([curve_train, curve_test, curve_validation]).reset_index(drop=True)

    # --- Buy & Hold BTC ---
    btc_hold = btc_prices / btc_prices.iloc[0]
    btc_hold = btc_hold.iloc[:len(curve_total)]

    # --- Gráfica comparativa ---
    plt.figure(figsize=(12, 6))
    plt.plot(curve_total.index, curve_total.values / curve_total.iloc[0],
             label="Portfolio (Optimized)", color="orange", linewidth=2)
    plt.plot(btc_hold.index, btc_hold.values,
             label="Buy & Hold BTC", color="gray", linestyle="--", linewidth=2)
    plt.title("Comparación: Estrategia Óptima vs. Comprar y Mantener BTC (6 años)")
    plt.xlabel("Tiempo (horas)")
    plt.ylabel("Rendimiento normalizado")
    plt.legend()
    plt.grid(True)
    plt.show()