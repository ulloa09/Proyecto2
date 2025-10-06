import pandas as pd

# --- Función principal para mostrar rendimientos del portafolio ---
def show_results(curve_train, curve_test, curve_validation, train_df, test_df, validation_df):
    """
    Genera e imprime los rendimientos mensuales, trimestrales y anuales del portafolio
    sin depender de cómo venga definida la columna de fechas en los splits.
    """

    # --- Detección y conversión de columnas de fecha en los DataFrames ---
    def parse_dates(df):
        """Detecta y convierte la columna de fecha o timestamp."""
        if 'timestamp' in df.columns:
            dates = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
        elif 'Date' in df.columns and 'Hour' in df.columns:
            combined = df['Date'].astype(str) + ' ' + df['Hour'].astype(str)
            dates = pd.to_datetime(combined, errors='coerce', dayfirst=True)
        elif 'Date' in df.columns:
            dates = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
        else:
            raise ValueError("No se encontró columna de fecha o timestamp válida.")
        return dates

    # --- Obtención de fechas limpias desde cada split de datos ---
    train_dates = parse_dates(train_df)
    test_dates = parse_dates(test_df)
    val_dates = parse_dates(validation_df)

    # --- Emparejamiento de longitudes entre fechas y curvas de rendimiento ---
    min_len_train = min(len(train_dates), len(curve_train))
    min_len_test = min(len(test_dates), len(curve_test))
    min_len_val = min(len(val_dates), len(curve_validation))

    df_train = pd.DataFrame({'Date': train_dates.iloc[:min_len_train], 'Value': curve_train[:min_len_train]})
    df_test = pd.DataFrame({'Date': test_dates.iloc[:min_len_test], 'Value': curve_test[:min_len_test]})
    df_val = pd.DataFrame({'Date': val_dates.iloc[:min_len_val], 'Value': curve_validation[:min_len_val]})

    # --- Concatenación de todas las curvas en un único DataFrame para análisis conjunto ---
    df_all = pd.concat([df_train, df_test, df_val], ignore_index=True)
    df_all = df_all.dropna(subset=['Date', 'Value'])
    df_all = df_all.sort_values('Date').set_index('Date')

    # Validación para asegurar que el DataFrame resultante no esté vacío
    if df_all.empty:
        print("⚠️ No se pudieron alinear las curvas con las fechas (DataFrame vacío).")
        return None, None, None

    # --- Presentación de información básica sobre el rango y cantidad de datos disponibles ---
    print(f"Fechas disponibles: {df_all.index.min()}  →  {df_all.index.max()}")
    print(f"Total de registros válidos: {len(df_all)}")

    # --- Cálculo de retornos diarios a partir de los valores concatenados ---
    df_all['Return'] = df_all['Value'].pct_change()

    # --- Cálculo de rendimientos acumulados por periodos: mensual, trimestral y anual ---
    monthly = (1 + df_all['Return']).resample('ME').prod() - 1
    quarterly = (1 + df_all['Return']).resample('QE').prod() - 1
    annual = (1 + df_all['Return']).resample('YE').prod() - 1

    # --- Impresión formateada de los resultados de rendimiento por periodo ---
    print("\n📅 Rendimientos Mensuales (%):")
    print((monthly * 100).round(4).to_frame('Return (%)'))

    print("\n📆 Rendimientos Trimestrales (%):")
    print((quarterly * 100).round(4).to_frame('Return (%)'))

    print("\n📈 Rendimientos Anuales (%):")
    print((annual * 100).round(4).to_frame('Return (%)'))

    return monthly, quarterly, annual