import pandas as pd

def show_results(curve_train, curve_test, curve_validation, train_df, test_df, validation_df):
    """
    Genera e imprime los rendimientos mensuales, trimestrales y anuales del portafolio
    sin depender de cómo venga definida la columna de fechas en los splits.
    """

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

    # Obtener fechas limpias desde cada split
    train_dates = parse_dates(train_df)
    test_dates = parse_dates(test_df)
    val_dates = parse_dates(validation_df)

    # Emparejar longitudes
    min_len_train = min(len(train_dates), len(curve_train))
    min_len_test = min(len(test_dates), len(curve_test))
    min_len_val = min(len(val_dates), len(curve_validation))

    df_train = pd.DataFrame({'Date': train_dates.iloc[:min_len_train], 'Value': curve_train[:min_len_train]})
    df_test = pd.DataFrame({'Date': test_dates.iloc[:min_len_test], 'Value': curve_test[:min_len_test]})
    df_val = pd.DataFrame({'Date': val_dates.iloc[:min_len_val], 'Value': curve_validation[:min_len_val]})

    # Concatenar todas las curvas
    df_all = pd.concat([df_train, df_test, df_val], ignore_index=True)
    df_all = df_all.dropna(subset=['Date', 'Value'])
    df_all = df_all.sort_values('Date').set_index('Date')

    if df_all.empty:
        print("⚠️ No se pudieron alinear las curvas con las fechas (DataFrame vacío).")
        return None, None, None

    # Validación de rango temporal
    print(f"Fechas disponibles: {df_all.index.min()}  →  {df_all.index.max()}")
    print(f"Total de registros válidos: {len(df_all)}")

    # Calcular retornos diarios
    df_all['Return'] = df_all['Value'].pct_change()

    # Calcular rendimientos por periodo
    monthly = (1 + df_all['Return']).resample('ME').prod() - 1
    quarterly = (1 + df_all['Return']).resample('QE').prod() - 1
    annual = (1 + df_all['Return']).resample('YE').prod() - 1

    # Mostrar resultados
    print("\n📅 Rendimientos Mensuales (%):")
    print((monthly * 100).round(4).to_frame('Return (%)'))

    print("\n📆 Rendimientos Trimestrales (%):")
    print((quarterly * 100).round(4).to_frame('Return (%)'))

    print("\n📈 Rendimientos Anuales (%):")
    print((annual * 100).round(4).to_frame('Return (%)'))

    return monthly, quarterly, annual