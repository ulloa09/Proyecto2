import pandas as pd

from backtest import backtest


# --- Función principal para mostrar rendimientos del portafolio ---
def show_results(train_df, test_df, validation_df, params):

    _, value_train, _ = backtest(trial=None, data=train_df, params=params)
    _, value_test, _ = backtest(trial=None, data=test_df, params=params)
    _, value_validation, _ = backtest(trial=None, data=validation_df, params=params)

    print(value_train.shape)
    print(value_test.shape)
    print("Validation shape", validation_df.shape)
    print("Curva validation shape",value_validation.shape)

    return