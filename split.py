import pandas as pd

def split_dfs(data, train:int, test:int, validation:int):

    assert train + test + validation == 100, "La suma de train, test y validation debe ser 100 exacto."

    data = data.sort_values("timestamp")

    n = len(data)
    train_corte = int(n * train / 100)
    test_corte = train_corte + int(n * test / 100)

    train_df = data.iloc[:train_corte]
    test_df = data.iloc[train_corte:test_corte]
    validation_df = data.iloc[test_corte:]

    return train_df, test_df, validation_df