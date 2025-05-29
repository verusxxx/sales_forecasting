import pandas as pd
import numpy as np
import datetime

def log(msg):
    print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}")

def save_to_csv_and_json(df, path_base):
    csv_path = path_base + ".csv"
    json_path = path_base + ".json"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    df.to_json(json_path, orient='records', force_ascii=False, date_format='iso', indent=2)

def calc_metrics(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mask = y_true != 0
    if mask.sum() > 0:
        mape = (np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])).mean() * 100
    else:
        mape = np.nan
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
