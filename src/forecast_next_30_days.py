from src.model import load_model
from src.preprocessing import load_data, clean_data, feature_engineering
from src.forecast import forecast
from src.utils import save_to_csv_and_json, log, calc_metrics
import pandas as pd

def make_future_dataframe(df, periods=30):
    last_date = df['Дата'].max()
    unique_combinations = df[['Категория_товара', 'Товар', 'Город', 'Группа_клиентов', 'Формат_точки']].drop_duplicates()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods)
    future_df = unique_combinations.assign(key=1).merge(
        pd.DataFrame({'Дата': future_dates, 'key': 1}), on='key'
    ).drop('key', axis=1)
    return future_df

def forecast_next_30_days():
    log("Прогноз на следующие 30 дней...")
    df = load_data('data/raw/sales_data.xlsx')
    df = clean_data(df)
    future_df = make_future_dataframe(df, periods=30)
    future_metadata = future_df[['Дата', 'Категория_товара', 'Товар', 'Город', 'Группа_клиентов', 'Формат_точки']].copy()
    future_df['Продажи_кг'] = 0
    full_df = pd.concat([df, future_df], ignore_index=True)
    full_df = feature_engineering(full_df)
    mask_future = full_df['Дата'] > df['Дата'].max()
    features = [col for col in full_df.columns if col not in ['Продажи_кг', 'Дата']]
    X_future = full_df.loc[mask_future, features]
    preds = forecast(load_model(), X_future)
    result = future_metadata.copy()
    result['prediction'] = preds
    save_to_csv_and_json(result, 'data/processed/sales_forecast_next_30_days')
    log("Прогноз на 30 дней сохранён в data/processed/sales_forecast_next_30_days.csv и .json.")
    return result
