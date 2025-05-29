import os
import warnings

# Отключаем ворнинги про ядра
os.environ["LOKY_MAX_CPU_COUNT"] = "2"
warnings.filterwarnings("ignore", message="Could not find the number of physical cores*")

import pandas as pd
from src.preprocessing import load_data, clean_data, feature_engineering
from src.model import train_model, load_model
from src.forecast import forecast
from src.forecast_next_7_days import forecast_next_7_days
from src.forecast_next_30_days import forecast_next_30_days
from src.utils import log, save_to_csv_and_json, calc_metrics
from src.visualization import plot_forecast_matplotlib, plot_forecast_plotly

def train_test_split_by_date(df, date_col='Дата', days_test=30):
    df = df.sort_values(date_col)
    test_start = df[date_col].max() - pd.Timedelta(days=days_test-1)
    train_df = df[df[date_col] < test_start].reset_index(drop=True)
    test_df = df[df[date_col] >= test_start].reset_index(drop=True)
    return train_df, test_df

def train():
    log("Загрузка и обработка данных...")
    df = load_data('data/raw/sales_data.xlsx')
    df = clean_data(df)
    df = feature_engineering(df)
    train_df, _ = train_test_split_by_date(df, date_col='Дата', days_test=30)
    features = [col for col in train_df.columns if col not in ['Продажи_кг', 'Дата']]
    log("Обучение модели...")
    model = train_model(train_df, features, 'Продажи_кг')
    log("Модель обучена и сохранена.")

def predict():
    log("Генерация прогноза для новых данных...")
    model = load_model()
    df = load_data('data/raw/sales_data.xlsx')
    df = clean_data(df)
    df = feature_engineering(df)
    features = [col for col in df.columns if col not in ['Продажи_кг', 'Дата']]
    preds = forecast(model, df[features])
    df['prediction'] = preds
    save_to_csv_and_json(df, 'data/processed/sales_forecast')
    log("Прогноз сохранён в data/processed/sales_forecast.csv и .json.")

def evaluate():
    log("Оценка качества модели на тестовом периоде (последние 30 дней)...")
    model = load_model()
    df = load_data('data/raw/sales_data.xlsx')
    df = clean_data(df)
    df = feature_engineering(df)
    _, test_df = train_test_split_by_date(df, date_col='Дата', days_test=30)
    features = [col for col in test_df.columns if col not in ['Продажи_кг', 'Дата']]
    preds = forecast(model, test_df[features])
    metrics = calc_metrics(test_df['Продажи_кг'], preds)
    log(f"MAE: {metrics['MAE']:.2f}, RMSE: {metrics['RMSE']:.2f}, MAPE: {metrics['MAPE'] if not pd.isna(metrics['MAPE']) else '—'}%")
    test_df['prediction'] = preds
    save_to_csv_and_json(test_df, 'data/processed/sales_forecast_test_period')
    log("Тестовый прогноз сохранён в data/processed/sales_forecast_test_period.csv и .json.")

if __name__ == '__main__':
    train()
    predict()
    evaluate()
    forecast_next_7_days()
    forecast_next_30_days()

    # Визуализация 7 дней
    plot_forecast_matplotlib(
        path='data/processed/sales_forecast_next_7_days.csv',
        title='Прогноз продаж по городам (следующие 7 дней)'
    )
    plot_forecast_plotly(
        path='data/processed/sales_forecast_next_7_days.csv',
        title='Прогноз продаж по городам (следующие 7 дней)'
    )

    # Визуализация 30 дней
    plot_forecast_matplotlib(
        path='data/processed/sales_forecast_next_30_days.csv',
        title='Прогноз продаж по городам (следующие 30 дней)'
    )
    plot_forecast_plotly(
        path='data/processed/sales_forecast_next_30_days.csv',
        title='Прогноз продаж по городам (следующие 30 дней)'
    )

    print("Результат: Прогнозы и графики сохранены.")
