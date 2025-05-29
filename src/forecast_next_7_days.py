from src.model import load_model
from src.preprocessing import load_data, clean_data, feature_engineering
from src.forecast import forecast
from src.utils import save_to_csv_and_json, log, calc_metrics
import pandas as pd

def make_future_dataframe(df, periods=7):
    last_date = df['Дата'].max()
    unique_combinations = df[['Категория_товара', 'Товар', 'Город', 'Группа_клиентов', 'Формат_точки']].drop_duplicates()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods)
    future_df = unique_combinations.assign(key=1).merge(
        pd.DataFrame({'Дата': future_dates, 'key': 1}), on='key'
    ).drop('key', axis=1)
    return future_df

def forecast_next_7_days():
    log("Прогноз на следующие 7 дней...")
    df = load_data('data/raw/sales_data.xlsx')
    df = clean_data(df)

    # Формируем будущие даты
    future_df = make_future_dataframe(df, periods=7)
    future_metadata = future_df[['Дата', 'Категория_товара', 'Товар', 'Город', 'Группа_клиентов', 'Формат_точки']].copy()

    # Сохраняем настоящие значения, если вдруг есть
    future_df_with_truth = df[df['Дата'].isin(future_df['Дата'])][
        ['Дата', 'Категория_товара', 'Товар', 'Город', 'Группа_клиентов', 'Формат_точки', 'Продажи_кг']
    ]

    # Отладка: покажем есть ли продажи на будущие 7 дней
    print("\n=== ОТЛАДКА: Данные о продажах на будущие 7 дней ===")
    print(future_df_with_truth)
    print("Количество строк с реальными продажами на будущие даты:", len(future_df_with_truth))
    print("Максимальная дата в исходных данных:", df['Дата'].max())
    print("Будущие даты прогноза:", future_df['Дата'].unique())

    # Устанавливаем фиктивные нули
    future_df['Продажи_кг'] = 0
    full_df = pd.concat([df, future_df], ignore_index=True)
    full_df = feature_engineering(full_df)

    # Выделяем будущие строки
    mask_future = full_df['Дата'] > df['Дата'].max()
    features = [col for col in full_df.columns if col not in ['Продажи_кг', 'Дата']]
    X_future = full_df.loc[mask_future, features]

    # Прогноз
    preds = forecast(load_model(), X_future)

    # Собираем результат
    result = future_metadata.copy()
    result['prediction'] = preds

    # Добавим реальные продажи (если были)
    result = result.merge(
        future_df_with_truth,
        on=['Дата', 'Категория_товара', 'Товар', 'Город', 'Группа_клиентов', 'Формат_точки'],
        how='left'
    )

    # Сохраняем
    save_to_csv_and_json(result, 'data/processed/sales_forecast_next_7_days')
    log("Прогноз на 7 дней сохранён в data/processed/sales_forecast_next_7_days.csv и .json.")

    # Вывод метрик, если есть реальные значения
    if 'Продажи_кг' in result.columns and result['Продажи_кг'].notna().any():
        y_true = result['Продажи_кг'].fillna(0)
        y_pred = result['prediction']
        metrics = calc_metrics(y_true, y_pred)

        mape_str = f"{metrics['MAPE']:.2f}%" if not pd.isna(metrics['MAPE']) else "—"
        log(f"[7-day forecast] MAE: {metrics['MAE']:.2f}, RMSE: {metrics['RMSE']:.2f}, MAPE: {mape_str}")
        print("\n=== METRICS FOR 7-DAY FORECAST ===")
        print(f"MAE:  {metrics['MAE']:.2f}")
        print(f"RMSE: {metrics['RMSE']:.2f}")
        print(f"MAPE: {mape_str}\n")

    return result
