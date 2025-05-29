import streamlit as st
import pandas as pd
import tempfile

from src.model import load_model
from src.preprocessing import clean_data, feature_engineering
from src.forecast import forecast
from src.utils import calc_metrics

st.set_page_config(page_title="Прогноз продаж", layout="wide")
st.title("📈 Автоматический прогноз продаж на 7 дней")

# Загрузка файла
uploaded_file = st.file_uploader("Загрузите Excel-файл с историей продаж", type=["xlsx"])

if uploaded_file:
    st.success("Файл загружен успешно. Чтение данных...")

    # Временное сохранение
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    df = pd.read_excel(file_path)

    # Переименование колонок для соответствия коду
    df = df.rename(columns={
        'Категория товара': 'Категория_товара',
        'Группа клиентов': 'Группа_клиентов',
        'Формат точки': 'Формат_точки',
        'Продажи, кг': 'Продажи_кг'
    })

    st.subheader("🔍 Превью загруженных данных")
    st.dataframe(df.head())

    df = clean_data(df)

    # Генерация будущих дат
    def make_future_dataframe(df, periods=7):
        last_date = df['Дата'].max()
        unique_combinations = df[['Категория_товара', 'Товар', 'Город', 'Группа_клиентов', 'Формат_точки']].drop_duplicates()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods)
        future_df = unique_combinations.assign(key=1).merge(
            pd.DataFrame({'Дата': future_dates, 'key': 1}), on='key'
        ).drop('key', axis=1)
        return future_df

    future_df = make_future_dataframe(df, periods=7)
    future_df['Продажи_кг'] = 0

    full_df = pd.concat([df, future_df], ignore_index=True)
    full_df = feature_engineering(full_df)

    # Отбор признаков для будущего периода
    mask_future = full_df['Дата'] > df['Дата'].max()
    features = [col for col in full_df.columns if col not in ['Продажи_кг', 'Дата']]
    X_future = full_df.loc[mask_future, features]

    # Прогноз
    model = load_model()
    preds = forecast(model, X_future)

    result = future_df.copy()
    result['prediction'] = preds

    st.subheader("📊 Прогноз на следующие 7 дней")
    st.dataframe(result)

    # Скачивание
    csv = result.to_csv(index=False).encode('utf-8-sig')
    st.download_button("📥 Скачать прогноз в CSV", data=csv, file_name="sales_forecast_next_7_days.csv")

    # Если есть настоящие продажи — сравниваем
    real = df[df['Дата'].isin(result['Дата'])][
        ['Дата', 'Категория_товара', 'Товар', 'Город', 'Группа_клиентов', 'Формат_точки', 'Продажи_кг']
    ]
    result = result.merge(real, on=['Дата', 'Категория_товара', 'Товар', 'Город', 'Группа_клиентов', 'Формат_точки'], how='left')

    if 'Продажи_кг' in result.columns and result['Продажи_кг'].notna().any():
        y_true = result['Продажи_кг'].fillna(0)
        y_pred = result['prediction']
        metrics = calc_metrics(y_true, y_pred)

        st.subheader("📏 Метрики качества прогноза")
        st.markdown(f"- **MAE**: `{metrics['MAE']:.2f}`")
        st.markdown(f"- **RMSE**: `{metrics['RMSE']:.2f}`")
        mape_str = f"{metrics['MAPE']:.2f}%`" if not pd.isna(metrics['MAPE']) else "—"
        st.markdown(f"- **MAPE**: `{mape_str}`")
