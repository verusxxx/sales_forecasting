import sys
import os
import pandas as pd
import tempfile
import streamlit as st
import matplotlib.pyplot as plt

# Добавляем src/ в путь импорта
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from model import load_model
from preprocessing import clean_data, feature_engineering
from forecast import forecast
from utils import calc_metrics
from visualization import plot_forecast_matplotlib  # ✅ заменено на верную функцию

st.set_page_config(page_title="📊 Прогноз продаж", layout="wide")
st.title("📈 Прогноз продаж (на 30 и 7 дней)")

uploaded_file = st.file_uploader("📤 Загрузите Excel-файл с историческими продажами", type=["xlsx"])

def make_future_dataframe(df, periods):
    last_date = df['Дата'].max()
    unique_combinations = df[['Категория_товара', 'Товар', 'Город', 'Группа_клиентов', 'Формат_точки']].drop_duplicates()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods)
    future_df = unique_combinations.assign(key=1).merge(
        pd.DataFrame({'Дата': future_dates, 'key': 1}), on='key'
    ).drop('key', axis=1)
    future_df['Продажи_кг'] = 0
    return future_df

def run_forecast(df, days):
    future_df = make_future_dataframe(df, periods=days)
    future_metadata = future_df.drop(columns=['Продажи_кг'])
    full_df = pd.concat([df, future_df], ignore_index=True)
    full_df = feature_engineering(full_df)

    mask_future = full_df['Дата'] > df['Дата'].max()
    features = [col for col in full_df.columns if col not in ['Продажи_кг', 'Дата']]
    X_future = full_df.loc[mask_future, features]

    model = load_model()
    preds = forecast(model, X_future)

    result = future_metadata.copy()
    result['prediction'] = preds

    # Присоединяем реальные значения если они есть
    truth_df = df[df['Дата'].isin(result['Дата'])][
        ['Дата', 'Категория_товара', 'Товар', 'Город', 'Группа_клиентов', 'Формат_точки', 'Продажи_кг']
    ]
    result = result.merge(truth_df, on=['Дата', 'Категория_товара', 'Товар', 'Город', 'Группа_клиентов', 'Формат_точки'], how='left')

    return result

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    df = pd.read_excel(file_path)

    df = df.rename(columns={
        'Категория товара': 'Категория_товара',
        'Группа клиентов': 'Группа_клиентов',
        'Формат точки': 'Формат_точки',
        'Продажи, кг': 'Продажи_кг'
    })

    st.subheader("🔍 Загруженные данные")
    st.dataframe(df.head())

    df = clean_data(df)

    # ==== Прогноз на 30 дней ====
    st.header("📆 Прогноз на следующие 30 дней")
    forecast_30 = run_forecast(df, days=30)
    st.dataframe(forecast_30)

    csv_30 = forecast_30.to_csv(index=False).encode('utf-8-sig')
    st.download_button("📥 Скачать прогноз на 30 дней", csv_30, file_name="forecast_next_30_days.csv")

    st.subheader("📈 Визуализация прогноза на 30 дней")
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_csv:
        forecast_30.to_csv(tmp_csv.name, index=False)
        tmp_csv.flush()
        plot_forecast_matplotlib(tmp_csv.name)
        st.pyplot(plt.gcf())

    if 'Продажи_кг' in forecast_30.columns and forecast_30['Продажи_кг'].notna().any():
        y_true_30 = forecast_30['Продажи_кг'].fillna(0)
        y_pred_30 = forecast_30['prediction']
        metrics_30 = calc_metrics(y_true_30, y_pred_30)

        st.markdown("#### 🧪 Метрики на 30 дней:")
        st.markdown(f"- MAE: `{metrics_30['MAE']:.2f}`")
        st.markdown(f"- RMSE: `{metrics_30['RMSE']:.2f}`")
        st.markdown(f"- MAPE: `{metrics_30['MAPE']:.2f}%`" if not pd.isna(metrics_30['MAPE']) else "- MAPE: —")

    # ==== Прогноз на 7 дней ====
    st.header("📆 Прогноз на следующие 7 дней")
    forecast_7 = run_forecast(df, days=7)
    st.dataframe(forecast_7)

    csv_7 = forecast_7.to_csv(index=False).encode('utf-8-sig')
    st.download_button("📥 Скачать прогноз на 7 дней", csv_7, file_name="forecast_next_7_days.csv")

    st.subheader("📈 Визуализация прогноза на 7 дней")
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_csv:
        forecast_7.to_csv(tmp_csv.name, index=False)
        tmp_csv.flush()
        plot_forecast_matplotlib(tmp_csv.name)
        st.pyplot(plt.gcf())

    if 'Продажи_кг' in forecast_7.columns and forecast_7['Продажи_кг'].notna().any():
        y_true_7 = forecast_7['Продажи_кг'].fillna(0)
        y_pred_7 = forecast_7['prediction']
        metrics_7 = calc_metrics(y_true_7, y_pred_7)

        st.markdown("#### 🧪 Метрики на 7 дней:")
        st.markdown(f"- MAE: `{metrics_7['MAE']:.2f}`")
        st.markdown(f"- RMSE: `{metrics_7['RMSE']:.2f}`")
        st.markdown(f"- MAPE: `{metrics_7['MAPE']:.2f}%`" if not pd.isna(metrics_7['MAPE']) else "- MAPE: —")
