import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

def plot_forecast_matplotlib(path, title='Прогноз продаж'):
    df = pd.read_csv(path)
    if 'Дата' in df.columns:
        df['Дата'] = pd.to_datetime(df['Дата'])
    plt.figure(figsize=(12, 6))
    if 'Город' in df.columns:
        for city in df['Город'].unique():
            city_data = df[df['Город'] == city]
            plt.plot(city_data['Дата'], city_data['prediction'], marker='o', label=f"Город {city}")
        plt.legend()
    else:
        plt.plot(df['Дата'], df['prediction'], marker='o')
    plt.title(title)
    plt.xlabel("Дата")
    plt.ylabel("Продажи (прогноз)")
    plt.grid()
    plt.tight_layout()
    plt.show()

def plot_forecast_plotly(path, title='Прогноз продаж'):
    df = pd.read_csv(path)
    if 'Дата' in df.columns:
        df['Дата'] = pd.to_datetime(df['Дата'])
    if 'Город' in df.columns:
        fig = px.line(df, x='Дата', y='prediction', color='Город', title=title, markers=True)
    else:
        fig = px.line(df, x='Дата', y='prediction', title=title, markers=True)
    fig.show()
