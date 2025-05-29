import pandas as pd
from sklearn.impute import SimpleImputer

def load_data(path):
    df = pd.read_excel(path, parse_dates=['Дата'])
    df.columns = [col.replace(' ', '_').replace(',', '') for col in df.columns]
    return df

def clean_data(df):
    df = df[df['Продажи_кг'] >= 0].copy()
    # Обрезка выбросов (по желанию)
    q_low = df['Продажи_кг'].quantile(0.01)
    q_hi  = df['Продажи_кг'].quantile(0.99)
    df = df[(df['Продажи_кг'] >= q_low) & (df['Продажи_кг'] <= q_hi)]
    imputer = SimpleImputer(strategy='median')
    num_cols = ['Категория_товара', 'Товар', 'Город', 'Группа_клиентов', 'Формат_точки', 'Продажи_кг']
    df.loc[:, num_cols] = imputer.fit_transform(df[num_cols])
    return df

def feature_engineering(df, allow_nan_future=False):
    df = df.sort_values(['Товар', 'Город', 'Дата'])
    # Лаги
    for lag in [1, 7, 14]:
        df[f'lag_{lag}'] = df.groupby(['Товар', 'Город'])['Продажи_кг'].shift(lag)
    # Скользящее среднее
    df['rolling_mean_7'] = df.groupby(['Товар', 'Город'])['Продажи_кг'].transform(lambda x: x.shift(1).rolling(7).mean())
    df['rolling_mean_14'] = df.groupby(['Товар', 'Город'])['Продажи_кг'].transform(lambda x: x.shift(1).rolling(14).mean())
    # Временные признаки
    df['dayofweek'] = df['Дата'].dt.dayofweek
    df['month'] = df['Дата'].dt.month
    df['year'] = df['Дата'].dt.year
    df['prod_city'] = df['Товар'].astype(str) + '_' + df['Город'].astype(str)
    df = pd.get_dummies(df, columns=['Категория_товара', 'Товар', 'Город', 'Группа_клиентов', 'Формат_точки', 'prod_city'])
    df = df.fillna(0)
    if allow_nan_future:
        return df
    else:
        df = df.dropna().reset_index(drop=True)
        return df
