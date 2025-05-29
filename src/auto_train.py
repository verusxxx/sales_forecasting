from src.preprocessing import load_data, clean_data, feature_engineering
from src.model import train_model

def run_training_pipeline(data_path):
    df = load_data(data_path)
    df = clean_data(df)
    df = feature_engineering(df)
    features = [col for col in df.columns if col not in ['Продажи,_кг', 'Дата']]
    train_model(df, features, 'Продажи,_кг')

if __name__ == '__main__':
    run_training_pipeline('data/raw/sales_data.xlsx')