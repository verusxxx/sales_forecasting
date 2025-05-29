import lightgbm as lgb
from sklearn.model_selection import train_test_split
import joblib

def train_model(df, features, target):
    X = df[features]
    y = df[target]
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X, y)
    joblib.dump(model, 'model.pkl')
    return model

def load_model(path='model.pkl'):
    return joblib.load(path)