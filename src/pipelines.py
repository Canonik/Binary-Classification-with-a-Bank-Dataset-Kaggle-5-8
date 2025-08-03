import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from pathlib import Path
import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def day_month_encoding(id):
    month_int= {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    month = month_int[id["month"]]
    day = id["day"]
    try:
        return datetime.date(2024, month, day).timetuple().tm_yday
    except ValueError:
        return datetime.date(2024, month, min(day, 29)).timetuple().tm_yday
    
def day_month_encoding_pipeline(df):
    month_map = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }

    df = df.copy()
    df["month_int"] = df["month"].map(month_map)
    result = []

    for day, month in zip(df["day"], df["month_int"]):
        try:
            doy = datetime.date(2024, month, day).timetuple().tm_yday
        except ValueError:
            doy = datetime.date(2024, month, min(day, 28)).timetuple().tm_yday
        result.append(doy)

    return np.array(result).reshape(-1, 1)


def month_sin_encoding(id):
    month_int = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    month = month_int[id["month"]]
    return np.sin(2 * np.pi * month / 12)

def month_cos_encoding(id):
    month_int = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    month = month_int[id["month"]]
    return np.cos(2 * np.pi * month / 12)


def log_encoding(dataframe):
    df = dataframe.copy()
    return df.apply(lambda x : np.sign(x) * np.log1p(abs(x))).to_numpy()
 
 
class LogTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        self.input_features_ = X.columns if hasattr(X, "columns") else None
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        return log_encoding(X)
    
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            if self.input_features_ is not None:
                input_features = self.input_features_
            else:
                input_features = [f"log_{i}" for i in range(self.n_features_in_)]
        return np.array([f"log_{name}" for name in input_features])
        
class DayOfYearTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        self.input_features_ = X.columns if hasattr(X, "columns") else None
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        return day_month_encoding_pipeline(X)
    
    def get_feature_names_out(self, input_features=None):
        return np.array(["day_of_the_year"])


class KNN_useless_features(BaseEstimator, TransformerMixin):
    def __init__(self, features, n_neighbors=5, n_splits=5, random_state=42):
        self.features = features 
        self.n_neighbors = n_neighbors
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(self, X, y):
        X_weak = X[self.features].copy()

        self.knn_models = []
        self.oof_predictions = np.zeros(len(X_weak.index))
        self.imputer = SimpleImputer(strategy="mean")
        X_weak = self.imputer.fit_transform(X_weak)
        
        skf = KFold(n_splits = self.n_splits, random_state=self.random_state, shuffle=True)
        
        for train_index, test_index in skf.split(X_weak, y):
            X_weak_train = X_weak[train_index]
            Y_train = y[train_index]

            X_weak_test = X_weak[test_index]

            k_ngb = KNeighborsRegressor(n_neighbors=self.n_neighbors, weights="distance")
            k_ngb = k_ngb.fit(X_weak_train, Y_train)
            self.oof_predictions[test_index] = k_ngb.predict(X_weak_test)
            self.knn_models.append(k_ngb)

        return self
    
    def transform(self, X):
        X_wk = X[self.features].copy()
        X_wk = self.imputer.transform(X_wk)

        all_pred = np.zeros(shape=(X_wk.shape[0], self.n_splits))

        for i, model in enumerate(self.knn_models):
            all_pred[:, i] = model.predict(X_wk)

        return np.mean(all_pred, axis=1).reshape(X_wk.shape[0],1)
    
    def get_feature_names_out(self):
        return [f"knn_oof_mean__{'_'.join(self.features)}"]