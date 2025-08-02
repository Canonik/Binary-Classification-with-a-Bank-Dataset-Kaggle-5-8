import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
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

    def fit(self, x=None, y=None):
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        return log_encoding(X)
        
class DayOfYearTransformer(BaseEstimator, TransformerMixin):

    def fit(self, x=None, y=None):
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        return day_month_encoding_pipeline(X)



