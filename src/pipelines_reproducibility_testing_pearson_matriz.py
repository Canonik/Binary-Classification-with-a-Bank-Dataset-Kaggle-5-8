import joblib
import numpy as np
import random
np.random.seed(42)
random.seed(42)
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from pipelines import LogTransformer, DayOfYearTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from category_encoders import TargetEncoder
from sklearn.pipeline import FeatureUnion
from lightgbm import LGBMClassifier
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_loading import standard_training_set, standard_test_set, full_original_database

num_att = ["age","pdays", "previous", "balance", "campaign", "duration"]
cat_high_card = ["job", "education"]
cat_low_card = ["marital", "default", "housing", "loan", "contact", "poutcome"]
log_att =["balance", "campaign", "duration"] 
time_att =["month", "day"]
cat_pipeline_high = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    TargetEncoder()  # OOF in practice
)

cat_pipeline_low = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore")
)

# Feature crossing example
def cross_features(X):
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=["duration", "campaign", "balance", "age", "pdays", "previous"])

    X = X.copy()
    X["duration_campaign_ratio"] = X["duration"] / (X["campaign"] + 1)
    X["balance_age_ratio"] = X["balance"] / (X["age"] + 1)
    X["pdays_previous_ratio"] = X["pdays"] / (X["previous"] + 1)
    return X


cross_pipeline = FunctionTransformer(cross_features, validate=False)

clients = standard_training_set().reset_index(drop=True)
clients_attr = clients.drop("y", axis=1)
clients_labels = clients["y"]

clients_test = standard_test_set()

print("data retrieved")

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))

num_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    PolynomialFeatures(degree=6, interaction_only=True, include_bias=False)
)
log_pipeline = make_pipeline(
    LogTransformer(),
    StandardScaler())
day_pipeline = make_pipeline(
    DayOfYearTransformer(),
    StandardScaler())
full_preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_att),
    ("cat_high", cat_pipeline_high, cat_high_card),
    ("cat_low", cat_pipeline_low, cat_low_card),
    ("log", log_pipeline, log_att),
    ("day", day_pipeline, time_att),
    ("cross", cross_pipeline, ["duration", "campaign", "balance", "age", "pdays", "previous"])
], remainder="drop")

print("pipeline created")

full_pipeline = Pipeline([
    ("full_pipeline", full_preprocessor),
    ("model", LGBMClassifier(
    boosting_type="gbdt",
    n_estimators=40000,          
    learning_rate=0.002,          
    num_leaves=8192,             
    max_depth=-1,                
    min_data_in_leaf=1,            
    feature_fraction=0.95,       
    bagging_fraction=0.95,
    bagging_freq=1,
    lambda_l1=0.2,         
    lambda_l2=0.2,
    max_bin=2048,                  
    min_gain_to_split=0.0,        
    extra_trees=True,              
    n_jobs=-1,                    
    random_state=42,
    verbose=-1
))

    ])

if __name__ == "__main__":

    full_pipeline.fit(clients_attr, clients_labels)

    print("model fitted")
    

    preds = full_pipeline.predict_proba(clients_test)[:, 1]
    pd.DataFrame(preds, columns=["y"], index=clients_test.index).to_csv("reports/GBM_maxxing0.csv")