import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
import datetime

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_loading import standard_training_set, standard_test_set, full_original_database
from pipelines import day_month_encoding, log_encoding, LogTransformer

num_att = ["age","day", "pdays", "previous", "day_of_the_year"]
cat_att =["job", "marital", "education", "default", "housing", "loan", "contact","month", "poutcome"]
log_att =["balance", "duration", "campaign"]

clients = standard_training_set().reset_index(drop=True)
clients_attr = clients.drop("y", axis=1)
clients_labels = clients["y"]

full_data = full_original_database().reset_index(drop=True)
full_data_clients_attr = full_data.drop("y", axis=1)
full_data_clients_labels = full_data["y"]

clients_attr = pd.concat([clients_attr, full_data_clients_attr], ignore_index=True)
clients_labels = pd.concat([clients_labels, full_data_clients_labels], ignore_index=True)
clients_test = standard_test_set()

clients_attr["day_of_the_year"] = clients_attr.apply(day_month_encoding, axis=1)
clients_test["day_of_the_year"] = clients_test.apply(day_month_encoding, axis=1)

num_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler())

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"))

log_pipeline = make_pipeline(
    LogTransformer(),
    StandardScaler())

preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_att),
    ("cat", cat_pipeline, cat_att),
    ("log", log_pipeline, log_att)
])

def K_fold_estimation():
    XGB_model = make_pipeline(preprocessing, XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42))
    scores = cross_val_score(XGB_model, clients_attr, clients_labels,
                         scoring="roc_auc", cv=5, 
                         verbose=2)

    print(scores.mean())

def Total_dataset_training():
    best_params = {
    "subsample": 0.6,
    "reg_lambda": 1,
    "reg_alpha": 0.01,
    "n_estimators": 500,
    "max_depth": 13,
    "learning_rate": 0.03,
    "gamma": 0.05,
    "colsample_bytree": 0.6
}
    XGB_model = make_pipeline(preprocessing, XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42, **best_params))
    XGB_model.fit(clients_attr, clients_labels)
    joblib.dump(XGB_model, "models/XGB_finetuned0.pkl")

    predictions = XGB_model.predict_proba(clients_test)
    predictions_df = pd.DataFrame( predictions[ :, 1], columns=["y"], index= clients_test.index)

    predictions_df.to_csv("reports/XGB_finetuned0.csv")

def Fine_tuning_pipeline():
    parameters = {
        "xgbclassifier__n_estimators": [ 300, 400, 500, 600],
        "xgbclassifier__max_depth": [7, 9, 11, 13],
        "xgbclassifier__learning_rate": [0.03, 0.05, 0.1],
        "xgbclassifier__subsample": [0.4, 0.6, 0.8],
        "xgbclassifier__colsample_bytree": [0.6, 0.8, 1.0],
        "xgbclassifier__gamma": [ 0.05 ,0.1, 0.3],
        "xgbclassifier__reg_alpha": [0, 0.01, 0.1, 1, 10, 100],
        "xgbclassifier__reg_lambda": [0.5, 1, 1.5],
    }

    XGB_model = make_pipeline(preprocessing, XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42))
    rnd_search = RandomizedSearchCV(XGB_model, parameters, n_iter=25, cv=5, scoring="roc_auc", random_state = 42, verbose=3)
    rnd_search.fit(clients_attr, clients_labels)
    print(rnd_search.best_score_)
    print(rnd_search.best_params_)

if __name__ == "__main__":
    Total_dataset_training()