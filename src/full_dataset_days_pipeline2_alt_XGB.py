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
import datetime

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from Top_models.data_loading import standard_training_set, standard_test_set, full_original_database
from pipelines import day_month_encoding, month_sin_encoding, month_cos_encoding, log_encoding, LogTransformer


num_att = ["age","day_sin", "day_cos", "month_sin", "month_cos", "pdays", "previous", "day_sin", "day_cos"]
cat_att =["job", "marital", "education", "default", "housing", "loan", "contact", "poutcome"]
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

# we have created 3 dataasets, training attributes and labels, and test attributes
# now it's time for feature engineering (additional columns are added outside the transformation pipeline for simpplicity)
clients_attr["day_of_the_year"] = clients_attr.apply(day_month_encoding, axis=1)   #calculates which day of the year it is (so correlates month and day)
clients_test["day_of_the_year"] = clients_test.apply(day_month_encoding, axis=1)

clients_attr["month_sin"] = clients_attr.apply(month_sin_encoding, axis=1)
clients_attr["month_cos"] = clients_attr.apply(month_cos_encoding, axis=1)
clients_attr["day_sin"] = clients_attr["day_of_the_year"].apply(lambda x: np.sin(2*np.pi *x/366 ))    #express months and days as cyclical objects
clients_attr["day_cos"] = clients_attr["day_of_the_year"].apply(lambda x: np.cos(2*np.pi *x/366 ))
clients_test["month_sin"] = clients_test.apply(month_sin_encoding, axis=1)
clients_test["month_cos"] = clients_test.apply(month_cos_encoding, axis=1)
clients_test["day_sin"] = clients_test["day_of_the_year"].apply(lambda x: np.sin(2*np.pi *x/366 ))
clients_test["day_cos"] = clients_test["day_of_the_year"].apply(lambda x: np.cos(2*np.pi *x/366 ))

clients_attr.drop(["month", "day", "day_of_the_year"], axis=1, inplace=True)
clients_test.drop(["month", "day", "day_of_the_year"], axis=1, inplace=True)

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
    XGB_model = make_pipeline(preprocessing, XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42))
    XGB_model.fit(clients_attr, clients_labels)
    joblib.dump(XGB_model, "models/XGB_pipeline4.pkl")

    predictions = XGB_model.predict_proba(clients_test)
    predictions_df = pd.DataFrame( predictions[ :, 1], columns=["y"], index= clients_test.index)

    predictions_df.to_csv("reports/XGB_pipeline4.csv")

if __name__ == "__main__":
    Total_dataset_training()