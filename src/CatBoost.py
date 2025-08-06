import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostClassifier
from scipy.stats import randint, uniform
import datetime

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from Top_models.data_loading import standard_training_set, standard_test_set, full_original_database
from pipelines import day_month_encoding, log_encoding, LogTransformer, DayOfYearTransformer, RF_useless_features

num_att = ["age", "pdays", "previous"]
cat_att =["job", "marital", "education", "default", "housing", "loan", "contact", "poutcome"]
log_att =["balance", "duration", "campaign"]
time_att = ["day", "month"]
useless_att = ["previous", "age","balance","duration","campaign"]

clients = standard_training_set().reset_index(drop=True)
clients_attr = clients.drop("y", axis=1)
clients_labels = clients["y"]

full_data = full_original_database().reset_index(drop=True)
full_data_clients_attr = full_data.drop("y", axis=1)
full_data_clients_labels = full_data["y"]

clients_attr = pd.concat([clients_attr, full_data_clients_attr], ignore_index=True)
clients_labels = pd.concat([clients_labels, full_data_clients_labels], ignore_index=True)
clients_test = standard_test_set()

num_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler())

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore", sparse_output=False))

log_pipeline = make_pipeline(
    LogTransformer(),
    StandardScaler())

day_pipeline = make_pipeline(
    DayOfYearTransformer(),
    StandardScaler()
)

useless_oof_pipeline = make_pipeline(
    RF_useless_features(features=useless_att),
    StandardScaler()
)

preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_att),
    ("cat", cat_pipeline, cat_att),
    ("log", log_pipeline, log_att),
], remainder="drop")

preprocessing_doy = ColumnTransformer([
    ("day", day_pipeline, time_att)
])

full_feature_pipeline = FeatureUnion([
    ("preprocessing", preprocessing),
    ("preprocessing_doy", preprocessing_doy),
    ("rf_oof", useless_oof_pipeline)
])



def K_fold_estimation():

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cat_model = make_pipeline(
        full_feature_pipeline,
        CatBoostClassifier(
        iterations=2000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3,
        eval_metric="AUC",
        random_seed=42,
        verbose=0
        )
)
    
    scores = cross_val_score(cat_model, clients_attr, clients_labels,
                             cv=kf, scoring="roc_auc", verbose=2)

    print(f"CatBoost AUC scores: {scores}")
    print(f"Mean AUC      : {scores.mean():.5f}")


def Total_dataset_training():

    cat_model = make_pipeline(
        full_feature_pipeline,
        CatBoostClassifier(
        iterations=2000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3,
        eval_metric="AUC",
        random_seed=42,
        verbose=0
        )
)
    cat_model = cat_model.fit(clients_attr, clients_labels)
    joblib.dump(cat_model, "models/Catboostv0.pkl")

    predictions = cat_model.predict_proba(clients_test)[:,1]
    predictions_df = pd.DataFrame({
        "id": clients_test.index + 750000,
        "y": predictions
    })
    predictions_df.to_csv("reports/Catboost0.csv", index=False)

if __name__ == "__main__":
    Total_dataset_training()
