import joblib
import numpy as np
import random
np.random.seed(42)
random.seed(42)
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
from pipelines import day_month_encoding, log_encoding, month_sin_encoding, month_cos_encoding, LogTransformer, DayOfYearTransformer, RF_useless_features, WrapperWithY

num_att = ["age", "pdays", "previous"]
cat_att =["job", "marital", "education", "default", "housing", "loan", "contact", "poutcome"]
log_att =["balance", "duration", "campaign"]
time_att = ["day", "month"]
useless_att = ["previous", "age","balance","duration","campaign"]
cyclical_att = ["month_sin", "month_cos", "day_sin", "day_cos"]

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

clients_attr["month_sin"] = clients_attr.apply(month_sin_encoding, axis=1)
clients_attr["month_cos"] = clients_attr.apply(month_cos_encoding, axis=1)
clients_attr["day_sin"] = clients_attr["day_of_the_year"].apply(lambda x: np.sin(2*np.pi *x/366 ))    
clients_attr["day_cos"] = clients_attr["day_of_the_year"].apply(lambda x: np.cos(2*np.pi *x/366 ))
clients_test["month_sin"] = clients_test.apply(month_sin_encoding, axis=1)
clients_test["month_cos"] = clients_test.apply(month_cos_encoding, axis=1)
clients_test["day_sin"] = clients_test["day_of_the_year"].apply(lambda x: np.sin(2*np.pi *x/366 ))
clients_test["day_cos"] = clients_test["day_of_the_year"].apply(lambda x: np.cos(2*np.pi *x/366 ))

num_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler())

log_pipeline = make_pipeline(
    LogTransformer(),
    StandardScaler())

day_pipeline = make_pipeline(
    DayOfYearTransformer(),
    StandardScaler()
)

useless_oof_pipeline = make_pipeline(
    WrapperWithY(RF_useless_features(features=useless_att)),
    StandardScaler()
)

cyclical_oof_pipeline = make_pipeline(
    WrapperWithY(RF_useless_features(features=cyclical_att)),
    StandardScaler()
)

preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_att),
    ("log", log_pipeline, log_att),
    ("cat", "passthrough", cat_att)
], remainder="drop")

preprocessing_doy = ColumnTransformer([
    ("day", day_pipeline, time_att)
])

full_feature_pipeline = FeatureUnion([
    ("preprocessing", preprocessing),
    ("preprocessing_doy", preprocessing_doy),
    ("rf_oof", useless_oof_pipeline),
    ("cyclical_rf_oof", cyclical_oof_pipeline)
])

full_feature_pipeline.fit(clients_attr, clients_labels) 
X_train_transformed = full_feature_pipeline.transform(clients_attr)
X_test_transformed = full_feature_pipeline.transform(clients_test)

cat_feature_indices = list(range(len(num_att) + len(log_att), len(num_att) + len(log_att) + len(cat_att)))


def Total_dataset_training():
    cat_model = CatBoostClassifier(
        iterations=2500,
        learning_rate=0.09706494663217939,
        depth=6,
        l2_leaf_reg=.385284379693478,
        eval_metric="AUC",
        cat_features=cat_feature_indices,
        random_seed=42,
        verbose=100,
        bagging_temperature=0.6393738721699852,
        border_count=128,
    )

    cat_model.fit(X_train_transformed, clients_labels)
    joblib.dump(cat_model, "models/Catboost_cyclical.pkl")

    predictions = cat_model.predict_proba(X_test_transformed)[:, 1]
    predictions_df = pd.DataFrame({
        "id": clients_test.index,
        "y": predictions
    })
    predictions_df.to_csv("reports/Catboost_cyclical.csv", index=False)


if __name__ == "__main__":
    Total_dataset_training()