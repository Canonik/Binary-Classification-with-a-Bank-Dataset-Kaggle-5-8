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
from pipelines import day_month_encoding, log_encoding, LogTransformer, DayOfYearTransformer, RF_useless_features, WrapperWithY

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
    ("rf_oof", useless_oof_pipeline)
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
    joblib.dump(cat_model, "models/Catboost_15h.pkl")

    predictions = cat_model.predict_proba(X_test_transformed)[:, 1]
    predictions_df = pd.DataFrame({
        "id": clients_test.index,
        "y": predictions
    })
    predictions_df.to_csv("reports/Catboost_15h.csv", index=False)



def catboost_fine_tuning(X, y):


    param_dist = {
        "depth": [4, 5, 6, 7, 8, 9, 10],
        "learning_rate": uniform(0.01, 0.09),  
        "l2_leaf_reg": uniform(1, 9),          
        "iterations": [1500, 2000, 2500],
        "bagging_temperature": uniform(0, 1),
        "border_count": [32, 64, 128, 254]
    }

    cat_model = CatBoostClassifier(
        eval_metric="AUC",
        random_seed=42,
        verbose=0,
        cat_features=cat_feature_indices,
        task_type="CPU"
    )

    search = RandomizedSearchCV(
        estimator=cat_model,
        param_distributions=param_dist,
        n_iter=2,  
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="roc_auc",
        verbose=2,
        n_jobs=-1  
    )

    search.fit(X, y)

    print("Best ROC AUC:", search.best_score_)
    print("Best params:", search.best_params_)

    final_model = CatBoostClassifier(
    **search.best_params_,
    eval_metric="AUC",
    cat_features=cat_feature_indices,
    random_seed=42,
    verbose=100,
    task_type="CPU"
)
    final_model.fit(X, y)
    joblib.dump(final_model, "models/CatBoost_finetuned_15h.pkl")
    pd.DataFrame([search.best_params_]).to_csv("reports/CatBoost_best_params.csv", index=False)

    return search
  
if __name__ == "__main__":
    Total_dataset_training()