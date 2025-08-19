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
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, RandomizedSearchCV
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.feature_selection import SelectFromModel
from catboost import CatBoostClassifier
from scipy.stats import randint, uniform
import datetime

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_loading import standard_training_set, standard_test_set, full_original_database
from pipelines import day_month_encoding, log_encoding, month_sin_encoding, month_cos_encoding, LogTransformer, DayOfYearTransformer, RF_useless_features, WrapperWithY, OOFJobTransformer

num_att = ["pdays", "previous", "duration"] 
cat_att =["housing", "loan", "contact", "poutcome"] 
log_att =["balance", "duration"] 


clients = standard_training_set().reset_index(drop=True)
clients_attr = clients.drop("y", axis=1)
clients_labels = clients["y"]

full_data = full_original_database().reset_index(drop=True)
full_data_clients_attr = full_data.drop("y", axis=1)
full_data_clients_labels = full_data["y"]

clients_attr = pd.concat([clients_attr, full_data_clients_attr], ignore_index=True)
clients_labels = pd.concat(
    [clients_labels.reset_index(drop=True), full_data_clients_labels.reset_index(drop=True)],
    ignore_index=True
)
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
job_pipeline = make_pipeline(
    OOFJobTransformer(columns=cat_att)
)

preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_att),
    ("log", log_pipeline, log_att),
    ("cat", cat_pipeline, cat_att),
], remainder="drop")

oof_preprocessing = ColumnTransformer([
    ("oof", job_pipeline, cat_att)
], remainder="drop")

full_preprocessing = FeatureUnion([
    ("oof_preprocessing", oof_preprocessing),
    ("preprocessing", preprocessing)
])
lgb_params = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'metric': 'auc',
    'learning_rate': 0.03583,
    'num_leaves': 350,
    'max_depth': -1,
    'min_data_in_leaf': 5,
    'feature_fraction': 0.85,
    'bagging_fraction': 0.85,
    'bagging_freq': 1,
    'lambda_l1': 0.2,
    'lambda_l2': 0.2,
    'max_bin': 511,
    'cat_smooth': 10,
    'cat_l2': 5,
    'n_jobs': -1,
    'random_state': 42,
    'verbose': 1
}

full_pipeline = Pipeline([
    ("full_preprocessing", full_preprocessing),
    ("model", lgb.LGBMClassifier(**lgb_params, n_estimators=40000))
])

if __name__ == "__main__":
   
    #cross eval auc with 5 folds CV AUC: 0.96768
    '''
    scores = cross_val_score(
        full_pipeline, clients_attr, clients_labels, 
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1, 
        verbose=2,
        scoring="roc_auc")
    
    print(f"CV AUC: {scores.mean():.5f}")
    
    '''
    #full pipeline on full training set
    GBM_model = full_pipeline.fit(clients_attr, clients_labels)
    
    joblib.dump(GBM_model, "models/GBM_pipeline.pkl")
    
    predictions = GBM_model.predict_proba(clients_test)
    predictions_df = pd.DataFrame( predictions[ :, 1], columns=["y"], index= clients_test.index)

    predictions_df.to_csv("reports/GBM_best_oof_cat_plus_cat.csv")

    '''
    # cv5 cross training for stacking
    predictions_df = pd.DataFrame(np.zeros((len(clients_attr), 1)), columns=["y"], index=clients_attr.index)

    for train_idx, val_idx in StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(clients_attr, clients_labels):
        X_train_fold = clients_attr.iloc[train_idx]
        X_val_fold = clients_attr.iloc[val_idx]
        y_train_fold = clients_labels.iloc[train_idx]
        y_val_fold = clients_labels.iloc[val_idx]


        GBM_model = full_pipeline.fit(X_train_fold, y_train_fold)
        predictions_df.iloc[val_idx, 0] = GBM_model.predict_proba(X_val_fold)[:, 1]

    predictions_df.to_csv("reports/GBMh_pipeline_cv5.csv")
    '''