import joblib
import numpy as np
import random
np.random.seed(42)
random.seed(42)
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
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

from data_loading import standard_training_set, standard_test_set, full_original_database
from pipelines import LogTransformer, DayOfYearTransformer, RF_useless_features, WrapperWithY, sinMonthTransformer, cosMonthTransformer, sinDayTransformer, cosDayTransformer, MonthDayPreprocessor

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
clients_labels = pd.concat(
    [clients_labels.reset_index(drop=True), full_data_clients_labels.reset_index(drop=True)],
    ignore_index=True
)
clients_test = standard_test_set()

num_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler())

log_pipeline = make_pipeline(
    LogTransformer(),
    StandardScaler())

day_pipeline = make_pipeline(
    DayOfYearTransformer(),
    StandardScaler())


sinMonth_pipeline = make_pipeline(
    sinMonthTransformer())

cosMonth_pipeline = make_pipeline(
    cosMonthTransformer())

sinDay_pipeline = make_pipeline(
    sinDayTransformer())

cosDay_pipeline = make_pipeline(
    cosDayTransformer())


useless_oof_pipeline = make_pipeline(
    WrapperWithY(RF_useless_features(features=useless_att)),
    StandardScaler())

cyclical_oof_pipeline = make_pipeline(
    WrapperWithY(RF_useless_features(features=cyclical_att)),
    StandardScaler())


cat_pipeline = make_pipeline(
    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
)


preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_att),
    ("log", log_pipeline, log_att),
    ("cat", "passthrough", cat_att)  # will re-add later
], remainder="passthrough")


sincos_pipeline = Pipeline([
    ("prep", MonthDayPreprocessor()),  # Produces [month_num, day_of_the_year] as np array
    ("sincos", FeatureUnion([
        ("sin_month", sinMonthTransformer()),
        ("cos_month", cosMonthTransformer()),
        ("sin_day", sinDayTransformer()),
        ("cos_day", cosDayTransformer())
    ]))
])


full_feature_pipeline = FeatureUnion([
    ("num_log", ColumnTransformer([
        ("num", num_pipeline, num_att),
        ("log", log_pipeline, log_att),
    ], remainder="drop")),
    
    ("cat", ColumnTransformer([
        ("cat", cat_pipeline, cat_att)
    ], remainder="drop")),
    
    ("sincos", sincos_pipeline),
    ("useless_oof", useless_oof_pipeline),
])

cat_model = CatBoostClassifier(    #0.95607
        iterations=2500,
        learning_rate=0.09706494663217939,
        depth=6,
        l2_leaf_reg=.385284379693478,
        eval_metric="AUC",
        random_seed=42,
        verbose=100,
        bagging_temperature=0.6393738721699852,
        border_count=128,
    )

final_pipeline = Pipeline([
    ("features", full_feature_pipeline),
    ("catboost", cat_model)
])

print("MODEL LOADED")

if __name__ == "__main__":

    scores = cross_val_score(
        final_pipeline, clients_attr, clients_labels, 
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1, 
        verbose=2,
        scoring="roc_auc")
    
    print(f"CV AUC: {scores.mean():.5f}")


    '''
    XGB_model = full_feature_pipeline.fit(clients_attr, clients_labels)
    
    joblib.dump(XGB_model, "models/XGB_finetuned_30h_selector.pkl")

    predictions = XGB_model.predict_proba(clients_test)
    predictions_df = pd.DataFrame( predictions[ :, 1], columns=["y"], index= clients_test.index)

    predictions_df.to_csv("reports/XGB_finetuned_30h_selector.csv")
    '''