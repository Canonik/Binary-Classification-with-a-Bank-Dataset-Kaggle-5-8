import joblib
import numpy as np
import random
np.random.seed(42)
random.seed(42)
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, PolynomialFeatures
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

num_att = ["age", "pdays", "previous","balance","duration","campaign"]
cat_att =["job", "marital", "education", "default", "housing", "loan", "contact", "poutcome"]
time_att = ["day", "month"]
log_att =["balance", "campaign", "duration"]
cyclical_att = ["month_sin", "month_cos", "day_sin", "day_cos"]

clients = standard_training_set().reset_index(drop=True)
clients_attr = clients.drop("y", axis=1)
clients_labels = clients["y"]

clients_test = standard_test_set()

print("data retrieved")

num_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    PolynomialFeatures(degree=6, interaction_only=True, include_bias=False))

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

log_pipeline = make_pipeline(
    LogTransformer(),
    StandardScaler())

cyclical_oof_pipeline = make_pipeline(
    WrapperWithY(RF_useless_features(features=cyclical_att)),
    StandardScaler())



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
        ("log", log_pipeline, log_att)
    ], remainder="drop")),

    ("sincos", sincos_pipeline),
])

X_num = full_feature_pipeline.fit_transform(clients_attr)

# Get categorical columns from original dataframe
X_cat = clients_attr[cat_att].reset_index(drop=True)

# Combine numeric + cat as a single DataFrame (CatBoost can handle strings)
X_combined = pd.concat([pd.DataFrame(X_num), X_cat], axis=1)

# Cat indices (last len(cat_att) columns)
cat_features_indices = list(range(X_combined.shape[1]-len(cat_att), X_combined.shape[1]))

cat_model = CatBoostClassifier(    #0.95607
        iterations=100000,
        learning_rate=0.038,
        depth=13,
        l2_leaf_reg=4,
        eval_metric="AUC",
        task_type="GPU",
        random_seed=42,
        verbose=100,
        bagging_temperature=1,
        min_data_in_leaf=8,
        class_weights = [1, 7.288],
        border_count=512,
    )


print("MODEL LOADED")

if __name__ == "__main__":
    
    '''
    #cross evaluation
    scores = cross_val_score(
        final_pipeline, clients_attr, clients_labels, 
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1, 
        verbose=2,
        scoring="roc_auc")
    
    print(f"CV AUC: {scores.mean():.5f}")
   '''

    # full dataset training
    cat_model.fit(
    X_combined, 
    clients_labels, 
    cat_features=cat_features_indices,
    eval_set=None,  
    verbose=100,
    early_stopping_rounds=500)
    
    print("MODEL_FITTED")


    X_test_num = full_feature_pipeline.transform(clients_test)
    X_test_cat = clients_test[cat_att].reset_index(drop=True)
    X_test_combined = pd.concat([pd.DataFrame(X_test_num), X_test_cat], axis=1)

    predictions = cat_model.predict_proba(X_test_combined)
    predictions_df = pd.DataFrame( predictions[ :, 1], columns=["y"], index= clients_test.index)
    
    print("MODEL_TRAINED")
    predictions_df.to_csv("reports/CatBoost_maxxing1.csv")
    '''

    # out of fold predictions for meta stacking
    predictions_df = pd.DataFrame(np.zeros((len(clients_attr), 1)), columns=["y"], index=clients_attr.index)

    for train_idx, val_idx in StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(clients_attr, clients_labels):
        X_train_fold = clients_attr.iloc[train_idx]
        X_val_fold = clients_attr.iloc[val_idx]
        y_train_fold = clients_labels.iloc[train_idx]
        y_val_fold = clients_labels.iloc[val_idx]


        cat_model = final_pipeline.fit(X_train_fold, y_train_fold)
        predictions_df.iloc[val_idx, 0] = cat_model.predict_proba(X_val_fold)[:, 1]

    predictions_df.to_csv("reports/CatBoost_cyclical_pipeline_cv5.csv")

   '''