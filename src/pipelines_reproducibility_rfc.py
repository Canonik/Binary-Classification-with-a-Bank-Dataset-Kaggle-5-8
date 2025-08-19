import joblib
import numpy as np
import random
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel
from catboost import CatBoostClassifier
from scipy.stats import randint, uniform
import datetime

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_loading import standard_training_set, standard_test_set, full_original_database
from pipelines import day_month_encoding, log_encoding, month_sin_encoding, month_cos_encoding, LogTransformer, DayOfYearTransformer, RF_useless_features, WrapperWithY, OOFJobTransformer, sinDayTransformer,cosDayTransformer,sinMonthTransformer,cosMonthTransformer, MonthDayPreprocessor, NoiseAdder

num_att = ["age", "pdays"] 
time_att = ["day", "month"]
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
    StandardScaler()
    )


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

sincos_pipeline = Pipeline([
    ("prep", MonthDayPreprocessor()),  # Produces [month_num, day_of_the_year] as np array
    ("sincos", FeatureUnion([
        ("sin_month", sinMonthTransformer()),
        ("cos_month", cosMonthTransformer()),
        ("sin_day", sinDayTransformer()),
        ("cos_day", cosDayTransformer())
    ]))
])

preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_att),
    ("sincos", sincos_pipeline, time_att),
], remainder="drop")


full_pipeline = Pipeline([
    ("full_preprocessing", preprocessing),
    ("model", KNeighborsClassifier(
    n_neighbors=20,
    weights='distance',    
    metric='cosine',     
    p=1  
))
])

if __name__ == "__main__":
    '''
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    auc_scores = []
    feature_importances = []
    feature_names = None  

    for fold, (train_idx, val_idx) in enumerate(cv.split(clients_attr, clients_labels), 1):
        print(f"Training fold {fold}...")
    
        X_train, X_val = clients_attr.iloc[train_idx], clients_attr.iloc[val_idx]
        y_train, y_val = clients_labels.iloc[train_idx], clients_labels.iloc[val_idx]
    
    
        full_pipeline.fit(X_train, y_train)
    
    
        if feature_names is None:
            preprocessing_step = full_pipeline.named_steps["full_preprocessing"]

            if isinstance(preprocessing_step, FeatureUnion):
                feature_names_parts = []
                for name, transformer in preprocessing_step.transformer_list:
                    try:
                        feature_names_parts.extend(transformer.get_feature_names_out())
                    except AttributeError:
                        pass
                feature_names = np.array(feature_names_parts)

            # If it's a Pipeline with a ColumnTransformer (like your second XGB)
            elif isinstance(preprocessing_step, ColumnTransformer):
                try:
                    feature_names = preprocessing_step.get_feature_names_out()
                except AttributeError:
                    # If no get_feature_names_out, fallback to generic numbering
                    feature_names = np.array([f"feature_{i}" for i in range(full_pipeline.named_steps["model"].n_features_in_)])

            else:
                raise ValueError("Unknown preprocessing structure.")

            print(f"Extracted {len(feature_names)} feature names.")



    
        y_pred = full_pipeline.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)
        auc_scores.append(auc)
    

        model = full_pipeline.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feature_importances.append(importances)
    
        print(f"Fold {fold} AUC: {auc:.5f}")


    print(f"\nMean CV AUC: {np.mean(auc_scores):.5f} ± {np.std(auc_scores):.5f}")


    mean_importances = np.mean(feature_importances, axis=0)
    importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": mean_importances
    }).sort_values(by="importance", ascending=False)

    print("\nTop 20 features by importance:")
    print(importance_df.head(20))

    importance_df.to_csv("xgb_feature_importance.csv", index=False)
    '''

#full pipeline on full training set
    XGB_model = full_pipeline.fit(clients_attr, clients_labels)
    
    joblib.dump(XGB_model, "models/kneighbor_blending_weak.pkl")
    
    predictions = XGB_model.predict_proba(clients_test)
    predictions_df = pd.DataFrame( predictions[ :, 1], columns=["y"], index= clients_test.index)

    predictions_df.to_csv("reports/kneighbor_blending_weak.csv")

     