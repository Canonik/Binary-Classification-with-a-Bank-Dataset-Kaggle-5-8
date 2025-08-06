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
from scipy.stats import randint, uniform
import datetime

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from Top_models.data_loading import standard_training_set, standard_test_set, full_original_database
from pipelines import day_month_encoding, log_encoding, LogTransformer, DayOfYearTransformer

num_att = ["age", "pdays", "previous"]
cat_att =["job", "marital", "education", "default", "housing", "loan", "contact", "poutcome"]
log_att =["balance", "duration", "campaign"]
time_att = ["day", "month"]


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



preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_att),
    ("cat", cat_pipeline, cat_att),
    ("log", log_pipeline, log_att),
])
preprocessing_doy = ColumnTransformer([
    ("day", day_pipeline, time_att)
])

full_feature_pipeline = FeatureUnion([
    ("preprocessing", preprocessing),
    ("preprocessing_doy", preprocessing_doy)
])

def K_fold_estimation():
    XGB_model = make_pipeline(full_feature_pipeline, XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42))
    scores = cross_val_score(XGB_model, clients_attr, clients_labels,
                         scoring="roc_auc", cv=5, 
                         verbose=2)

    print(scores.mean())

def Total_dataset_training():
    
    best_params = {

    'colsample_bytree': 0.42177,
    'gamma': 1.3786,
    'learning_rate': 0.0174,
    'max_depth': 13,
    'min_child_weight': 10,
    'n_estimators': 5014,
    'reg_alpha': 4.4974,
    'reg_lambda': 15.5342,
    'scale_pos_weight': 1.5883,
    'subsample': 0.7950
}
    XGB_model = make_pipeline(full_feature_pipeline, XGBClassifier(use_label_encoder=False, eval_metric="auc", random_state=42, **best_params))
    XGB_model.fit(clients_attr, clients_labels)
    joblib.dump(XGB_model, "models/XGB_finetuned_30h.pkl")

    predictions = XGB_model.predict_proba(clients_test)
    predictions_df = pd.DataFrame( predictions[ :, 1], columns=["y"], index= clients_test.index)

    predictions_df.to_csv("reports/XGB_finetuned_30h.csv")

def Fine_tuning_pipeline():
    parameters = {
    "xgbclassifier__n_estimators": randint(3000, 6001),      
    "xgbclassifier__learning_rate": uniform(0.005, 0.045),        
    "xgbclassifier__max_depth": randint(6, 17),                   
    "xgbclassifier__min_child_weight": randint(1, 21),            
    "xgbclassifier__gamma": uniform(0, 5),                   
    "xgbclassifier__subsample": uniform(0.6, 0.4),                
    "xgbclassifier__colsample_bytree": uniform(0.4, 0.6),        
    "xgbclassifier__reg_alpha": uniform(0, 10),                  
    "xgbclassifier__reg_lambda": uniform(0, 20),
    "xgbclassifier__scale_pos_weight": uniform(1,9)                 
    }

    XGB_model = make_pipeline(full_feature_pipeline, XGBClassifier(use_label_encoder=False, eval_metric="auc", random_state=42, n_estimators=6000))
    rnd_search = RandomizedSearchCV(XGB_model, parameters, n_iter=400, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="roc_auc", random_state = 42, verbose=2)


    rnd_search.fit(clients_attr, clients_labels)

    print(f"Best ROC AUC: {rnd_search.best_score_:.5f}")
    print("Best parameters:", rnd_search.best_params_)

    best_model = rnd_search.best_estimator_
    xgb_model = best_model.named_steps["xgbclassifier"]

    preprocessing = best_model.named_steps["featureunion"].transformer_list[0][1]
    preprocessing_doy = best_model.named_steps["featureunion"].transformer_list[1][1]

    feature_names = np.concatenate([
        preprocessing.get_feature_names_out(),
        preprocessing_doy.get_feature_names_out()
    ])

    sorted_features = sorted(zip(xgb_model.feature_importances_, feature_names), reverse=True)

    for importance, name in sorted_features[:20]:
        print(f"{name}: {importance:.4f}")

    pd.DataFrame(sorted_features, columns = ["importance", "feature"]).to_csv("reports/XGB_30hours.csv")
    pd.DataFrame([rnd_search.best_params_]).to_csv("reports/XGB_30hours.csv")

if __name__ == "__main__":
    Total_dataset_training()