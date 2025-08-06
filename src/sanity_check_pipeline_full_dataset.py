import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from Top_models.data_loading import standard_training_set, standard_test_set, full_original_database


num_att = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
cat_att =["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]

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
    OneHotEncoder(handle_unknown="ignore"))

preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_att),
    ("cat", cat_pipeline, cat_att)
])

def K_fold_estimation():
    rnd_forest_model = make_pipeline(preprocessing, RandomForestClassifier())
    scores = cross_val_score(rnd_forest_model, clients_attr, clients_labels,
                         scoring="roc_auc", cv=5, 
                         verbose=2)

    print(scores.mean())

def Total_dataset_training():
    sanity_check_rnd_forest_model = make_pipeline(preprocessing, RandomForestClassifier())
    sanity_check_rnd_forest_model.fit(clients_attr, clients_labels)
    joblib.dump(sanity_check_rnd_forest_model, "models/sanity_check_rnd_forest_model_full_dataset.pkl")

    predictions = sanity_check_rnd_forest_model.predict_proba(clients_test)
    predictions_df = pd.DataFrame( predictions[ :, 1], columns=["y"], index= clients_test.index)

    predictions_df.to_csv("reports/sanity_check_full_dataset_predictions.csv")

if __name__ == "__main__":
    Total_dataset_training()
