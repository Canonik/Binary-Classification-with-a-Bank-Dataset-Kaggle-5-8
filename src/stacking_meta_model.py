import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
import joblib
from data_loading import standard_test_set, standard_training_set, full_original_database

scaler = StandardScaler()
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
xgb_cv5_preds =  pd.read_csv("reports/XGB_finetuned_30h_pipeline_cv5.csv", index_col=0)
cat_cv5_preds = pd.read_csv("reports/CatBoost_cyclical_pipeline_cv5.csv", index_col=0)
  
clients = standard_training_set().reset_index(drop=True)
clients_attr = clients.drop("y", axis=1)
Y1_meta = clients["y"]

full_data = full_original_database().reset_index(drop=True)
full_data_clients_attr = full_data.drop("y", axis=1)
Y2_meta = full_data["y"]
clients_test = standard_test_set()

Y_train_meta = pd.concat(
    [Y1_meta.reset_index(drop=True), Y2_meta.reset_index(drop=True)],
    ignore_index=True
)

xgb_cv5_preds = xgb_cv5_preds.reset_index(drop=True)
cat_cv5_preds = cat_cv5_preds.reset_index(drop=True)

#additional features
original_cat_train = pd.concat([clients_attr, full_data_clients_attr], ignore_index=True)[["job", "month"]].reset_index(drop=True)
original_num_train = pd.concat([clients_attr, full_data_clients_attr], ignore_index=True)[["day"]].reset_index(drop=True)

original_cat_test = clients_test[["job","month"]].reset_index(drop=True)
original_num_test = clients_test[["day"]].reset_index(drop=True)


encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoded_cat_train = encoder.fit_transform(original_cat_train)
encoded_cat_test = encoder.transform(original_cat_test)


scaler = StandardScaler()
scaled_num_train = scaler.fit_transform(original_num_train)
scaled_num_test = scaler.transform(original_num_test)


encoded_cat_train_df = pd.DataFrame(encoded_cat_train, columns=encoder.get_feature_names_out(["job","month"]))
encoded_cat_test_df = pd.DataFrame(encoded_cat_test, columns=encoder.get_feature_names_out(["job","month"]))

scaled_num_train_df = pd.DataFrame(scaled_num_train, columns=["day"])
scaled_num_test_df = pd.DataFrame(scaled_num_test, columns=["day"])


scaled_train_df = pd.concat([encoded_cat_train_df, scaled_num_train_df], axis=1).reset_index(drop=True)
scaled_test_df = pd.concat([encoded_cat_test_df, scaled_num_test_df], axis=1).reset_index(drop=True)

X_train_meta = pd.concat(
    [xgb_cv5_preds.rename(columns={"y":"xgboost"}), 
     cat_cv5_preds.rename(columns={"y":"catboost"}),
     scaled_train_df], 
     axis=1
)

preds1 = pd.read_csv("reports/CatBoost_cyclical_pipeline.csv", index_col=0)  
preds2 = pd.read_csv("reports/XGB_best_oof_cat_plus_cat.csv", index_col=0)  

assert all(preds1.index == preds2.index)

preds1 = preds1.reset_index(drop=True)
preds2 = preds2.reset_index(drop=True)

X_test_meta = pd.concat(
    [preds1.rename(columns={"y":"xgboost"}), 
     preds2.rename(columns={"y":"catboost"}),
     scaled_test_df], 
     axis=1
)


from lightgbm import LGBMClassifier

meta_model = XGBClassifier(
    colsample_bytree=0.8,
    gamma=0.3,
    learning_rate=0.02,
    max_depth=4,                
    min_child_weight=1,
    n_estimators=5000,
    reg_alpha=0.1,
    reg_lambda=1.0,
    subsample=0.8,
    use_label_encoder=False,
    eval_metric="auc",
    tree_method="gpu_hist",         
    random_state=42
)


meta_model.fit(X_train_meta, Y_train_meta)
predictions = meta_model.predict_proba(X_test_meta)
predictions_df = pd.DataFrame( predictions[ :, 1], columns=["y"], index= preds1.index)
predictions_df["id"] = predictions_df.index + 750000
predictions_df= predictions_df[["id", "y"]]

predictions_df.to_csv("reports/model_stacking_xgb.csv", index=False)
'''
scores = cross_val_score(meta_model, X_train_meta, Y_train_meta,
                         scoring="roc_auc", cv=5, 
                         verbose=2)

print(scores.mean())
'''