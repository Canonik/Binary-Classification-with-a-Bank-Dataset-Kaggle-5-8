import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_loading import standard_training_set, standard_test_set, full_original_database

#XGB_finetuned_30h_pipeline.csv
#XGB_blended_submission11.csv
#model_stacking_xgb.csv
#reports/XGB_best_oof_cat_plus_cat.csv
#XGB_finetuned_overfit_beast.csv
#rfc_blending_weak.csv
#kneighbor_blending_weak.csv
#"reports/rf_poly_pipeline.csv"
#CatBoost_maxxing1.csv"

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


preds1 = pd.read_csv("reports/XGB_maxxing5.csv")
preds2 = pd.read_csv("reports/XGB_maxxing1.csv")   
preds3 = pd.read_csv("reports/XGB_blended_submission11.csv")       


assert all(preds1["id"] == preds2["id"]), "IDs do not match!"
assert all(preds2["id"] == preds3["id"]), "IDs do not match!"

# Extract y values and reshape
p_1 = preds1["y"].values.reshape(-1, 1)
p_2 = preds2["y"].values.reshape(-1, 1)
p_3 = preds3["y"].values.reshape(-1, 1)

# Pearson correlations & mean diffs (optional, for info)
corr_2 = np.corrcoef(p_1.flatten(), p_2.flatten())[0, 1]
diff_2 = (p_1.flatten() - p_2.flatten()).mean()
corr_3 = np.corrcoef(p_3.flatten(), p_1.flatten())[0, 1]
diff_3 = (p_3.flatten() - p_1.flatten()).mean()

print(f"Second Pearson correlation with XGB: {corr_2:.6f}, mean diff: {diff_2:.6f}")
print(f"Third Pearson correlation with XGB: {corr_3:.6f}, mean diff: {diff_3:.6f}")

# Scale each prediction separately to [0,1]
scaler_xgb = MinMaxScaler()
scaler_ffnn = MinMaxScaler()
scaler_nn3 = MinMaxScaler()

p_xgb_scaled = scaler_xgb.fit_transform(p_1).flatten()
p_ffnn_scaled = scaler_ffnn.fit_transform(p_2).flatten()
p_nn3_scaled = scaler_nn3.fit_transform(p_3).flatten()

# Set blend weights - sum to 1.0
w_xgb = 0.25
w_2 = 0.35
w_3 = 0.40

# Blend predictions
blended_y = w_xgb * p_xgb_scaled + w_2 * p_ffnn_scaled + w_3 * p_nn3_scaled

# Save submission
blended = pd.DataFrame({
    "id": preds1["id"],
    "y": blended_y
})
blended.to_csv("reports/blended_submission15.csv", index=False)
print("Blended submission saved.")


'''

uncertain_xgb = preds1[(preds1['y'] > 0.45) & (preds1['y'] < 0.55)].copy()

uncertain_cat = preds2[(preds2['y'] > 0.45) & (preds2['y'] < 0.55)].copy()


uncertain_xgb['uncertainty'] = (0.5 - uncertain_xgb.loc[:, 'y']).abs()
uncertain_xgb = uncertain_xgb.sort_values('uncertainty')

#print(uncertain_xgb.head(9999))

uncertain_cat['uncertainty'] = (0.5 - uncertain_cat.loc[:, 'y']).abs()
uncertain_cat = uncertain_cat.sort_values('uncertainty')

#print(uncertain_cat.head(9999))



test_set = pd.read_csv(PROJECT_ROOT / "data/playground-series-s5e8/train.csv")
df = pd.DataFrame(test_set, index=uncertain_xgb.index)
df.to_csv("src/XGB_uncertains.csv")

print(test_set.describe())
print(df.describe())
'''