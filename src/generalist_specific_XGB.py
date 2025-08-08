import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_loading import standard_training_set, standard_test_set, full_original_database

preds1 = pd.read_csv("reports/CatBoost_cyclical.csv")  
preds2 = pd.read_csv("reports/XGB_best_oof_cat_plus_cat.csv")  


assert all(preds1.index == preds2.index)


print(f'difference between the two datasets: {(preds1["y"]-preds2["y"]).mean()}')



blended = pd.DataFrame({"id": preds1["id"],
    "y": 0.15 * preds1["y"] + preds2["y"] * 0.85
}, columns=["id", "y"])

blended.to_csv("reports/XGB_blended_submission14.csv", index=False)

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