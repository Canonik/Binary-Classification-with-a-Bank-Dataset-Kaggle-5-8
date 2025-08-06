import pandas as pd

preds1 = pd.read_csv("reports/XGB_finetuned_30h.csv")  
preds2 = pd.read_csv("reports/Catboost_cyclical.csv")  


assert all(preds1.index == preds2.index)


print(f"difference between the two datasets: {(preds1["y"]-preds2["y"]).mean()}")


'''
blended = pd.DataFrame({"id": preds1["id"],
    "y": 0.85 * preds1["y"] + preds2["y"] * 0.15
}, columns=["id", "y"])

blended.to_csv("reports/XGB_blended_submission6.csv", index=False)
'''