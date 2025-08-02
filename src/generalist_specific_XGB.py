import pandas as pd

preds1 = pd.read_csv("reports/XGB_finetuned0.csv")  
preds2 = pd.read_csv("reports/XGB_finetuned1.csv")  


assert all(preds1.index == preds2.index)

blended = pd.DataFrame({"id": preds1["id"],
    "y": 0.7 * preds1["y"] + preds2["y"] * 0.3
}, columns=["id", "y"])

blended.to_csv("reports/XGB_blended_submission.csv", index=False)