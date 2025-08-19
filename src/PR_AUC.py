import joblib
import numpy as np
import random
np.random.seed(42)
random.seed(42)
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
from matplotlib.patches import *
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_loading import standard_training_set, standard_test_set, full_original_database

full_data = full_original_database().reset_index(drop=True)
full_data_clients_attr = full_data.drop("y", axis=1)
full_data_clients_labels = full_data["y"]

model = joblib.load("monster_model3.pkl")

model_scores = model.predict_proba(full_data_clients_attr)[:,1]
precisions_model, recalls_model, thresholds_model = precision_recall_curve(full_data_clients_labels, model_scores)

plt.figure(figsize=(6, 5))  # extra code – not needed, just formatting

plt.plot(recalls_model, precisions_model, "b-", linewidth=2,
         label="model")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.axis([0, 1, 0, 1])
plt.grid()
plt.legend(loc="lower left")


plt.show()


fpr, tpr, thresholds = roc_curve(full_data_clients_labels, model_scores)

plt.figure(figsize=(6, 5))  
plt.plot(fpr, tpr, linewidth=2, label="ROC curve")
plt.plot([0, 1], [0, 1], 'k:', label="Random classifier's ROC curve")
plt.xlabel('False Positive Rate (Fall-Out)')
plt.ylabel('True Positive Rate (Recall)')
plt.grid()
plt.axis([0, 1, 0, 1])
plt.legend(loc="lower right", fontsize=13)


plt.show()