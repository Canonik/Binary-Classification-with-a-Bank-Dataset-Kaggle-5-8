import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from Top_models.data_loading import standard_training_set, standard_test_set, full_original_database


clients = standard_training_set().copy()
clients_features = clients.drop("y", axis=1)
client_labels = clients["y"]


def int64_plot():
    for column in clients_features.columns:
        if column != "id":
            print(clients_features[column].value_counts())


def object_plot():
    for column in clients_features.columns:
        if clients_features[column].dtype == 'object':  
            clients_features[column].value_counts().plot(kind='bar')
            plt.title(f"Value counts of {column}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

def correlation_matrix():
    corr_matrix = clients.corr(numeric_only=True)
    sorted_corr = corr_matrix["y"].sort_values(ascending=False)
    return sorted_corr

print(correlation_matrix())