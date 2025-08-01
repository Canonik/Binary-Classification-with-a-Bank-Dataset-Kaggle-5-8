import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data_loading import standard_training_set

clients_original = standard_training_set()
clients = clients_original.copy()

clients_features = clients.drop("y", axis = 1)
clients_labels = clients["y"]
print(clients_labels.head())