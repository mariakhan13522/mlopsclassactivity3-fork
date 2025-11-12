import os
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target'] = iris.target

# Create data directory
os.makedirs("data", exist_ok=True)

data.to_csv("data/preprocessed.csv", index=False)
print("✅ Data preprocessed and saved to data/preprocessed.csv")
