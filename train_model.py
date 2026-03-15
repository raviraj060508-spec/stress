import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Example CSV: features already computed
df = pd.read_csv("data/dataset.csv")

X = df.drop("stress", axis=1)
y = df["stress"]

# Train RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/stress_model.pkl")
print("Pre-trained model saved!")
