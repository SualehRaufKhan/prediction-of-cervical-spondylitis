import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 500

data = {
    "age": np.random.randint(25, 70, n_samples),
    "gender": np.random.choice(["Male", "Female"], n_samples),
    "neck_pain": np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
    "stiffness": np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
    "headache": np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
    "dizziness": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
    "numbness": np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
    "occupation": np.random.choice(["Sedentary", "Manual"], n_samples, p=[0.7, 0.3]),
    "duration_months": np.random.randint(1, 36, n_samples),
    "activity_level": np.random.choice(["Low", "Medium", "High"], n_samples, p=[0.5, 0.3, 0.2]),
}

df = pd.DataFrame(data)

# Simple rule-based target: more symptoms and sedentary = higher risk
df["risk_score"] = (
    df["neck_pain"] + df["stiffness"] + df["headache"] +
    df["dizziness"] + df["numbness"] +
    (df["occupation"] == "Sedentary").astype(int) +
    (df["activity_level"] == "Low").astype(int)
)
df["cervical_spondylitis"] = (df["risk_score"] >= 4).astype(int)
df.drop(columns=["risk_score"], inplace=True)

df.to_csv("cervical_spondylitis_data.csv", index=False)
print("Dataset generated: cervical_spondylitis_data.csv") 