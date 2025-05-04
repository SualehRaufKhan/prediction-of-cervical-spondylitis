import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load data
df = pd.read_csv("cervical_spondylitis_data.csv")

# Encode categorical variables
le_gender = LabelEncoder()
le_occupation = LabelEncoder()
le_activity = LabelEncoder()
df["gender"] = le_gender.fit_transform(df["gender"])
df["occupation"] = le_occupation.fit_transform(df["occupation"])
df["activity_level"] = le_activity.fit_transform(df["activity_level"])

X = df.drop("cervical_spondylitis", axis=1)
y = df["cervical_spondylitis"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model and encoders
joblib.dump(model, "model.pkl")
joblib.dump(le_gender, "le_gender.pkl")
joblib.dump(le_occupation, "le_occupation.pkl")
joblib.dump(le_activity, "le_activity.pkl")
print("Model and encoders saved.") 