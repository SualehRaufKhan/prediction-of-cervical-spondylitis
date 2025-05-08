from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import joblib
import numpy as np

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


model = joblib.load("model.pkl")
le_gender = joblib.load("le_gender.pkl")
le_occupation = joblib.load("le_occupation.pkl")
le_activity = joblib.load("le_activity.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = [
        int(data["age"]),
        le_gender.transform([data["gender"]])[0],
        int(data["neck_pain"]),
        int(data["stiffness"]),
        int(data["headache"]),
        int(data["dizziness"]),
        int(data["numbness"]),
        le_occupation.transform([data["occupation"]])[0],
        int(data["duration_months"]),
        le_activity.transform([data["activity_level"]])[0],
    ]
    pred = model.predict([features])[0]
    return jsonify({"risk": int(pred)})

if __name__ == "__main__":
    app.run(debug=True) 