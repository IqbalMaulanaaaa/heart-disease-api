from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Load model, scaler, dan kolom training
model_data = joblib.load("model_random_forest_full.pkl")
model = model_data["model"]
model_columns = model_data["columns"]

scaler = joblib.load("scaler.pkl")

numerical_features = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']

# Fungsi preprocessing
def prepare_input_json(raw_data_dict):
    raw_df = pd.DataFrame([raw_data_dict])
    df_encoded_input = pd.get_dummies(raw_df)
    # Tambahkan kolom hilang
    for col in model_columns:
        if col not in df_encoded_input.columns:
            df_encoded_input[col] = 0
    df_encoded_input = df_encoded_input[model_columns]
    # Scaling numerik
    df_encoded_input[numerical_features] = scaler.transform(df_encoded_input[numerical_features])
    return df_encoded_input

def convert_flutter_input_to_dummy(flutter_input):
    mapping = {
        'Sex': {0:'F',1:'M'},
        'ChestPainType': {0:'ATA',1:'NAP',2:'TA',3:'ASY'},
        'RestingECG': {0:'Normal',1:'ST',2:'LVH'},
        'ExerciseAngina': {0:'N',1:'Y'},
        'ST_Slope': {0:'Down',1:'Flat',2:'Up'}
    }
    raw_data = {}
    for key in flutter_input:
        if key in mapping:
            raw_data[key] = mapping[key][flutter_input[key]]
        else:
            raw_data[key] = flutter_input[key]
    return prepare_input_json(raw_data)

# Fungsi prediksi
def predict_heart_disease(input_json):
    X_input_ready = convert_flutter_input_to_dummy(input_json)
    prediction = model.predict(X_input_ready)[0]
    probability = model.predict_proba(X_input_ready)[0][1]
    return {"prediction": int(prediction), "probability": float(probability)}

# Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        result = predict_heart_disease(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

import os

if __name__ == "__main__":
    app.run(debug=True)
