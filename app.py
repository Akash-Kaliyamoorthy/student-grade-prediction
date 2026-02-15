
from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.get("/")
def home():
    return {"message": "Student Grade Prediction API Running"}

@app.post("/predict/")
def predict(data: list):
    data = scaler.transform([data])
    prediction = model.predict(data)
    return {"predicted_grade": float(prediction[0])}
