from fastapi import FastAPI
from pydantic import BaseModel
from flask_cors import CORS
import joblib

app = FastAPI()
model = joblib.load("film_model.pkl")
CORS(app)
class FilmFeatures(BaseModel):
    R: int
    G: int
    B: int

@app.post("/predict")
def predict(features: FilmFeatures):
    X = [[features.R, features.G, features.B]]
    prediction = model.predict(X)[0]
    return {"predicted_film_type": prediction}
