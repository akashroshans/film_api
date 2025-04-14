from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your ML model
model = joblib.load("film_model.pkl")

# Define request schema
class FilmFeatures(BaseModel):
    R: int
    G: int
    B: int

# Define prediction endpoint
@app.post("/predict")
def predict(features: FilmFeatures):
    X = [[features.R, features.G, features.B]]
    prediction = model.predict(X)[0]
    return {"predicted_film_type": prediction}
