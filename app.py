from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load the saved model + encoders
bundle = joblib.load("food_recommender_bundle.pkl")
model = bundle["model"]
le_spice = bundle["le_spice"]
le_taste = bundle["le_taste"]
le_diet = bundle["le_diet"]

# Initialize FastAPI app
app = FastAPI(title="Food Recommendation API")

# Input format
class FoodInput(BaseModel):
    spice: str
    taste: str
    diet: str

# Root route
@app.get("/")
def home():
    return {
        "message": "Welcome to the Food Recommendation API!",
        "note": "Use /predict endpoint with JSON: {'spice': 'high', 'taste': 'sweet', 'diet': 'veg'}"
    }

# Prediction route
@app.post("/predict")
def predict_food(data: FoodInput):
    try:
        # Normalize input (to handle upper/lowercase)
        spice = le_spice.transform([data.spice.lower()])[0]
        taste = le_taste.transform([data.taste.lower()])[0]
        diet = le_diet.transform([data.diet.lower()])[0]

        pred = model.predict([[spice, taste, diet]])[0]
        return {"Recommended Dish": pred}

    except Exception as e:
        return {"error": str(e)}
