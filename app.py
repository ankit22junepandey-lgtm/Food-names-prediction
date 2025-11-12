from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="Food Recommendation API", version="0.1.0")

# Load model bundle
bundle = joblib.load("food_recommender_bundle.pkl")
model = bundle["model"]
le_spice = bundle["le_spice"]
le_taste = bundle["le_taste"]
le_diet = bundle["le_diet"]

# Input schema
class FoodInput(BaseModel):
    spice: str
    taste: str
    diet: str

@app.get("/")
def home():
    return {"message": "Welcome to Food Recommendation API"}

@app.post("/predict")
def predict_food(input_data: FoodInput):
    # Convert inputs to the same format as training
    spice = input_data.spice.strip().capitalize()
    taste = input_data.taste.strip().capitalize()
    diet = input_data.diet.strip().capitalize()

    # Encode
    try:
        spice_encoded = le_spice.transform([spice])[0]
        taste_encoded = le_taste.transform([taste])[0]
        diet_encoded = le_diet.transform([diet])[0]
    except ValueError as e:
        return {"error": f"Invalid input: {e}"}

    # Predict
    prediction = model.predict([[spice_encoded, taste_encoded, diet_encoded]])[0]
    return {"predicted_food": prediction}
