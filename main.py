from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
import pandas as pd
import os

# --- Correctly Import All Modules ---
# Imports from the specific files, not a single 'model.py'
from profiler import calculateNutritionalTarget
from recommender import content_based_recommend, output_recommended_recipes
from allergy_filter import EmbeddingAllergyFilter

# --- Initialize Models and Load Data at Startup ---

app = FastAPI(
    title="Intelligent Diet Recommendation API",
    description="A fully integrated system using a profiler, semantic allergy filter, and a from-scratch recommender.",
    version="5.0.0",
)

# Global variables to hold the loaded data and models for efficiency
dataset = pd.DataFrame()
allergy_filter = None

@app.on_event("startup")
def load_models_and_data():
    """
    This function runs once when the API starts up. It loads the dataset and
    initializes the NLP model to ensure the API is ready for requests.
    """
    global dataset, allergy_filter
    
    data_path = 'data.csv'  # Define the path to your data file
    
    try:
        dataset = pd.read_csv(data_path)
        print(f"Dataset loaded successfully with {len(dataset)} recipes.")

        # Initialize the advanced allergy filter. This will load the spaCy model.
        allergy_filter = EmbeddingAllergyFilter()
        if allergy_filter.nlp:
            # Pre-compute all ingredient vectors for fast filtering later.
            dataset = allergy_filter.precompute_dataset_vectors(dataset)

    except FileNotFoundError:
        print(f"--- FATAL ERROR ---")
        print(f"Dataset '{data_path}' not found. The application cannot start.")
        print(f"Please make sure '{data_path}' is in the same directory as this script.")
        print(f"-------------------")
        dataset = pd.DataFrame()

# --- API Data Models (Pydantic for validation) ---

class Gender(str, Enum): male = "male"; female = "female"
class ActivityLevel(str, Enum): sedentary = "sedentary"; light = "light"; moderate = "moderate"; active = "active"
class Goal(str, Enum): lose = "lose"; maintain = "maintain"; gain = "gain"
class SugarLevelStatus(str, Enum): normal = "normal"; high = "high"

class UserProfile(BaseModel):
    """The structure for receiving a user's health data in a request."""
    age: int = Field(..., gt=0, example=30)
    gender: Gender = Field(..., example="male")
    weight_kg: float = Field(..., gt=0, example=85)
    height_cm: float = Field(..., gt=0, example=180)
    activity_level: ActivityLevel = Field(..., example="moderate")
    goal: Goal = Field(..., example="lose")
    sugar_level_status: SugarLevelStatus = Field(..., example="high")
    allergies: List[str] = Field([], example=["peanut", "shrimp"])

class Recipe(BaseModel):
    """The structure for the recipe data sent back in the response."""
    Name: str
    CookTime: str
    PrepTime: str
    TotalTime: str
    RecipeIngredientParts: list[str]
    Calories: float
    FatContent: float
    SaturatedFatContent: float
    CholesterolContent: float
    SodiumContent: float
    CarbohydrateContent: float
    FiberContent: float
    SugarContent: float
    ProteinContent: float
    RecipeInstructions: list[str]

class PredictionOut(BaseModel):
    """The final structure of the API's response."""
    output: Optional[List[Recipe]] = None
    generated_target: Optional[List[float]] = None
    info: str

# --- API Endpoints ---

@app.get("/")
def home():
    """Health check endpoint to verify the service is running and models are loaded."""
    status = "ready" if not dataset.empty and allergy_filter and allergy_filter.nlp else "error"
    return {"health_check": "OK", "model_status": status}

@app.post("/predict/", response_model=PredictionOut)
def predict_recipes(user_profile: UserProfile):
    """
    The main prediction endpoint that orchestrates the entire recommendation workflow.
    """
    if dataset.empty or not allergy_filter or not allergy_filter.nlp:
        raise HTTPException(status_code=503, detail="Model or dataset not loaded. The service is unavailable.")

    # STAGE 1: Calculate the user's personalized nutritional target
    nutrition_vector = calculateNutritionalTarget(
        weight=user_profile.weight_kg, height=user_profile.height_cm, age=user_profile.age,
        gender=user_profile.gender, activeLevel=user_profile.activity_level,
        goal=user_profile.goal, sugarLevel=user_profile.sugar_level_status
    )

    # STAGE 1A: Use the embedding model to filter the dataset for allergies
    filtered_data = allergy_filter.filter_dataset(dataset, user_profile.allergies)
    
    if filtered_data.empty:
        return {
            "output": None, 
            "generated_target": nutrition_vector, 
            "info": "No recipes found that match your criteria after allergy filtering."
        }

    # STAGE 2: Use the from-scratch recommender to find the single best recipe
    recommendation_df = content_based_recommend(
        dataset=filtered_data,
        nutrition_input=nutrition_vector
    )
    
    output = output_recommended_recipes(recommendation_df)
    
    info_message = f"Successfully found the best match: '{output[0]['Name']}'" if output else "No recommendations found after similarity search."
    
    return {
        "output": output, 
        "generated_target": nutrition_vector, 
        "info": info_message
    }

