
import os
import wandb
import sklearn
from joblib import load
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
import wandb
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

# Define input data schema
class PredictionRequest(BaseModel):
    City: str
    Area_name: str
    Property_Type: str
    Baths:int
    Total_Area: int
    Bedrooms: int
    Floor: int

# Load the model from WandB
def load_model_from_wandb(project_name: str, artifact_name: str):
    try:
        # Initialize WandB
        run = wandb.init(project=project_name, job_type="inference", reinit=True)
        artifact = run.use_artifact(artifact_name)
        model_path = artifact.file()  # Assumes a single model file in the artifact
        model = joblib.load(model_path)
        run.finish()  # End the WandB run
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

project_name = "mlops_property_pricing"  # Replace with your project name
artifact_name = "mlops_property_pricing:v5"  # Replace with your artifact name
ml_model = load_model_from_wandb(project_name, artifact_name)

# Prediction endpoint
@app.post("/predict")
def predict(input_data: PredictionRequest):
    try:
        # Convert input data to a dictionary for prediction
        input_dict = input_data.dict()

        df = pd.DataFrame(input_dict, index = [0])
        print(df)

        # Call the model's prediction method
        prediction = ml_model.predict(df)
        print(prediction)

        # Return the prediction result
        return {f"Estimated House Price: ₹ {np.round(prediction[0], 2)}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
