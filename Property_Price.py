
# The code for the API where we are getting the static data for the inputs form the raw data
import os
import streamlit as st
import wandb
import sklearn
from joblib import load
#from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
import wandb
import joblib
import pandas as pd
import numpy as np
import re

os.environ["WANDB_API_KEY"] = "b39f649b2bd3c0fa1e2627832f55119cf5c51ec6"
# Load the dataset from GitHub
#file_path = 'https://raw.githubusercontent.com/karapradeepkumar/Group14_CT1/blob/main/datasets/processed_real_estate_data.parquet'
#data = pd.read_parquet(file_path)

file_path = 'https://raw.githubusercontent.com/karapradeepkumar/data/main/Real%20Estate%20Data%20V21.csv'
data = pd.read_csv(file_path)

# Extract the area name within the city (text after 'in') from Property Title
def extract_area_name(title):
    match = re.search(r'in\s+(.+)', title, re.IGNORECASE)
    return match.group(1).strip() if match else np.nan

data['Area_name'] = data['Property Title'].apply(extract_area_name)
unique_Area = data['Area_name'].unique()

def extract_city(title):
    words = title.strip().split()
    return words[-1] if words else np.nan

data['City'] = data['Property Title'].apply(extract_city)
Unique_City = data['City'].unique()
# To load the model file form local filesystem
#HomePrice = load('PropertyPrice.pkl')

# Load the model from WandB
def load_model_from_wandb(project_name: str, artifact_name: str):
    try:
        # Initialize WandB
        run = wandb.init(project=project_name, job_type="inference", reinit=False)
        artifact = run.use_artifact(artifact_name)
        model_path = artifact.file()  # Assumes a single model file in the artifact
        model = joblib.load(model_path)
        run.finish()  # End the WandB run
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

project_name = "mlops_property_pricing"  # Replace with your project name
artifact_name = "mlops_property_pricing:latest"  # Replace with your artifact name
HomePrice = load_model_from_wandb(project_name, artifact_name)


def predict_price(City,
                  Area_name,
                  Property_Type,
                  Baths,
                  Floor,
                  Total_Area,
                  Bedrooms
                  ):

    inputs_dict = {'City' : City,
                   'Area_name': Area_name,
                   'Property_Type': Property_Type,
                   'Baths': Baths,
                   'Floor': Floor,
                   'Total_Area': Total_Area,
                   'Bedrooms': Bedrooms}

    df = pd.DataFrame(inputs_dict, index = [0])


    price = HomePrice.predict(df)[0]
    return price


#function to define the app_layout
def app_layout():

    st.title('Proprty Price Estimation')
    st.header('Enter property search details:')

    ## Creating the user input fields

    City = st.selectbox('City:',
                         Unique_City)

    filtered_areas = data.loc[data['City'] == City, 'Area_name'].dropna().unique()

    Area_name = st.selectbox('Area Name:', filtered_areas)


    Property_Type = st.radio('Property Type:',
                            ['Flat', 'Independent House', 'Villa', 'Other'],
                            horizontal=True)

    Bedrooms = st.number_input('Bedrooms:',
                               min_value=1,
                               max_value=10,
                               value=1)

    Baths = st.number_input('Baths:',
                           min_value=1,
                           max_value=6,
                           value=1)

    Floor = st.number_input('Floor:',
                          min_value=1,
                          max_value=100,
                          value=1)

    Total_Area = st.number_input('Area in sqft:',
                               min_value=1.0,
                               max_value=50000.0,
                               value=10.0)


    if st.button('Estimate Price'):
        price = predict_price(City,
                              Area_name,
                              Property_Type,
                              Baths,
                              Floor,
                              Total_Area,
                              Bedrooms
                              )
        st.success(f'Estimated price of the property with this configuration is: ₹ {np.round(price, 2)} ')

if __name__=='__main__':
  app_layout()
