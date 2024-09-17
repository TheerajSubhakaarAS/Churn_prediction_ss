from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import shap
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Define paths
models_folder = "./models"

def load_model(model_name, dataset_name):
    model_path = os.path.join(models_folder, f"{model_name}_{dataset_name}.pkl")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model {model_name} for dataset {dataset_name} not found")
    return joblib.load(model_path)

def load_preprocessor(dataset_name):
    preprocessor_path = os.path.join(models_folder, f"preprocessor_{dataset_name}.pkl")
    if not os.path.exists(preprocessor_path):
        raise HTTPException(status_code=404, detail=f"Preprocessor for dataset {dataset_name} not found")
    return joblib.load(preprocessor_path)

# Define request body model
class PredictionRequest(BaseModel):
    dataset: str
    features: dict

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_html():
    with open("static/index.html") as f:
        return f.read()

@app.post("/predict/")
async def predict(request: PredictionRequest):
    try:
        model = load_model("RandomForest", request.dataset)  # Example: Using RandomForest
        preprocessor = load_preprocessor(request.dataset)
        
        feature_df = pd.DataFrame([request.features])
        X_transformed = preprocessor.transform(feature_df)
        
        prediction = model.predict(X_transformed)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain/")
async def explain(request: PredictionRequest):
    try:
        model = load_model("RandomForest", request.dataset)  # Example: Using RandomForest
        preprocessor = load_preprocessor(request.dataset)
        
        feature_df = pd.DataFrame([request.features])
        X_transformed = preprocessor.transform(feature_df)
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed)
        feature_names = preprocessor.get_feature_names_out()
        
        shap_summary = dict(zip(feature_names, shap_values.mean(axis=0)))
        
        return {"shap_summary": shap_summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the app using: uvicorn fastapi:app --reload
