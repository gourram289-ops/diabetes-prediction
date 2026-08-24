from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import pandas as pd
import joblib


app = FastAPI(
    title="Diabetes Prediction API",
    description="API for diabetes prediction",
    version="1.0.0"
)


# ==========================================
# Input Schema
# ==========================================

class DiabetesData(BaseModel):

    gender: str = Field(
        ...,
        examples=["Male"]
    )

    age: float = Field(
        ...,
        ge=0,
        le=120,
        examples=[40]
    )

    hypertension: int = Field(
        ...,
        ge=0,
        le=1,
        examples=[0]
    )

    heart_disease: int = Field(
        ...,
        ge=0,
        le=1,
        examples=[0]
    )

    smoking_history: str = Field(
        ...,
        examples=["never"]
    )

    bmi: float = Field(
        ...,
        ge=10,
        le=100,
        examples=[23.23]
    )

    HbA1c_level: float = Field(
        ...,
        ge=3,
        le=20,
        examples=[5.8]
    )

    blood_glucose_level: int = Field(
        ...,
        ge=40,
        le=500,
        examples=[150]
    )


# ==========================================
# Load Model
# ==========================================

model_path = "diabetes_model.pkl"

model_pipeline = None


@app.on_event("startup")
def load_model():

    global model_pipeline

    try:
        model_pipeline = joblib.load(model_path)

        print("Model loaded successfully!")

    except Exception as e:

        print(f"Model loading error: {e}")

        raise RuntimeError(
            "Could not load the model"
        )


# ==========================================
# Health Check
# ==========================================

@app.get("/")
def health_check():

    return {
        "status": "online",
        "model_loaded": model_pipeline is not None
    }


# ==========================================
# Prediction
# ==========================================

@app.post("/predict")
def predict_diabetes(data: DiabetesData):

    if model_pipeline is None:

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded or unavailable."
        )

    try:

        # Convert Pydantic model to dictionary
        input_dict = data.model_dump()

        # Convert dictionary to DataFrame
        input_df = pd.DataFrame([input_dict])

        # Prediction
        prediction = model_pipeline.predict(input_df)

        diabetes = int(prediction[0])

        # Convert prediction into readable result
        if diabetes == 1:
            result = "Diabetes"
        else:
            result = "No Diabetes"

        return {
            "prediction": diabetes,
            "result": result
        }

    except Exception as e:

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prediction error: {str(e)}"
        )