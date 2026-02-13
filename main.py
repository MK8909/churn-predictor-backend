from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
import pickle
import os

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts whether a telecom customer will churn using a trained RandomForest model.",
    version="1.0.0"
)

# Allow all origins (update in production to your frontend URL)
'''app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)'''


app.add_middleware(
    CORSMiddleware,
   allow_origins=["*"],

    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Load model and encoders once at startup ──────────────────────────────────
'''MODEL_PATH    = os.getenv("MODEL_PATH",   "customer_churn_model_new.pkl")
ENCODER_PATH  = os.getenv("ENCODER_PATH", "encoders_new.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(ENCODER_PATH, "rb") as f:
    encoders = pickle.load(f)

FEATURE_ORDER = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]'''









# ── Load model and encoders once at startup ──────────────────────────────────
MODEL_PATH    = os.getenv("MODEL_PATH",   "customer_churn_model_new.pkl")
ENCODER_PATH  = os.getenv("ENCODER_PATH", "encoders_new.pkl")

with open(MODEL_PATH, "rb") as f:
    model_data = pickle.load(f)

model = model_data["rfc"]                 # 🔥 Extract actual trained model
FEATURE_ORDER = model_data["feature_names"]  # 🔥 Use saved feature order

with open(ENCODER_PATH, "rb") as f:
    encoders = pickle.load(f)


# ── Request / Response schemas ────────────────────────────────────────────────
class CustomerFeatures(BaseModel):
    gender:           Literal["Male", "Female"]
    SeniorCitizen:    Literal[0, 1]             = Field(..., description="0 = No, 1 = Yes")
    Partner:          Literal["Yes", "No"]
    Dependents:       Literal["Yes", "No"]
    tenure:           int                        = Field(..., ge=0, description="Months with the company")
    PhoneService:     Literal["Yes", "No"]
    MultipleLines:    Literal["Yes", "No", "No phone service"]
    InternetService:  Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity:   Literal["Yes", "No", "No internet service"]
    OnlineBackup:     Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport:      Literal["Yes", "No", "No internet service"]
    StreamingTV:      Literal["Yes", "No", "No internet service"]
    StreamingMovies:  Literal["Yes", "No", "No internet service"]
    Contract:         Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod:    Literal[
                          "Electronic check",
                          "Mailed check",
                          "Bank transfer (automatic)",
                          "Credit card (automatic)"
                      ]
    MonthlyCharges:   float = Field(..., ge=0)
    TotalCharges:     float = Field(..., ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85
            }
        }


class PredictionResponse(BaseModel):
    prediction:        str   # "Churn" or "No Churn"
    churn_probability: float = Field(..., description="Probability of churning (0–1)")
    no_churn_probability: float


class BatchRequest(BaseModel):
    customers: list[CustomerFeatures]


class BatchResponse(BaseModel):
    results: list[PredictionResponse]


# ── Helper ────────────────────────────────────────────────────────────────────
def preprocess(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])[FEATURE_ORDER]
    for col in df.columns:
        if col in encoders:
            df[col] = encoders[col].transform(df[col])
    return df


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Churn Prediction API is running."}


@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "healthy",
        "model": str(model),
        "features": FEATURE_ORDER
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(customer: CustomerFeatures):
    """
    Predict churn for a **single** customer.

    Returns:
    - `prediction`: "Churn" or "No Churn"
    - `churn_probability`: probability of churning (0–1)
    - `no_churn_probability`: probability of staying (0–1)
    """
    try:
        df = preprocess(customer.model_dump())
        pred       = model.predict(df)[0]
        proba      = model.predict_proba(df)[0]   # [no_churn, churn]
        return PredictionResponse(
            prediction          = "Churn" if pred == 1 else "No Churn",
            churn_probability   = round(float(proba[1]), 4),
            no_churn_probability= round(float(proba[0]), 4),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
def predict_batch(request: BatchRequest):
    """
    Predict churn for a **list** of customers in one call.
    """
    try:
        rows = [c.model_dump() for c in request.customers]
        df   = pd.DataFrame(rows)[FEATURE_ORDER]
        for col in df.columns:
            if col in encoders:
                df[col] = encoders[col].transform(df[col])

        preds  = model.predict(df)
        probas = model.predict_proba(df)

        results = [
            PredictionResponse(
                prediction          = "Churn" if p == 1 else "No Churn",
                churn_probability   = round(float(pb[1]), 4),
                no_churn_probability= round(float(pb[0]), 4),
            )
            for p, pb in zip(preds, probas)
        ]
        return BatchResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/schema", tags=["Info"])
def schema():
    """Returns the list of required input features and their allowed values."""
    return {
        "features": FEATURE_ORDER,
        "categorical_fields": list(encoders.keys()),
        "numeric_fields": ["tenure", "MonthlyCharges", "TotalCharges"],
    }
