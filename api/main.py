# ============================================================
# FASTAPI BACKEND
# FILE: api/main.py
# ============================================================

from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd
import pickle

from database.database import (
    insert_prediction,
    fetch_history
)

# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(

    title="Last Mile Delivery Failure Prediction API",

    description="AI-Powered Delivery Analytics Backend",

    version="1.0"

)

# ============================================================
# LOAD MODEL
# ============================================================

model = pickle.load(
    open("../models/delivery_failure_model.pkl", "rb")
)

# ============================================================
# REQUEST MODEL
# ============================================================

class DeliveryInput(BaseModel):

    Store_Latitude: float

    Store_Longitude: float

    Drop_Latitude: float

    Drop_Longitude: float

    Weather: int

    Traffic: int

    Vehicle: int

    Area: int

    Category: int

    Agent_Age: int

    Agent_Rating: float

    day_of_week: int

    month: int

    is_weekend: int

    distance_km: float

# ============================================================
# HOME ROUTE
# ============================================================

@app.get("/")

def home():

    return {

        "message": "🚚 Last Mile Delivery Failure Prediction API Running"

    }

# ============================================================
# PREDICTION API
# ============================================================

@app.post("/predict")

def predict_delivery(data: DeliveryInput):

    # ========================================================
    # CONVERT INPUT TO DATAFRAME
    # ========================================================

    input_data = pd.DataFrame([[

        data.Store_Latitude,
        data.Store_Longitude,

        data.Drop_Latitude,
        data.Drop_Longitude,

        data.Weather,
        data.Traffic,
        data.Vehicle,
        data.Area,
        data.Category,

        data.Agent_Age,
        data.Agent_Rating,

        data.day_of_week,
        data.month,
        data.is_weekend,

        data.distance_km

    ]], columns=[

        'Store_Latitude',
        'Store_Longitude',

        'Drop_Latitude',
        'Drop_Longitude',

        'Weather',
        'Traffic',
        'Vehicle',
        'Area',
        'Category',

        'Agent_Age',
        'Agent_Rating',

        'day_of_week',
        'month',
        'is_weekend',

        'distance_km'

    ])

    # ========================================================
    # MODEL PREDICTION
    # ========================================================

    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)

    success_prob = float(probability[0][0])

    failure_prob = float(probability[0][1])

    # ========================================================
    # PREDICTION LABEL
    # ========================================================

    prediction_text = (

        "Failure"

        if prediction[0] == 1

        else "Success"

    )

    # ========================================================
    # SAVE TO DATABASE
    # ========================================================

    insert_prediction(

        str(data.Weather),
        str(data.Traffic),
        float(data.distance_km),

        prediction_text,

        failure_prob

    )

    # ========================================================
    # RETURN RESPONSE
    # ========================================================

    return {

        "prediction": prediction_text,

        "success_probability": round(success_prob, 4),

        "failure_probability": round(failure_prob, 4)

    }

# ============================================================
# HISTORY API
# ============================================================

@app.get("/history")

def prediction_history():

    history = fetch_history()

    results = []

    for row in history:

        results.append({

            "id": row[0],

            "weather": row[1],

            "traffic": row[2],

            "distance": row[3],

            "prediction": row[4],

            "probability": row[5],

            "created_at": str(row[6])

        })

    return {

        "prediction_history": results

    }

# ============================================================
# ANALYTICS API
# ============================================================

@app.get("/analytics")

def analytics():

    history = fetch_history()

    total_predictions = len(history)

    total_failures = sum(

        1 for row in history

        if row[4] == "Failure"

    )

    total_success = sum(

        1 for row in history

        if row[4] == "Success"

    )

    failure_rate = (

        (total_failures / total_predictions) * 100

        if total_predictions > 0

        else 0

    )

    return {

        "total_predictions": total_predictions,

        "total_failures": total_failures,

        "total_success": total_success,

        "failure_rate": round(failure_rate, 2)

    }

# ============================================================
# RUN MESSAGE
# ============================================================

print("✅ FastAPI Backend Running Successfully")