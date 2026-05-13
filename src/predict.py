# ============================================================
# 🚚 predict.py
# LAST MILE DELIVERY FAILURE PREDICTION
# ============================================================

import pickle
import pandas as pd

# ============================================================
# LOAD MODEL
# ============================================================

model = pickle.load(
    open("../models/delivery_failure_model.pkl", "rb")
)

print("\n✅ Model Loaded Successfully")

# ============================================================
# CREATE SAMPLE INPUT
# ============================================================

sample = pd.DataFrame([[

    12.9716,   # Store_Latitude
    77.5946,   # Store_Longitude

    12.9352,   # Drop_Latitude
    77.6245,   # Drop_Longitude

    0,         # Weather
    1,         # Traffic
    0,         # Vehicle
    2,         # Area
    1,         # Category

    25,        # Agent_Age
    4.5,       # Agent_Rating

    2,         # day_of_week
    5,         # month
    0,         # is_weekend

    10.5       # distance_km

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

print("\n✅ Sample Input Created")

print("\nInput Data:\n")

print(sample)

# ============================================================
# PREDICTION
# ============================================================

prediction = model.predict(sample)

probability = model.predict_proba(sample)

# ============================================================
# RESULTS
# ============================================================

success_prob = probability[0][0]

failure_prob = probability[0][1]

print("\n===================================")
print("PREDICTION RESULT")
print("===================================")

if prediction[0] == 1:

    print("\n❌ Delivery Failure Predicted")

else:

    print("\n✅ Delivery Success Predicted")

print("\nSuccess Probability:", round(success_prob, 2))

print("Failure Probability:", round(failure_prob, 2))

print("\n===================================")

print("🚚 PREDICTION COMPLETED SUCCESSFULLY")

print("===================================")