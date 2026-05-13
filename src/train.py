# ============================================================
# 🚚 train.py
# LAST MILE DELIVERY FAILURE PREDICTION
# ============================================================


# ============================================================
# IMPORT LIBRARIES
# ============================================================

import pandas as pd
import pickle

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

import seaborn as sns
import matplotlib.pyplot as plt


# ============================================================
# LOAD PREPROCESSED DATA
# ============================================================

from preprocess import preprocess_pipeline


# ============================================================
# LOAD & PREPROCESS DATASET
# ============================================================

file_path = "../data/amazon_delivery.csv"

df = preprocess_pipeline(file_path)

print("\n✅ Dataset Ready For Training")


# ============================================================
# DEFINE FEATURES & TARGET
# ============================================================

X = df[[

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

]]

print(X.columns.tolist())

y = df['delivery_status']

print("\nFeature Shape:", X.shape)

print("Target Shape:", y.shape)


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42

)

print("\n✅ Train-Test Split Completed")

print("\nTraining Data Shape:", X_train.shape)

print("Testing Data Shape:", X_test.shape)


# ============================================================
# MODEL TRAINING
# ============================================================

model = RandomForestClassifier(

    n_estimators=100,

    random_state=42,

    n_jobs=-1

)

model.fit(
    X_train,
    y_train
)

print("\n✅ Model Training Completed")


# ============================================================
# PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:,1]


# ============================================================
# MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

roc_score = roc_auc_score(
    y_test,
    y_prob
)

print("\n===================================")
print("MODEL PERFORMANCE")
print("===================================")

print("\nAccuracy Score:")
print(round(accuracy * 100, 2), "%")

print("\nROC-AUC Score:")
print(round(roc_score, 4))

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(6,5))

sns.heatmap(

    cm,

    annot=True,

    fmt='d',

    cmap='Blues'

)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame({

    'Feature': X.columns,

    'Importance': model.feature_importances_

})

importance_df = importance_df.sort_values(

    by='Importance',

    ascending=False

)

print("\nTop 10 Important Features:\n")

print(
    importance_df.head(10)
)


# ------------------------------------------------------------
# FEATURE IMPORTANCE PLOT
# ------------------------------------------------------------

plt.figure(figsize=(10,6))

sns.barplot(

    x='Importance',

    y='Feature',

    data=importance_df.head(10)

)

plt.title("Top 10 Important Features")

plt.show()


# ============================================================
# SAVE MODEL
# ============================================================

model_path = "../models/delivery_failure_model.pkl"

pickle.dump(

    model,

    open(model_path, "wb")

)

print("\n✅ Model Saved Successfully")

print("\nModel Path:")

print(model_path)


# ============================================================
# SAMPLE PREDICTION
# ============================================================

sample = X_test.iloc[0:1]

prediction = model.predict(sample)

print("\n===================================")
print("SAMPLE PREDICTION")
print("===================================")

if prediction[0] == 1:

    print("\n❌ Delivery Failure Predicted")

else:

    print("\n✅ Delivery Success Predicted")


# ============================================================
# PROJECT COMPLETED
# ============================================================

print("\n==============================================")

print("🚚 MODEL TRAINING COMPLETED SUCCESSFULLY")

print("==============================================")