# ============================================================
# 🚚 preprocess.py
# LAST MILE DELIVERY FAILURE PREDICTION
# ============================================================


# ============================================================
# IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder


# ============================================================
# LOAD DATASET
# ============================================================

def load_data(file_path):

    df = pd.read_csv(file_path)

    print("\n✅ Dataset Loaded Successfully")

    print("\nDataset Shape:")
    print(df.shape)

    print("\nDataset Columns:")
    print(df.columns.tolist())

    return df


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

def handle_missing_values(df):

    df = df.ffill()

    print("\n✅ Missing Values Handled")

    return df


# ============================================================
# CREATE TARGET VARIABLE
# ============================================================

def create_target_variable(df):

    # Delivery_Time > 120 minutes = Failure

    df['delivery_status'] = np.where(
        df['Delivery_Time'] > 120,
        1,
        0
    )

    print("\n✅ Target Variable Created")

    print(df['delivery_status'].value_counts())

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def feature_engineering(df):

    # --------------------------------------------------------
    # DATE FEATURES
    # --------------------------------------------------------

    df['Order_Date'] = pd.to_datetime(
        df['Order_Date']
    )

    df['day_of_week'] = df['Order_Date'].dt.dayofweek

    df['month'] = df['Order_Date'].dt.month

    df['is_weekend'] = np.where(
        df['day_of_week'] >= 5,
        1,
        0
    )

    print("\n✅ Date Features Created")


    # --------------------------------------------------------
    # DISTANCE FEATURE
    # --------------------------------------------------------

    def haversine(lat1, lon1, lat2, lon2):

        R = 6371

        lat1 = np.radians(lat1)
        lon1 = np.radians(lon1)

        lat2 = np.radians(lat2)
        lon2 = np.radians(lon2)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat1)
            * np.cos(lat2)
            * np.sin(dlon / 2) ** 2
        )

        return 2 * R * np.arcsin(np.sqrt(a))


    # YOUR DATASET COLUMNS
    df['distance_km'] = haversine(

        df['Store_Latitude'],
        df['Store_Longitude'],

        df['Drop_Latitude'],
        df['Drop_Longitude']
    )

    print("\n✅ Distance Feature Created")

    return df


# ============================================================
# DROP UNNECESSARY COLUMNS
# ============================================================

def drop_columns(df):

    drop_cols = [

        'Order_ID',
        'Order_Date',
        'Delivery_Time',
        'Order_Time',
        'Pickup_Time'

    ]

    for col in drop_cols:

        if col in df.columns:

            df.drop(col, axis=1, inplace=True)

    print("\n✅ Unnecessary Columns Removed")

    return df


# ============================================================
# ENCODE CATEGORICAL FEATURES
# ============================================================

def encode_categorical_features(df):

    le = LabelEncoder()

    categorical_cols = df.select_dtypes(
        include='object'
    ).columns

    print("\nCategorical Columns:")
    print(categorical_cols)

    for col in categorical_cols:

        df[col] = le.fit_transform(
            df[col].astype(str)
        )

    print("\n✅ Categorical Features Encoded")

    return df


# ============================================================
# COMPLETE PREPROCESSING PIPELINE
# ============================================================

def preprocess_pipeline(file_path):

    # Load dataset
    df = load_data(file_path)

    # Handle missing values
    df = handle_missing_values(df)

    # Create target variable
    df = create_target_variable(df)

    # Feature engineering
    df = feature_engineering(df)

    # Drop unnecessary columns
    df = drop_columns(df)

    # Encode categorical features
    df = encode_categorical_features(df)

    print("\n✅ PREPROCESSING COMPLETED SUCCESSFULLY")

    return df


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    file_path = "../data/amazon_delivery.csv"

    df = preprocess_pipeline(file_path)

    print("\nProcessed Dataset Shape:")
    print(df.shape)

    print("\nFirst 5 Rows:")
    print(df.head())