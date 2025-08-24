import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import joblib

def train_and_save_model():
    """
    Loads raw data, trains an XGBoost model to predict flight delays,
    and saves the entire pipeline to a file.
    """
    print("--- Starting ML Model Training ---")
    
    # --- 1. Load Data ---
    try:
        df = pd.read_csv('data/bom_week_flights_synthetic.csv')
        df['sched_time_local'] = pd.to_datetime(df['sched_time_local'])
        df['actual_time_local'] = pd.to_datetime(df['actual_time_local'])
    except FileNotFoundError:
        print("Error: Raw data file not found. Please ensure it's in the data/ folder.")
        return

    # --- 2. Feature Engineering ---
    df['delay_min'] = (df['actual_time_local'] - df['sched_time_local']).dt.total_seconds() / 60
    # Handle potential outliers or negative delays (early flights)
    df['delay_min'] = df['delay_min'].clip(lower=0)
    
    df['sched_hour'] = df['sched_time_local'].dt.hour
    df['sched_weekday'] = df['sched_time_local'].dt.weekday
    df['sched_month'] = df['sched_time_local'].dt.month

    # --- 3. Define Features and Target ---
    # Note: 'tail_id' can have too many unique values, making the model complex.
    # We will use 'origin', 'destination', and time-based features for a robust model.
    features = ['origin', 'destination', 'sched_hour', 'sched_weekday', 'sched_month']
    target = 'delay_min'

    X = df[features]
    y = df[target]

    # --- 4. Preprocessing ---
    categorical_features = ['origin', 'destination']
    numeric_features = ['sched_hour', 'sched_weekday', 'sched_month']

    # Create a preprocessor pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # --- 5. Create and Train the XGBoost Model Pipeline ---
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training XGBoost model...")
    model_pipeline.fit(X_train, y_train)
    
    # --- 6. Evaluate the Model ---
    y_pred = model_pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Model Training Complete. Mean Absolute Error on test data: {mae:.2f} minutes.")
    
    # --- 7. Save the Trained Model Pipeline ---
    model_filename = 'flight_delay_model.joblib'
    joblib.dump(model_pipeline, model_filename)
    print(f"✅ Model pipeline saved successfully as '{model_filename}'")


if __name__ == '__main__':
    train_and_save_model()