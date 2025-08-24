import pandas as pd
import os
import joblib
import re
import numpy as np # THIS IS THE FIX
from warnings import filterwarnings

# Ignore harmless warnings from sklearn
filterwarnings('ignore', category=UserWarning, module='sklearn')

# Load the trained ML model once when the script is imported
try:
    ML_MODEL = joblib.load('flight_delay_model.joblib')
    print("✅ XGBoost delay prediction model loaded successfully.")
except FileNotFoundError:
    print("⚠️ Model file 'flight_delay_model.joblib' not found. Please run train_model.py first.")
    ML_MODEL = None

def parse_delay_from_string(text: str) -> float:
    # ... (this function is correct)
    match = re.search(r"(\d+\.\d+)", text)
    if match:
        return float(match.group(1))
    match_int = re.search(r"(\d+)", text)
    if match_int:
        return float(match_int.group(1))
    return 0.0

# Define file paths
RAW_DATA_PATH = 'data/bom_week_flights_synthetic.csv'
OUTPUT_DIR = 'data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_flight_data():
    # ... (this function is correct)
    print("Starting data processing...")
    try:
        df = pd.read_csv(RAW_DATA_PATH)
        df['sched_time_local'] = pd.to_datetime(df['sched_time_local'])
        df['actual_time_local'] = pd.to_datetime(df['actual_time_local'])
        print(f"Loaded raw data: {df.shape[0]} flights.")
    except FileNotFoundError:
        print(f"ERROR: Raw data file not found at {RAW_DATA_PATH}")
        return None, None
    df['delay_min'] = (df['actual_time_local'] - df['sched_time_local']).dt.total_seconds() / 60
    df['hour'] = df['sched_time_local'].dt.hour
    busiest_hours = df.groupby('hour').size().reset_index(name='ops_count').sort_values('ops_count', ascending=False)
    busiest_hours.to_csv(os.path.join(OUTPUT_DIR, 'busiest_hours.csv'), index=False)
    avg_delay_by_hour = df.groupby('hour')['delay_min'].mean().reset_index()
    avg_delay_by_hour.to_csv(os.path.join(OUTPUT_DIR, 'avg_delay_by_hour.csv'), index=False)
    best_hours = avg_delay_by_hour.sort_values('delay_min').head(5)
    best_hours.to_csv(os.path.join(OUTPUT_DIR, 'best_hours.csv'), index=False)
    print("Data processing complete.")
    return df, avg_delay_by_hour

def predict_delay_for_new_time(flight_id: str, new_time_hour: int, full_flight_df: pd.DataFrame):
    # ... (this function is correct)
    if ML_MODEL is None:
        return "ML model is not loaded. Please run train_model.py."
    try:
        flight_info = full_flight_df[full_flight_df['flight'] == flight_id].iloc[0]
    except IndexError:
        return f"Flight with ID '{flight_id}' not found."
    prediction_data = flight_info.copy()
    new_time = prediction_data['sched_time_local'].replace(hour=new_time_hour, minute=0, second=0)
    prediction_data['sched_hour'] = new_time.hour
    prediction_data['sched_weekday'] = new_time.weekday()
    prediction_data['sched_month'] = new_time.month
    features_df = pd.DataFrame([prediction_data])[['origin', 'destination', 'sched_hour', 'sched_weekday', 'sched_month']]
    predicted_delay = ML_MODEL.predict(features_df)[0]
    return (f"PREDICTION for flight {flight_id} at {new_time_hour}:00:\n"
            f"The XGBoost model predicts a delay of **{predicted_delay:.2f} minutes**.")

def find_top_cascading_flights(full_flight_df):
    # ... (this function is correct)
    df = full_flight_df.copy()
    df['date'] = df['sched_time_local'].dt.date
    df = df.sort_values(['tail_id', 'date', 'sched_time_local'])
    df['cascade_effect_min'] = df.groupby(['tail_id', 'date'])['delay_min'].diff().fillna(0)
    top_cascades = df[df['cascade_effect_min'] > 30].sort_values('cascade_effect_min', ascending=False)
    return top_cascades[['flight', 'tail_id', 'sched_time_local', 'origin', 'destination', 'delay_min', 'cascade_effect_min']].head(10)

def optimize_flight_schedule(flight_id: str, full_flight_df: pd.DataFrame, avg_delay_df, window_mins=90, step_mins=15):
    # ... (this function is correct)
    try:
        flight_info = full_flight_df[full_flight_df['flight'] == flight_id].iloc[0]
        original_time = flight_info['sched_time_local']
    except IndexError:
        return f"Flight with ID '{flight_id}' not found in the dataset."
    original_prediction_str = predict_delay_for_new_time(flight_id, original_time.hour, full_flight_df)
    original_delay = parse_delay_from_string(original_prediction_str)
    best_time = original_time
    best_delay = original_delay
    for shift in range(-window_mins, window_mins + step_mins, step_mins):
        if shift == 0:
            continue
        candidate_time = original_time + pd.Timedelta(minutes=shift)
        prediction_str = predict_delay_for_new_time(flight_id, candidate_time.hour, full_flight_df)
        predicted_delay = parse_delay_from_string(prediction_str)
        if predicted_delay < best_delay:
            best_delay = predicted_delay
            best_time = candidate_time
    if best_time == original_time:
        return (f"OPTIMIZATION COMPLETE for flight {flight_id}:\n"
                f"The original time at {original_time.strftime('%H:%M')} is already the optimal slot.")
    else:
        improvement = original_delay - best_delay
        return (f"OPTIMIZATION COMPLETE for flight {flight_id}:\n"
                f"**Recommendation:** Move to **{best_time.strftime('%H:%M')}**.\n"
                f"Predicted delay will improve from {original_delay:.2f} mins to {best_delay:.2f} mins (a reduction of {improvement:.2f} mins).")

# (All the functions above this are the same, this is just the end of the file)
# ... (process_flight_data, predict_delay_for_new_time, etc.) ...

def run_system_wide_optimization(full_flight_df, avg_delay_df):
    if ML_MODEL is None: return None
    print("\n--- Starting System-Wide Schedule Optimization ---")
    flight_ids_to_optimize = full_flight_df.sample(n=100, random_state=42)['flight'].unique()
    results = []
    # (Loop logic is the same)
    for flight_id in flight_ids_to_optimize:
        try:
            original_time = full_flight_df[full_flight_df['flight'] == flight_id].iloc[0]['sched_time_local']
            original_predicted_delay = parse_delay_from_string(predict_delay_for_new_time(flight_id, original_time.hour, full_flight_df))
            recommendation = optimize_flight_schedule(flight_id, full_flight_df, avg_delay_df)
            if "Recommendation:" in recommendation:
                new_predicted_delay = parse_delay_from_string(recommendation.split("mins to ")[1])
                delay_reduction = original_predicted_delay - new_predicted_delay
            else:
                new_predicted_delay, delay_reduction = original_predicted_delay, 0
            results.append({"flight_id": flight_id, "original_predicted_delay": original_predicted_delay,
                            "optimized_predicted_delay": new_predicted_delay, "delay_reduction_mins": delay_reduction})
        except: continue
    if not results: return None
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'optimization_results.csv'), index=False)
    print(f"✅ Optimization results saved to 'data/optimization_results.csv'")
    
    total_original = results_df['original_predicted_delay'].sum()
    total_optimized = results_df['optimized_predicted_delay'].sum()
    total_reduction = results_df['delay_reduction_mins'].sum()
    avg_pct = (results_df['delay_reduction_mins'] / results_df['original_predicted_delay'].replace(0, 1)).replace([np.inf, -np.inf], 0).mean() * 100
    
    cost_before = total_original * 100
    cost_after = total_optimized * 100
    cost_saved = total_reduction * 100
    
    summary = {
        "Total Delay BEFORE": f"{total_original:.2f} mins", "Total Delay AFTER": f"{total_optimized:.2f} mins",
        "Total Delay SAVED": f"{total_reduction:.2f} mins", "Average Improvement per Flight": f"{avg_pct:.2f}%",
        "Estimated Cost BEFORE": f"${cost_before:,.2f}", "Estimated Cost AFTER": f"${cost_after:,.2f}",
        "Estimated Savings": f"${cost_saved:,.2f}"
    }
    
    # --- NEW: Save summary to a JSON file ---
    import json
    with open(os.path.join(OUTPUT_DIR, 'optimization_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"✅ Optimization summary saved to 'data/optimization_summary.json'")

    print("\n--- OPTIMIZATION INSIGHTS ---")
    for key, val in summary.items(): print(f"{key}: {val}")
    print("-----------------------------")
    return summary

if __name__ == '__main__':
    full_df, avg_delay_df = process_flight_data()
    if full_df is not None:
        run_system_wide_optimization(full_df, avg_delay_df)