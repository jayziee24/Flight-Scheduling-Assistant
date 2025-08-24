import pandas as pd
import os

# Define file paths
RAW_DATA_PATH = 'data/bom_week_flights_synthetic.csv'
OUTPUT_DIR = 'data'

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_flight_data():
    """
    Loads raw flight data, calculates delays and insights, 
    and saves them as new CSV files.
    """
    print("Starting data processing...")
    
    # 1. Load the raw data
    try:
        df = pd.read_csv(RAW_DATA_PATH)
        print(f"Loaded raw data: {df.shape[0]} flights.")
    except FileNotFoundError:
        print(f"ERROR: Raw data file not found at {RAW_DATA_PATH}")
        return None, None

    # 2. Data Cleaning and Feature Engineering
    df['sched_time_local'] = pd.to_datetime(df['sched_time_local'])
    df['actual_time_local'] = pd.to_datetime(df['actual_time_local'])
    df['delay_min'] = (df['actual_time_local'] - df['sched_time_local']).dt.total_seconds() / 60
    df['hour'] = df['sched_time_local'].dt.hour
    print("Calculated delays and extracted hour of day.")

    # 3. Perform Analyses and Save Results
    busiest_hours = df.groupby('hour').size().reset_index(name='ops_count')
    busiest_hours = busiest_hours.sort_values('ops_count', ascending=False)
    busiest_hours.to_csv(os.path.join(OUTPUT_DIR, 'busiest_hours.csv'), index=False)
    print("Generated busiest_hours.csv")

    avg_delay_by_hour = df.groupby('hour')['delay_min'].mean().reset_index()
    avg_delay_by_hour.to_csv(os.path.join(OUTPUT_DIR, 'avg_delay_by_hour.csv'), index=False)
    print("Generated avg_delay_by_hour.csv")

    best_hours = avg_delay_by_hour.sort_values('delay_min').head(5)
    best_hours.to_csv(os.path.join(OUTPUT_DIR, 'best_hours.csv'), index=False)
    print("Generated best_hours.csv")

    print("\nData processing complete. All analytical files saved in 'data/' directory.")
    
    return df, avg_delay_by_hour

def predict_delay_for_new_time(new_hour, avg_delay_df):
    """
    'Models' the impact of rescheduling a flight by looking up the average delay.
    """
    if not 0 <= new_hour <= 23:
        return "Invalid hour. Please provide an hour from 0 to 23."
    delay_info = avg_delay_df[avg_delay_df['hour'] == new_hour]
    if delay_info.empty:
        return f"No historical delay data found for hour {new_hour}."
    avg_delay = round(delay_info['delay_min'].iloc[0], 2)
    return (f"PREDICTION: A flight scheduled at hour {new_hour} has an expected average delay "
            f"of {avg_delay} minutes based on historical data.")

def find_top_cascading_flights(full_flight_df):
    """
    'Models' cascading delays by finding flights where the delay significantly
    increased from the previous flight operated by the same aircraft on the same day.
    """
    df = full_flight_df.copy()
    df['date'] = df['sched_time_local'].dt.date
    df = df.sort_values(['tail_id', 'date', 'sched_time_local'])
    df['cascade_effect_min'] = df.groupby(['tail_id', 'date'])['delay_min'].diff().fillna(0)
    top_cascades = df[df['cascade_effect_min'] > 30].sort_values('cascade_effect_min', ascending=False)
    print("Found top cascading flights.")
    return top_cascades[['flight', 'tail_id', 'sched_time_local', 'origin', 'destination', 'delay_min', 'cascade_effect_min']].head(10)

# This block is the "start button". If it's missing, the script does nothing.
if __name__ == '__main__':
    process_flight_data()