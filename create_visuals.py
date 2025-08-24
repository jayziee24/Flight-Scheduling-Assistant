# create_visuals.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_visualizations():
    """
    Loads the processed flight data and generates and saves
    several key EDA charts and a heatmap.
    """
    print("--- Starting Visualization Generation ---")
    
    # Define paths
    DATA_DIR = 'data'
    VISUALS_DIR = 'visuals'
    
    # Create a directory to save the charts
    os.makedirs(VISUALS_DIR, exist_ok=True)
    
    # --- Load Data ---
    try:
        # Load the raw data for the heatmap
        full_df = pd.read_csv(os.path.join(DATA_DIR, 'bom_week_flights_synthetic.csv'))
        full_df['sched_time_local'] = pd.to_datetime(full_df['sched_time_local'])
        full_df['actual_time_local'] = pd.to_datetime(full_df['actual_time_local'])
        full_df['delay_min'] = (full_df['actual_time_local'] - full_df['sched_time_local']).dt.total_seconds() / 60

        # Load the summary data for bar/line charts
        busiest_df = pd.read_csv(os.path.join(DATA_DIR, 'busiest_hours.csv'))
        avg_delay_df = pd.read_csv(os.path.join(DATA_DIR, 'avg_delay_by_hour.csv'))
        summary_df = pd.merge(busiest_df, avg_delay_df, on='hour')
        print("Successfully loaded data.")
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please run analysis.py first to generate the necessary CSV files.")
        return

    # --- Chart 1: Bar Chart of Busiest Hours ---
    plt.figure(figsize=(12, 7))
    sns.barplot(x='hour', y='ops_count', data=summary_df, palette='viridis', hue='hour', legend=False)
    plt.title('Total Flight Operations by Hour', fontsize=16)
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Number of Flights', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    barchart_path = os.path.join(VISUALS_DIR, 'busiest_hours_barchart.png')
    plt.savefig(barchart_path)
    print(f"Saved: {barchart_path}")
    plt.close()

    # --- Chart 2: Line Chart of Average Delay by Hour ---
    plt.figure(figsize=(12, 7))
    sns.lineplot(x='hour', y='delay_min', data=summary_df, marker='o', color='crimson')
    plt.title('Average Delay by Hour', fontsize=16)
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Average Delay (Minutes)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    linechart_path = os.path.join(VISUALS_DIR, 'average_delay_linechart.png')
    plt.savefig(linechart_path)
    print(f"Saved: {linechart_path}")
    plt.close()

    # --- Chart 3: The Heatmap (Delay by Day and Hour) ---
    print("Generating heatmap... this might take a moment.")
    full_df['day_of_week'] = full_df['sched_time_local'].dt.day_name()
    full_df['hour'] = full_df['sched_time_local'].dt.hour
    
    # Create a pivot table for the heatmap
    heatmap_data = full_df.pivot_table(
        values='delay_min', 
        index='day_of_week', 
        columns='hour', 
        aggfunc='mean'
    )
    # Order the days of the week correctly
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heatmap_data = heatmap_data.reindex(days_order)
    
    plt.figure(figsize=(16, 8))
    sns.heatmap(heatmap_data, cmap='YlOrRd', annot=True, fmt=".1f", linewidths=.5)
    plt.title('Heatmap of Average Delay (Minutes) by Day and Hour', fontsize=16)
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Day of the Week', fontsize=12)
    plt.tight_layout()
    heatmap_path = os.path.join(VISUALS_DIR, 'delay_heatmap.png')
    plt.savefig(heatmap_path)
    print(f"Saved: {heatmap_path}")
    plt.close()

    # --- Chart 4: Scatter Plot of Operations vs. Delay ---
    plt.figure(figsize=(10, 8))
    sns.regplot(x='ops_count', y='delay_min', data=summary_df, scatter_kws={'s':100, 'alpha':0.7})
    plt.title('Flight Congestion vs. Average Delay', fontsize=16)
    plt.xlabel('Number of Flights in Hour (Congestion)', fontsize=12)
    plt.ylabel('Average Delay (Minutes)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    scatter_path = os.path.join(VISUALS_DIR, 'congestion_vs_delay_scatter.png')
    plt.savefig(scatter_path)
    print(f"Saved: {scatter_path}")
    plt.close()

    print("\n--- All visualizations have been generated and saved in the 'visuals/' folder! ---")


# This is the "start button" for the script.
if __name__ == '__main__':
    create_visualizations()
