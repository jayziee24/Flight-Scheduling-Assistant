import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import networkx as nx

def create_visualizations():
    print("--- Starting Visualization Generation (Final Version) ---")
    
    DATA_DIR = 'data'
    VISUALS_DIR = 'visuals'
    os.makedirs(VISUALS_DIR, exist_ok=True)
    
    try:
        full_df = pd.read_csv(os.path.join(DATA_DIR, 'bom_week_flights_synthetic.csv'))
        full_df['sched_time_local'] = pd.to_datetime(full_df['sched_time_local'])
        full_df['actual_time_local'] = pd.to_datetime(full_df['actual_time_local'])
        full_df['delay_min'] = (full_df['actual_time_local'] - full_df['sched_time_local']).dt.total_seconds() / 60
        summary_df = pd.read_csv(os.path.join(DATA_DIR, 'busiest_hours.csv')).merge(
            pd.read_csv(os.path.join(DATA_DIR, 'avg_delay_by_hour.csv')), on='hour'
        )
        print("Successfully loaded base data.")
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return

    # Chart 1: Busiest Hours
    plt.figure(figsize=(12, 7))
    sns.barplot(x='hour', y='ops_count', data=summary_df, palette='viridis', hue='hour', legend=False)
    plt.title('Total Flight Operations by Hour', fontsize=16)
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Number of Flights', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, 'busiest_hours_barchart.png'))
    print(f"Saved: busiest_hours_barchart.png")
    plt.close()

    # Chart 2: Average Delay
    plt.figure(figsize=(12, 7))
    sns.lineplot(x='hour', y='delay_min', data=summary_df, marker='o', color='crimson')
    plt.title('Average Delay by Hour', fontsize=16)
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Average Delay (Minutes)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, 'average_delay_linechart.png'))
    print(f"Saved: average_delay_linechart.png")
    plt.close()

    # Chart 3: Heatmap
    full_df['day_of_week'] = full_df['sched_time_local'].dt.day_name()
    full_df['hour'] = full_df['sched_time_local'].dt.hour
    heatmap_data = full_df.pivot_table(values='delay_min', index='day_of_week', columns='hour', aggfunc='mean')
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heatmap_data = heatmap_data.reindex(days_order)
    plt.figure(figsize=(16, 8))
    sns.heatmap(heatmap_data, cmap='YlOrRd', annot=True, fmt=".1f", linewidths=.5)
    plt.title('Heatmap of Average Delay (Minutes) by Day and Hour', fontsize=16)
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Day of the Week', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, 'delay_heatmap.png'))
    print(f"Saved: delay_heatmap.png")
    plt.close()

    # Chart 4: Scatter Plot
    plt.figure(figsize=(10, 8))
    sns.regplot(x='ops_count', y='delay_min', data=summary_df, scatter_kws={'s':100, 'alpha':0.7})
    plt.title('Flight Congestion vs. Average Delay', fontsize=16)
    plt.xlabel('Number of Flights in Hour (Congestion)', fontsize=12)
    plt.ylabel('Average Delay (Minutes)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, 'congestion_vs_delay_scatter.png'))
    print(f"Saved: congestion_vs_delay_scatter.png")
    plt.close()

    # Chart 5: Optimization Improvements
    try:
        results_df = pd.read_csv(os.path.join(DATA_DIR, 'optimization_results.csv'))
        top_10_improvements = results_df.sort_values('delay_reduction_mins', ascending=False).head(10)
        plt.figure(figsize=(12, 8))
        sns.barplot(x='delay_reduction_mins', y='flight_id', data=top_10_improvements, palette='summer', hue='flight_id', legend=False)
        plt.title('Top 10 Flights by Predicted Delay Reduction', fontsize=16)
        plt.xlabel('Predicted Delay Saved (Minutes)', fontsize=12)
        plt.ylabel('Flight ID', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALS_DIR, 'optimization_improvements.png'))
        print(f"Saved: optimization_improvements.png")
        plt.close()
    except FileNotFoundError:
        print("Skipping optimization visuals: 'optimization_results.csv' not found.")

    # Chart 6: Cascading Delay Network Graph
    print("Generating advanced cascade network graph... this may take a moment.")
    try:
        df_sorted = full_df.sort_values(['tail_id', 'sched_time_local'])
        df_sorted['cascade_effect_min'] = df_sorted.groupby('tail_id')['delay_min'].diff()
        
        G = nx.DiGraph()
        for _, row in df_sorted.iterrows():
            G.add_node(row['flight'], tail_id=row['tail_id'])

        for tail_id, group in df_sorted.groupby('tail_id'):
            for i in range(len(group) - 1):
                flight1 = group.iloc[i]
                flight2 = group.iloc[i+1]
                # Only add an edge if there's a positive cascade effect
                cascade_weight = flight2['cascade_effect_min']
                if pd.notna(cascade_weight) and cascade_weight > 0:
                    G.add_edge(flight1['flight'], flight2['flight'], weight=cascade_weight)

        centrality = nx.betweenness_centrality(G, weight='weight', normalized=True)
        top_nodes = sorted(centrality, key=centrality.get, reverse=True)[:15]

        nodes_to_include = set(top_nodes)
        for node in top_nodes:
            for pred in G.predecessors(node):
                nodes_to_include.add(pred)
            for succ in G.successors(node):
                nodes_to_include.add(succ)
        H = G.subgraph(nodes_to_include)

        plt.figure(figsize=(22, 18))
        pos = nx.spring_layout(H, k=0.9, iterations=50, seed=42)
        
        node_sizes = [centrality.get(node, 0) * 40000 + 800 for node in H.nodes()]
        edge_weights = [H[u][v].get('weight', 0) for u, v in H.edges()]
        
        nx.draw_networkx_edges(H, pos, width=[w/20 for w in edge_weights], edge_color='red', alpha=0.6)
        nx.draw_networkx_nodes(H, pos, node_size=node_sizes, node_color='skyblue')
        nx.draw_networkx_labels(H, pos, font_size=10)
        
        plt.title('Cascading Delay Network (Top 15 Most Influential Flights)', fontsize=24)
        plt.box(False)
        graph_path = os.path.join(VISUALS_DIR, 'cascade_network_graph.png')
        plt.savefig(graph_path)
        print(f"Saved: {graph_path}")
        plt.close()
    except Exception as e:
        print(f"Could not generate network graph. Error: {e}")

    print("\n--- All visualizations have been generated! ---")

if __name__ == '__main__':
    create_visualizations()