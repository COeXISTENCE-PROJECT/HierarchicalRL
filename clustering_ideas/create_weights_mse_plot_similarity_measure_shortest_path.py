# TRADE-OFF WEIGHTS EVALUATION SCRIPT
import pandas as pd
import random
import json
import numpy as np
import matplotlib.pyplot as plt

K_TARGET = 19
MAX_ITERS = 20

TIME_WEIGHTS = np.linspace(0.0, 1.0, 11)
SPACE_WEIGHTS = 1.0 - TIME_WEIGHTS

CSV_PATH = 'clustering_ideas\\saint_arnoult_clustering\\saint_arnoult_agents_coords.csv'
JSON_PATH = 'clustering_ideas\\saint_arnoult_clustering\\shortest_path_metric_matrix.json'
PLOT_OUTPUT = 'clustering_ideas\\saint_arnoult_clustering\\weights_tradeoff_plot_similarity_measure_shortest_path.png'

df = pd.read_csv(CSV_PATH)

with open(JSON_PATH, 'r') as f:
    dist_matrix = json.load(f)

min_t, max_t = df['start_time'].min(), df['start_time'].max()
range_t = max_t - min_t if max_t != min_t else 1
df['t_norm'] = (df['start_time'] - min_t) / range_t

features = ['t_norm', 'origin_real_id', 'dest_real_id']

def calculate_network_distance(agent_row, center_row, tw, sw):
    diff_time = abs(agent_row['t_norm'] - center_row['t_norm'])
    orig_dist = dist_matrix[str(agent_row['origin_real_id'])][str(center_row['origin_real_id'])]
    dest_dist = dist_matrix[str(agent_row['dest_real_id'])][str(center_row['dest_real_id'])]
    return (tw * diff_time) + (sw * (orig_dist + dest_dist))

def get_pure_spatial_error(agent_row, center_row):
    orig_dist = dist_matrix[str(agent_row['origin_real_id'])][str(center_row['origin_real_id'])]
    dest_dist = dist_matrix[str(agent_row['dest_real_id'])][str(center_row['dest_real_id'])]
    return orig_dist + dest_dist

def get_pure_temporal_error(agent_row, center_row):
    return abs(agent_row['t_norm'] - center_row['t_norm'])

def get_mode(lst):
    return max(set(lst), key=lst.count)

spatial_mse_results = []
temporal_mse_results = []
total_runs = len(TIME_WEIGHTS)


for i in range(total_runs):
    current_tw = TIME_WEIGHTS[i]
    current_sw = SPACE_WEIGHTS[i]
    
    print(f"[{i+1}/{total_runs}] Time_weight: {current_tw:.1f} | Space_weight: {current_sw:.1f}")
    
    centroids = df[features].sample(n=K_TARGET, random_state=42).to_dict('records')
    
    for iteration in range(MAX_ITERS):
        clusters = []
        for index, row in df.iterrows():
            distances = [calculate_network_distance(row, center, current_tw, current_sw) for center in centroids]
            closest_cluster = distances.index(min(distances))
            clusters.append(closest_cluster)
            
        df['cluster'] = clusters
        
        new_centroids = []
        for c_id in range(K_TARGET):
            cluster_cars = df[df['cluster'] == c_id]
            if not cluster_cars.empty:
                ideal_t = cluster_cars['t_norm'].mean()
                ideal_o = get_mode(cluster_cars['origin_real_id'].tolist())
                ideal_d = get_mode(cluster_cars['dest_real_id'].tolist())
                virtual_center = {'t_norm': ideal_t, 'origin_real_id': ideal_o, 'dest_real_id': ideal_d}
                
                best_car = None
                min_dist = float('inf')
                for _, car_row in cluster_cars.iterrows():
                    d = calculate_network_distance(car_row, virtual_center, current_tw, current_sw)
                    if d < min_dist:
                        min_dist = d
                        best_car = car_row.to_dict()
                        
                new_centroids.append({'t_norm': best_car['t_norm'], 
                                      'origin_real_id': best_car['origin_real_id'], 
                                      'dest_real_id': best_car['dest_real_id']})
            else:
                new_centroids.append(centroids[c_id])
                
        if centroids == new_centroids:
            break
        centroids = new_centroids

    total_spatial_sq_err = 0
    total_temporal_sq_err = 0
    N = len(df)
    
    for index, row in df.iterrows():
        assigned_center = centroids[int(row['cluster'])]
        
        s_err = get_pure_spatial_error(row, assigned_center)
        t_err = get_pure_temporal_error(row, assigned_center)
        
        total_spatial_sq_err += (s_err ** 2)
        total_temporal_sq_err += (t_err ** 2)
        
    spatial_mse_results.append(total_spatial_sq_err / N)
    temporal_mse_results.append(total_temporal_sq_err / N)


fig, ax1 = plt.subplots(figsize=(12, 6))

x_labels = [f"{tw:.1f} : {sw:.1f}" for tw, sw in zip(TIME_WEIGHTS, SPACE_WEIGHTS)]
x_positions = np.arange(len(TIME_WEIGHTS))

color1 = 'tab:blue'
ax1.set_xlabel('Time : Space weight', fontweight='bold')
ax1.set_ylabel('MSE Spatial', color=color1, fontweight='bold')
ax1.plot(x_positions, spatial_mse_results, marker='o', color=color1, linewidth=2.5, label="Spatial MSE")
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_xticks(x_positions)
ax1.set_xticklabels(x_labels, rotation=45, ha='right')

ax2 = ax1.twinx()  
color2 = 'tab:orange'
ax2.set_ylabel('MSE Time', color=color2, fontweight='bold')
ax2.plot(x_positions, temporal_mse_results, marker='s', color=color2, linewidth=2.5, linestyle='--', label="Temporal MSE")
ax2.tick_params(axis='y', labelcolor=color2)

plt.title(f'Weights Vs. MSE for time and space (K={K_TARGET})', fontsize=14, fontweight='bold')
fig.tight_layout()

plt.savefig(PLOT_OUTPUT)
print(f"Saved to: {PLOT_OUTPUT}")
plt.show()