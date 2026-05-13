"""
K-PROTOTYPES TRADE-OFF WEIGHTS EVALUATION SCRIPT
"""
import pandas as pd
import random
import json
import numpy as np
import matplotlib.pyplot as plt

NUM_OF_ZONES = 5
FINAL_CLUSTERS_NUM = 8
MAX_ITERS = 20

TIME_WEIGHTS = np.linspace(0.0, 1.0, 11)
SPACE_WEIGHTS = 1.0 - TIME_WEIGHTS

CSV_PATH = 'clustering_ideas\\ingolstadt_custom_clustering\\ingolstadt_custom_agents_coords.csv'
JSON_PATH = 'clustering_ideas\\ingolstadt_custom_clustering\\shortest_path_metric_matrix.json'
PLOT_OUTPUT = 'clustering_ideas\\ingolstadt_custom_clustering\\weights_tradeoff_plot_k_prototypes_shortest_path.png'

df = pd.read_csv(CSV_PATH)

with open(JSON_PATH, 'r') as f:
    dist_matrix = json.load(f)

def simple_network_kmedoids(edge_list, K=NUM_OF_ZONES, iters=30):
    unique_edges = list(set(edge_list))
    unique_edges = [str(e) for e in unique_edges if pd.notna(e)]
    medoids = random.sample(unique_edges, K)
    
    for _ in range(iters):
        clusters = []
        for edge in edge_list:
            edge_str = str(edge)
            dists = [dist_matrix[edge_str][m] for m in medoids]
            clusters.append(dists.index(min(dists)))
        
        new_medoids = []
        for i in range(K):
            edges_in_cluster = [str(edge_list[j]) for j in range(len(edge_list)) if clusters[j] == i]
            unique_in_cluster = list(set(edges_in_cluster))
            
            if unique_in_cluster:
                best_medoid = None
                min_total_dist = float('inf')
                for candidate in unique_in_cluster:
                    total_dist = sum(dist_matrix[candidate][other] for other in unique_in_cluster)
                    if total_dist < min_total_dist:
                        min_total_dist = total_dist
                        best_medoid = candidate
                new_medoids.append(best_medoid)
            else:
                new_medoids.append(medoids[i])
                
        if medoids == new_medoids: 
            break
        medoids = new_medoids
    return clusters

df['orig_cluster'] = simple_network_kmedoids(df['origin_real_id'].tolist(), K=NUM_OF_ZONES)
df['dest_cluster'] = simple_network_kmedoids(df['dest_real_id'].tolist(), K=NUM_OF_ZONES)

min_t, max_t = df['start_time'].min(), df['start_time'].max()
range_t = max_t - min_t if max_t != min_t else 1
df['t_norm'] = (df['start_time'] - min_t) / range_t

features = ['t_norm', 'orig_cluster', 'dest_cluster']

def calculate_similarity(agent, proto, tw, sw):
    diff_time = abs(agent[0] - proto[0])
    penalty = 0
    if agent[1] != proto[1]: penalty += 1
    if agent[2] != proto[2]: penalty += 1
    return (tw * diff_time) + (sw * penalty)

def get_mode(lst):
    return max(set(lst), key=lst.count)

spatial_penalty_mse = []
temporal_mse = []
total_runs = len(TIME_WEIGHTS)

for i in range(total_runs):
    current_tw = TIME_WEIGHTS[i]
    current_sw = SPACE_WEIGHTS[i]
    
    print(f"[{i+1}/{total_runs}] Time weight: {current_tw:.1f} | Space weight: {current_sw:.1f}")
    
    prototypes = df[features].sample(n=FINAL_CLUSTERS_NUM, random_state=42).values.tolist()
    
    for iteration in range(MAX_ITERS):
        clusters = []
        for index, row in df[features].iterrows():
            agent = row.tolist()
            dists = [calculate_similarity(agent, p, current_tw, current_sw) for p in prototypes]
            closest_cluster = dists.index(min(dists))
            clusters.append(closest_cluster)
            
        df['final_cluster'] = clusters
        
        new_prototypes = []
        for c_id in range(FINAL_CLUSTERS_NUM):
            cluster_cars = df[df['final_cluster'] == c_id]
            if not cluster_cars.empty:
                avg_t = cluster_cars['t_norm'].mean()
                mode_orig = get_mode(cluster_cars['orig_cluster'].tolist())
                mode_dest = get_mode(cluster_cars['dest_cluster'].tolist())
                new_prototypes.append([avg_t, mode_orig, mode_dest])
            else:
                new_prototypes.append(prototypes[c_id])
                
        if prototypes == new_prototypes:
            break
        prototypes = new_prototypes

    total_penalty_sq_err = 0
    total_time_sq_err = 0
    N = len(df)
    
    for index, row in df.iterrows():
        agent = [row['t_norm'], row['orig_cluster'], row['dest_cluster']]
        proto = prototypes[int(row['final_cluster'])]
        
        penalty = 0
        if agent[1] != proto[1]: penalty += 1
        if agent[2] != proto[2]: penalty += 1
        
        time_diff = abs(agent[0] - proto[0])
        
        total_penalty_sq_err += (penalty ** 2)
        total_time_sq_err += (time_diff ** 2)
        
    spatial_penalty_mse.append(total_penalty_sq_err / N)
    temporal_mse.append(total_time_sq_err / N)

fig, ax1 = plt.subplots(figsize=(12, 6))

x_labels = [f"{tw:.1f} : {sw:.1f}" for tw, sw in zip(TIME_WEIGHTS, SPACE_WEIGHTS)]
x_positions = np.arange(len(TIME_WEIGHTS))

color1 = 'tab:blue'
ax1.set_xlabel('Time : Space weight', fontweight='bold')
ax1.set_ylabel('Spatial Penalty MSE (Square of city districts difference)', color=color1, fontweight='bold')
ax1.plot(x_positions, spatial_penalty_mse, marker='o', color=color1, linewidth=2.5, label="Spatial MSE")
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_xticks(x_positions)
ax1.set_xticklabels(x_labels, rotation=45, ha='right')

ax2 = ax1.twinx()  
color2 = 'tab:orange'
ax2.set_ylabel('Pure Temporal MSE (Square of time difference)', color=color2, fontweight='bold')
ax2.plot(x_positions, temporal_mse, marker='s', color=color2, linewidth=2.5, linestyle='--', label="Temporal MSE")
ax2.tick_params(axis='y', labelcolor=color2)

plt.title(f'Trade-off Curve - K-Prototypes (K={FINAL_CLUSTERS_NUM})', fontsize=14, fontweight='bold')
fig.tight_layout()

plt.savefig(PLOT_OUTPUT)
print(f"Zapisano wykres do: {PLOT_OUTPUT}")
plt.show()