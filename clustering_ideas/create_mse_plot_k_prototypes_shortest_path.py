# K-PROTOTYPES CLUSTERING SCRIPT WITH ROAD NETWORK DISTANCES & ELBOW PLOT

import pandas as pd
import random
import json
import matplotlib.pyplot as plt

NUM_OF_ZONES = 5
TIME_WEIGHT = 0.1 
SPACE_WEIGHT = 0.9 
max_iters = 50 

MIN_K = 2
MAX_K = 40
STEP_K = 1

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
df = pd.read_csv('clustering_ideas\\provins_clustering\\provins_agents_coords.csv')
with open('clustering_ideas\\provins_clustering\\shortest_path_metric_matrix.json', 'r') as f:
    dist_matrix = json.load(f)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print(f"Clustering origin and destination points ({NUM_OF_ZONES} zones each)")

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

def calculate_similarity(agent, proto):
    diff_time = abs(agent[0] - proto[0])
    penalty = 0
    if agent[1] != proto[1]: penalty += 1
    if agent[2] != proto[2]: penalty += 1
    return (TIME_WEIGHT * diff_time) + (SPACE_WEIGHT * penalty)

def get_mode(lst):
    return max(set(lst), key=lst.count)

k_values = []
cost_values = []

for current_k in range(MIN_K, MAX_K + 1, STEP_K):
    print(f"Testowanie K = {current_k}...")
    prototypes = df[features].sample(n=current_k, random_state=42).values.tolist()
    
    for iteration in range(max_iters):
        clusters = []
        for index, row in df[features].iterrows():
            agent = row.tolist()
            dists = [calculate_similarity(agent, p) for p in prototypes]
            closest_cluster = dists.index(min(dists))
            clusters.append(closest_cluster)
        df['final_cluster'] = clusters
        
        new_prototypes = []
        for i in range(current_k):
            cluster_cars = df[df['final_cluster'] == i]
            if not cluster_cars.empty:
                avg_t = cluster_cars['t_norm'].mean()
                mode_orig = get_mode(cluster_cars['orig_cluster'].tolist())
                mode_dest = get_mode(cluster_cars['dest_cluster'].tolist())
                new_prototypes.append([avg_t, mode_orig, mode_dest])
            else:
                new_prototypes.append(prototypes[i])
                
        if prototypes == new_prototypes:
            break
        prototypes = new_prototypes

    # --- OBLICZANIE MAE i MSE (Wariancji) DLA TEGO K ---
    total_error = 0
    total_squared_error = 0
    N = len(df)
    
    for index, row in df.iterrows():
        assigned_center = prototypes[int(row['final_cluster'])]
        
        agent = [row['t_norm'], row['orig_cluster'], row['dest_cluster']]
        
        d = calculate_similarity(agent, assigned_center)
        
        total_error += d
        total_squared_error += (d ** 2) 

    mae = total_error / N
    mse = total_squared_error / N
    
    k_values.append(current_k)
    cost_values.append(mse) 
    
    print(f"K={current_k} | MAE: {mae:.2f} | MSE (Wariancja): {mse:.2f}")

plt.figure(figsize=(10, 6))
plt.plot(k_values, cost_values, marker='o', linestyle='-', color='red', linewidth=2)
plt.title(f'MSE (K-Prototypes) Time_WEIGHT = {TIME_WEIGHT}, SPACE_WEIGHT = {SPACE_WEIGHT}')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Mean Squared Error (MSE)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(k_values)

plot_path = 'clustering_ideas\\provins_clustering\\mse_kprototypes_plot_shortest_path.png'
plt.savefig(plot_path)
print(f"Zapisano wykres do: {plot_path}")
plt.show()
