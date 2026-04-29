# CLUSTERING SCRIPT - K-MEDOIDS WITH ROAD NETWORK DISTANCES & ELBOW PLOT

# This script groups vehicles based on their departure time, origin coordinates, and destination coordinates
# Using K-Medoids instead of K-Means to ensure that cluster centers are actual data points (vehicles) rather than virtual averages
# Using a precomputed distance matrix based on shortest path distances in the road network instead of Euclidean distance

import pandas as pd
import random
import json
import matplotlib.pyplot as plt

TIME_WEIGHT = 0.2 # Importance of departure time similarity
SPACE_WEIGHT = 0.8 # Importance of route (origin/destination) similarity
max_iters = 50

MIN_K = 2
MAX_K = 40
STEP_K = 1

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ADJUST THIS PATH IF NEEDED
df = pd.read_csv('clustering_ideas\\saint_arnoult_clustering\\saint_arnoult_agents_coords.csv')

# Load the precomputed distance matrix (shortest path distances between edges)
with open('clustering_ideas\\saint_arnoult_clustering\\shortest_path_metric_matrix.json', 'r') as f:
    dist_matrix = json.load(f)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Normalization (min-max scaling 0 to 1) of time
min_t, max_t = df['start_time'].min(), df['start_time'].max()
range_t = max_t - min_t if max_t != min_t else 1
df['t_norm'] = (df['start_time'] - min_t) / range_t

features = ['t_norm', 'origin_real_id', 'dest_real_id']

# Similarity measure using the precomputed distance matrix
def calculate_network_distance(agent_row, center_row):
    diff_time = abs(agent_row['t_norm'] - center_row['t_norm'])
    
    orig_dist = dist_matrix[str(agent_row['origin_real_id'])][str(center_row['origin_real_id'])]
    dest_dist = dist_matrix[str(agent_row['dest_real_id'])][str(center_row['dest_real_id'])]
    
    return (TIME_WEIGHT * diff_time) + (SPACE_WEIGHT * (orig_dist + dest_dist))

def get_mode(lst):
    return max(set(lst), key=lst.count)

k_values = []
mae_values = []
mse_values = []

for current_k in range(MIN_K, MAX_K + 1, STEP_K):
    print(f"\nTestowanie K = {current_k}...")
    
    # K-medoids init dla obecnego K
    centroids = df[features].sample(n=current_k, random_state=42).to_dict('records')

    for iteration in range(max_iters):
        clusters = []
        # Assign each vehicle to the "closest" medoid
        for index, row in df.iterrows():
            distances = [calculate_network_distance(row, center) for center in centroids]
            closest_cluster = distances.index(min(distances))
            clusters.append(closest_cluster)
            
        df['cluster'] = clusters
        
        # Recalculate the medoids
        new_centroids = []
        for i in range(current_k):
            cluster_cars = df[df['cluster'] == i]
            if not cluster_cars.empty:
                ideal_t = cluster_cars['t_norm'].mean()
                ideal_o = get_mode(cluster_cars['origin_real_id'].tolist())
                ideal_d = get_mode(cluster_cars['dest_real_id'].tolist())
                virtual_center = {'t_norm': ideal_t, 'origin_real_id': ideal_o, 'dest_real_id': ideal_d}
                
                best_car = None
                min_dist = float('inf')
                
                for _, car_row in cluster_cars.iterrows():
                    d = calculate_network_distance(car_row, virtual_center)
                    if d < min_dist:
                        min_dist = d
                        best_car = car_row.to_dict()
                        
                new_centroids.append({'t_norm': best_car['t_norm'], 
                                      'origin_real_id': best_car['origin_real_id'], 
                                      'dest_real_id': best_car['dest_real_id']})
            else:
                new_centroids.append(centroids[i])
                
        if centroids == new_centroids:
            break
            
        centroids = new_centroids

    # --- OBLICZANIE MAE i MSE (Wariancji) ---
    total_error = 0
    total_squared_error = 0
    N = len(df)
    
    for index, row in df.iterrows():
        assigned_center = centroids[int(row['cluster'])]
        d = calculate_network_distance(row, assigned_center)
        
        total_error += d
        total_squared_error += (d ** 2)
        
    mae = total_error / N
    mse = total_squared_error / N
    
    k_values.append(current_k)
    mae_values.append(mae)
    mse_values.append(mse)
    
    print(f"K={current_k} zakończone | MAE: {mae:.2f} | MSE (Wariancja): {mse:.2f}")

# Wykres
plt.figure(figsize=(10, 6))
plt.plot(k_values, mse_values, marker='o', linestyle='-', color='red', linewidth=2)
plt.title(f'MSE (K-Prototypes) Time_WEIGHT = {TIME_WEIGHT}, SPACE_WEIGHT = {SPACE_WEIGHT}')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Mean Squared Error (MSE)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(k_values)

plot_path = 'clustering_ideas\\saint_arnoult_clustering\\mse_similarity_measure_plot_shortest_path.png'
plt.savefig(plot_path)
print(f"Zapisano wykres do: {plot_path}")
plt.show()