# CLUSTERING SCRIPT - K-MEDOIDS WITH ROAD NETWORK DISTANCES

# This script groups vehicles based on their departure time, origin coordinates, and destination coordinates
# Unlike standard K-Means (which uses pure Euclidean distance on all variables equally), this script uses a customized similarity measure
# Using K-Medoids instead of K-Means to ensure that cluster centers are actual data points (vehicles) rather than virtual averages
# Using a precomputed distance matrix based on shortest path distances in the road network instead of Euclidean distance

import pandas as pd
import random
import json

TIME_WEIGHT = 0.4 # Importance of departure time similarity
SPACE_WEIGHT = 20 # Importance of route (origin/destination) similarity
FINAL_CLUSTERS_NUM = 20 # Number of desired clusters
max_iters = 100

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ADJUST THIS PATH IF NEEDED
# Load the input data with already calculated X and Y coordinates
# df = pd.read_csv('clustering_ideas\\ingolstadt_custom_clustering\\ingolstadt_custom_agents_coords.csv')
df = pd.read_csv('clustering_ideas\\saint_arnoult_clustering\\saint_arnoult_agents_coords.csv')
# Load the precomputed distance matrix (shortest path distances between edges)
# with open('clustering_ideas\\ingolstadt_custom_clustering\\shortest_path_metric_matrix.json', 'r') as f:
#     dist_matrix = json.load(f)
with open('clustering_ideas\\saint_arnoult_clustering\\shortest_path_metric_matrix.json', 'r') as f:
    dist_matrix = json.load(f)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Normalization (min-max scaling 0 to 1) of time (coordinates are not normalized since we use the precomputed distance matrix which is normalized)
min_t, max_t = df['start_time'].min(), df['start_time'].max()
range_t = max_t - min_t if max_t != min_t else 1
df['t_norm'] = (df['start_time'] - min_t) / range_t

features = ['t_norm', 'origin_real_id', 'dest_real_id']

# K-medoids init
centroids = df[features].sample(n=FINAL_CLUSTERS_NUM, random_state=42).to_dict('records')

# Similarity measure using the precomputed distance matrix
def calculate_network_distance(agent_row, center_row):
    diff_time = abs(agent_row['t_norm'] - center_row['t_norm'])
    
    orig_dist = dist_matrix[str(agent_row['origin_real_id'])][str(center_row['origin_real_id'])]
    dest_dist = dist_matrix[str(agent_row['dest_real_id'])][str(center_row['dest_real_id'])]
    
    return (TIME_WEIGHT * diff_time) + (SPACE_WEIGHT * (orig_dist + dest_dist))

def get_mode(lst):
    return max(set(lst), key=lst.count)

for iteration in range(max_iters):
    clusters = []
    # Assign each vehicle to the "closest" medoid based on the custom similarity measure
    for index, row in df.iterrows():
        distances = [calculate_network_distance(row, center) for center in centroids]
        closest_cluster = distances.index(min(distances))
        clusters.append(closest_cluster)
        
    df['cluster'] = clusters
    
    # Recalculate the medoids as the exact mathematical mean of all vehicles in that group
    new_centroids = []
    for i in range(FINAL_CLUSTERS_NUM):
        cluster_cars = df[df['cluster'] == i]
        if not cluster_cars.empty:
            ideal_t = cluster_cars['t_norm'].mean()
            ideal_o = get_mode(cluster_cars['origin_real_id'].tolist())
            ideal_d = get_mode(cluster_cars['dest_real_id'].tolist())
            virtual_center = {'t_norm': ideal_t, 'origin_real_id': ideal_o, 'dest_real_id': ideal_d}
            
            # Find the actual car in the cluster that is closest to this center using the network distance
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
        print(f"Sukces! Klastry ustabilizowały się po {iteration + 1} iteracjach.")
        break
        
    centroids = new_centroids

columns_to_keep = ['start_time', 'origin', 'destination', 'kind', 'origin_real_id', 'dest_real_id', 'origin_x', 'origin_y', 'dest_x', 'dest_y', 'cluster']
df_final = df[columns_to_keep]

print(df_final[['start_time', 'origin', 'destination', 'cluster']].head(10))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ADJUST THIS PATH IF NEEDED
df_final.to_csv('clustering_ideas\\saint_arnoult_clustering\\agents_clustered_similarity_measure_shortest_path.csv', index=False)
# df_final.to_csv('clustering_ideas\\ingolstadt_custom_clustering\\agents_clustered_similarity_measure_shortest_path.csv', index=False)
print("Gotowe! Zapisano wyniki do agents_clustered_similarity_measure_shortest_path.csv")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~