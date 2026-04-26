# K-PROTOTYPES CLUSTERING SCRIPT

# This script groups vehicles based on their departure time and physical route
# It operates in two stages to handle different data types (time is continuous, map zone is categorical)

import pandas as pd
import random
import math

NUM_OF_ZONES = 5
FINAL_CLUSTERS_NUM = 20 # Target number of final clusters
TIME_WEIGHT = 1.0 # How heavily the algorithm penalizes for differences in departure time
SPACE_WEIGHT = 8.0 # How heavily the algorithm penalizes for departing from/arriving at a different zone
max_iters = 50

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ADJUST THIS PATH IF NEEDED
# Load the input data with already calculated X and Y coordinates
# df = pd.read_csv('clustering_ideas\\ingolstadt_custom_clustering\\ingolstadt_custom_agents_coords.csv')
df = pd.read_csv('clustering_ideas\\provins_clustering\\provins_agents_coords.csv')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print("Clustering origin and destination points ({NUM_OF_ZONES} zones each)")

# STAGE 1 (Spatial Clustering - K-Means):
# Ignores time, analyzes only the X and Y coordinates of the vehicles 
# Divides the city into K zones (e.g., 5) for origin points and K zones for destination points 
# Instead of exact coordinates, vehicles receive zone labels (e.g., "departs from zone 2, goes to zone 4")
def simple_spatial_kmeans(coords_list, K=NUM_OF_ZONES, iters=30):
    centroids = random.sample(coords_list, K)
    
    for _ in range(iters):
        clusters = []
        # Assign each point to the nearest center
        for pt in coords_list:
            dists = [math.sqrt((pt[0]-c[0])**2 + (pt[1]-c[1])**2) for c in centroids]
            clusters.append(dists.index(min(dists)))
        
        # Move zone centers to the average position of the assigned points
        new_centroids = []
        for i in range(K):
            pts_in_cluster = [coords_list[j] for j in range(len(coords_list)) if clusters[j] == i]
            if pts_in_cluster:
                avg_x = sum(p[0] for p in pts_in_cluster) / len(pts_in_cluster)
                avg_y = sum(p[1] for p in pts_in_cluster) / len(pts_in_cluster)
                new_centroids.append((avg_x, avg_y))
            else:
                new_centroids.append(centroids[i])
                
        if centroids == new_centroids: 
            break
        centroids = new_centroids
        
    return clusters

# Assing clusters for origins and destinations
orig_coords = list(zip(df['origin_x'], df['origin_y']))
df['orig_cluster'] = simple_spatial_kmeans(orig_coords, K=NUM_OF_ZONES)

dest_coords = list(zip(df['dest_x'], df['dest_y']))
df['dest_cluster'] = simple_spatial_kmeans(dest_coords, K=NUM_OF_ZONES)

print("Spatial clustering completed. Sample of assigned zones:")
print(df[['orig_cluster', 'dest_cluster']].head(10))

# Time normalization (0 to 1) (min-max scaling)
min_t, max_t = df['start_time'].min(), df['start_time'].max()
range_t = max_t - min_t if max_t != min_t else 1
df['t_norm'] = (df['start_time'] - min_t) / range_t

# STAGE 2 (Main Clustering - K-Prototypes)
# Combines time (continuous variable) and zones (categorical variables)
# Calculates a "similarity measure":
# - If vehicles depart at different times, it adds the time difference
# - If vehicles depart from different zones or go to different zones, it adds a fixed "penalty" (0, 1 or 2)
# Finally, it divides vehicles into final clusters, calculating new centers (prototypes) as the average for time and the most frequent value for zones

# Prototype stores: [normalized_time, origin_zone, dest_zone]
features = ['t_norm', 'orig_cluster', 'dest_cluster']
prototypes = df[features].sample(n=FINAL_CLUSTERS_NUM, random_state=42).values.tolist()

def calculate_similarity(agent, proto):
    diff_time = abs(agent[0] - proto[0])
    penalty = 0
    if agent[1] != proto[1]: 
        penalty += 1
    if agent[2] != proto[2]: 
        penalty += 1
        
    return (TIME_WEIGHT * diff_time) + (SPACE_WEIGHT * penalty)

def get_mode(lst):
    return max(set(lst), key=lst.count)

# Main K-Prototypes loop
for iteration in range(max_iters):
    clusters = []
    
    # Assign each vehicle to the prototype it is "closest" to (lowest penalty score)
    for index, row in df[features].iterrows():
        agent = row.tolist()
        dists = [calculate_similarity(agent, p) for p in prototypes]
        closest_cluster = dists.index(min(dists))
        clusters.append(closest_cluster)
    df['final_cluster'] = clusters
    
    # Update prototypes - for time we take the average, for zones we take the most common value among assigned vehicles
    new_prototypes = []
    for i in range(FINAL_CLUSTERS_NUM):
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

columns_to_keep = ['start_time', 'origin', 'destination', 'kind', 'origin_real_id', 'dest_real_id', 'origin_x', 'origin_y', 'dest_x', 'dest_y', 'final_cluster']
df_final = df[columns_to_keep]
df_final = df_final.rename(columns={'final_cluster': 'cluster'})
print(df_final.head(15))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ADJUST THIS PATH IF NEEDED
# df_final.to_csv('clustering_ideas\\ingolstadt_custom_clustering\\agents_clustered_k_prototypes.csv', index=False)
df_final.to_csv('clustering_ideas\\provins_clustering\\agents_clustered_k_prototypes.csv', index=False)

print("\nSaved to agents_clustered_k_prototypes.csv")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~