# CLUSTERING SCRIPT - MODIFIED K-MEANS

# This script groups vehicles based on their departure time, origin coordinates, and destination coordinates
# Unlike standard K-Means (which uses pure Euclidean distance on all variables equally), this script uses a customized similarity measure

import pandas as pd
import random
import math

TIME_WEIGHT = 0.4 # Importance of departure time similarity
SPACE_WEIGHT = 20.0 # Importance of route (origin/destination) similarity
FINAL_CLUSTERS_NUM = 20 # Number of desired clusters
max_iters = 100

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ADJUST THIS PATH IF NEEDED
# Load the input data with already calculated X and Y coordinates
# df = pd.read_csv('clustering_ideas\\ingolstadt_custom_clustering\\ingolstadt_custom_agents_coords.csv')
df = pd.read_csv('clustering_ideas\\saint_arnoult_clustering\\saint_arnoult_agents_coords.csv')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Normalization (min-max scaling 0 to 1) of time and coordinates
min_t, max_t = df['start_time'].min(), df['start_time'].max()
range_t = max_t - min_t if max_t != min_t else 1
df['t_norm'] = (df['start_time'] - min_t) / range_t

min_coord = min(df['origin_x'].min(), df['origin_y'].min(), df['dest_x'].min(), df['dest_y'].min())
max_coord = max(df['origin_x'].max(), df['origin_y'].max(), df['dest_x'].max(), df['dest_y'].max())
range_coord = max_coord - min_coord if max_coord != min_coord else 1

df['ox_norm'] = (df['origin_x'] - min_coord) / range_coord
df['oy_norm'] = (df['origin_y'] - min_coord) / range_coord
df['dx_norm'] = (df['dest_x'] - min_coord) / range_coord
df['dy_norm'] = (df['dest_y'] - min_coord) / range_coord

features = ['t_norm', 'ox_norm', 'oy_norm', 'dx_norm', 'dy_norm']
centroids = df[features].sample(n=FINAL_CLUSTERS_NUM, random_state=42).values.tolist()

# Similarity measure
def calculate_custom_distance(agent, center):
    # agent and center: [t, ox, oy, dx, dy]
    diff_time = abs(agent[0] - center[0])
    dist_origin = math.sqrt((agent[1] - center[1])**2 + (agent[2] - center[2])**2)
    dist_dest = math.sqrt((agent[3] - center[3])**2 + (agent[4] - center[4])**2)
    return (TIME_WEIGHT * diff_time) + (SPACE_WEIGHT * (dist_origin + dist_dest))

for iteration in range(max_iters):
    clusters = []
    # Assign each vehicle to the "closest" centroid based on the custom similarity measure
    for index, row in df[features].iterrows():
        point = row.tolist()
        distances = [calculate_custom_distance(point, center) for center in centroids]
        closest_cluster = distances.index(min(distances))
        clusters.append(closest_cluster)
        
    df['cluster'] = clusters
    
    # Recalculate the centroids as the exact mathematical mean of all vehicles in that group
    new_centroids = []
    for i in range(FINAL_CLUSTERS_NUM):
        cluster_cars = df[df['cluster'] == i][features]
        if not cluster_cars.empty:
            new_centroids.append(cluster_cars.mean().tolist())
        else:
            new_centroids.append(centroids[i])
            
    if centroids == new_centroids:
        break
        
    centroids = new_centroids

columns_to_keep = ['start_time', 'origin', 'destination', 'kind', 'origin_real_id', 'dest_real_id', 'origin_x', 'origin_y', 'dest_x', 'dest_y', 'cluster']
df_final = df[columns_to_keep]

print(df_final[['start_time', 'origin', 'destination', 'cluster']].head(10))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ADJUST THIS PATH IF NEEDED
df_final.to_csv('clustering_ideas\\saint_arnoult_clustering\\agents_clustered_with_similarity_measure.csv', index=False)
# df_final.to_csv('clustering_ideas\\ingolstadt_custom_clustering\\agents_clustered_with_similarity_measure.csv', index=False)
print("\nSaved to agents_clustered_with_similarity_measure.csv")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~