import json
import numpy as np
import pandas as pd

# Wczytujemy ścieżki i oryginalne dane (z czasem)
# with open('clustering_ideas/ingolstadt_custom_clustering/agent_paths.json', 'r') as f:
#     agent_paths = json.load(f)
# df = pd.read_csv('clustering_ideas/ingolstadt_custom_clustering/ingolstadt_custom_agents_coords.csv')
with open('clustering_ideas/provins_clustering/agent_paths.json', 'r') as f:
    agent_paths = json.load(f)
df = pd.read_csv('clustering_ideas/provins_clustering/provins_agents_coords.csv')

TIME_WEIGHT = 0.4 
PATH_WEIGHT = 0.6  

agent_ids = sorted([int(k) for k in agent_paths.keys()])
n = len(agent_ids)
dist_matrix = np.zeros((n, n))

times = df.set_index('id').loc[agent_ids, 'start_time'].values
max_time_diff = times.max() - times.min()


for i in range(n):
    path_i = set(agent_paths[str(agent_ids[i])])
    time_i = times[i]
    
    for j in range(i, n):
        path_j = set(agent_paths[str(agent_ids[j])])
        time_j = times[j]
        
        if not path_i or not path_j:
            p_dist = 1.0
        else:
            intersection = len(path_i.intersection(path_j))
            union = len(path_i.union(path_j))
            p_dist = 1.0 - (intersection / union)
        
        t_dist = abs(time_i - time_j) / max_time_diff
        
        combined_dist = (PATH_WEIGHT * p_dist) + (TIME_WEIGHT * t_dist)
        
        dist_matrix[i, j] = combined_dist
        dist_matrix[j, i] = combined_dist

np.save('clustering_ideas/provins_clustering/path_distance_matrix.npy', dist_matrix)
# np.save('clustering_ideas/ingolstadt_custom_clustering/path_distance_matrix.npy', dist_matrix)
print("Saved.")