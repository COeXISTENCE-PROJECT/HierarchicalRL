import numpy as np
import pandas as pd

K = 15  
MAX_ITER = 100
# MATRIX_PATH = 'clustering_ideas/ingolstadt_custom_clustering/path_distance_matrix.npy'
# AGENTS_PATH = 'clustering_ideas/ingolstadt_custom_clustering/ingolstadt_custom_agents_coords.csv'
# OUTPUT_CSV = 'clustering_ideas/ingolstadt_custom_clustering/agents_clustered_by_path.csv'
MATRIX_PATH = 'clustering_ideas/provins_clustering/path_distance_matrix.npy'
AGENTS_PATH = 'clustering_ideas/provins_clustering/provins_agents_coords.csv'
OUTPUT_CSV = 'clustering_ideas/provins_clustering/agents_clustered_by_path.csv'

dist_matrix = np.load(MATRIX_PATH)
n = dist_matrix.shape[0]

medoids = np.random.choice(n, K, replace=False)

for m_iter in range(MAX_ITER):
    
    clusters = np.argmin(dist_matrix[medoids, :], axis=0)
    
    new_medoids = np.copy(medoids)
    
    for k in range(K):
        cluster_indices = np.where(clusters == k)[0]
        if len(cluster_indices) == 0:
            continue
            
        cluster_dist = dist_matrix[np.ix_(cluster_indices, cluster_indices)]
        total_dists = np.sum(cluster_dist, axis=1)
        new_medoids[k] = cluster_indices[np.argmin(total_dists)]
    
    if np.array_equal(medoids, new_medoids):
        print(f"Convergence in  {m_iter} steps")
        break
    medoids = new_medoids

df = pd.read_csv(AGENTS_PATH)
df['cluster'] = clusters
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved in {OUTPUT_CSV}")