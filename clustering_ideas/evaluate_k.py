import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

MATRIX_PATH = 'clustering_ideas/ingolstadt_custom_clustering/path_distance_matrix.npy'
OUTPUT_PLOT = 'clustering_ideas/ingolstadt_custom_clustering/k_evaluation_plot.png'

K_VALUES_TO_TEST = [4, 6,7, 8,9, 10,11,12,13,14, 15]
MAX_ITER = 100
REPEATS = 3 



print(f"Wczytywanie macierzy z: {MATRIX_PATH} ...")
dist_matrix = np.load(MATRIX_PATH)
n = dist_matrix.shape[0]
print(f"Liczba agentów: {n}")

def run_kmedoids(dist_mat, K, max_iter=100):
    num_agents = dist_mat.shape[0]
    medoids = np.random.choice(num_agents, K, replace=False)
    
    for m_iter in range(max_iter):
        clusters = np.argmin(dist_mat[medoids, :], axis=0)
        new_medoids = np.copy(medoids)
        
        for k in range(K):
            cluster_indices = np.where(clusters == k)[0]
            if len(cluster_indices) == 0:
                continue
            cluster_dist = dist_mat[np.ix_(cluster_indices, cluster_indices)]
            total_dists = np.sum(cluster_dist, axis=1)
            new_medoids[k] = cluster_indices[np.argmin(total_dists)]
            
        if np.array_equal(medoids, new_medoids):
            break
        medoids = new_medoids
        
    clusters = np.argmin(dist_mat[medoids, :], axis=0)
    inertia = np.sum(np.min(dist_mat[medoids, :], axis=0))
    return clusters, inertia

results = []
print("\nRozpoczynam testowanie K...")
print(f"{'K':>4} | {'Inercja (Błąd)':>15} | {'Silhouette Score':>18}")
print("-" * 43)

for k in K_VALUES_TO_TEST:
    best_inertia = float('inf')
    best_clusters = None
    
    for _ in range(REPEATS):
        clusters, inertia = run_kmedoids(dist_matrix, k, MAX_ITER)
        if inertia < best_inertia:
            best_inertia = inertia
            best_clusters = clusters
            
    if 1 < k < n:
        sil_score = silhouette_score(dist_matrix, best_clusters, metric="precomputed")
    else:
        sil_score = -1.0
        
    results.append({'K': k, 'Inertia': best_inertia, 'Silhouette': sil_score})
    print(f"{k:4d} | {best_inertia:15.2f} | {sil_score:18.4f}")


df_results = pd.DataFrame(results)
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:red'
ax1.set_xlabel('Liczba klastrów (K)', fontsize=12)
ax1.set_ylabel('Inercja (Całkowity Błąd)', color=color, fontsize=12)
ax1.plot(df_results['K'], df_results['Inertia'], marker='o', color=color, linewidth=2)
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.6)

ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('Silhouette Score (max 1.0)', color=color, fontsize=12)  
ax2.plot(df_results['K'], df_results['Silhouette'], marker='s', color=color, linewidth=2, linestyle='--')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Ocena liczby klastrów K: Metoda Łokcia vs Silhouette Score', fontsize=14)
fig.tight_layout()  

plt.savefig(OUTPUT_PLOT, dpi=300)
print(f"\nZakończono! Wykres zapisany jako: {OUTPUT_PLOT}")