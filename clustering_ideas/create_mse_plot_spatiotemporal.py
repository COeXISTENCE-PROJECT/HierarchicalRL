"""
SPATIOTEMPORAL CLUSTERING & ELBOW METHOD SCRIPT
Evaluates multiple initial K values to find the optimal variance (MSE) 
while tracking the final number of clusters after geometric AND time-based merging.
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from shapely.geometry import LineString
import matplotlib.pyplot as plt

MIN_K = 2
MAX_K = 40
STEP_K = 2
TIME_THRESHOLD = 60

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Wczytywanie danych
df = pd.read_csv('clustering_ideas\\saint_arnoult_clustering\\saint_arnoult_agents_coords.csv')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

df['vec_x'] = df['dest_x'] - df['origin_x']
df['vec_y'] = df['dest_y'] - df['origin_y']

features = ['origin_x', 'origin_y', 'vec_x', 'vec_y', 'start_time']
X = StandardScaler().fit_transform(df[features])

k_values = []
mse_values = []
final_k_values = [] # Ilość klastrów po fuzji

for current_k in range(MIN_K, MAX_K + 1, STEP_K):
    print(f"Testowanie K_initial = {current_k}...")
    
    kmeans = KMeans(n_clusters=current_k, random_state=42)
    df['cluster'] = kmeans.fit_predict(X)
    
    mse = kmeans.inertia_ / len(X)
    
    cluster_info = []
    for i in range(current_k):
        c = df[df['cluster'] == i]
        if len(c) == 0: continue
        
        line = LineString([(c['origin_x'].mean(), c['origin_y'].mean()), 
                           (c['dest_x'].mean(), c['dest_y'].mean())])
        mean_time = c['start_time'].mean()
        
        cluster_info.append({
            'id': i,
            'line': line,
            'mean_time': mean_time
        })

    final_map = {i: i for i in range(current_k)}

    for i in range(len(cluster_info)):
        for j in range(i + 1, len(cluster_info)):
            c1 = cluster_info[i]
            c2 = cluster_info[j]
            
            if c1['line'].length > 0 and c2['line'].length > 0:
                intersects = c1['line'].intersects(c2['line'])
            else:
                intersects = False
            
            time_diff = abs(c1['mean_time'] - c2['mean_time'])
            close_in_time = time_diff < TIME_THRESHOLD
            
            if intersects and close_in_time:
                final_map[c2['id']] = final_map[c1['id']]

    df['final_cluster'] = df['cluster'].map(final_map)
    final_clusters_count = df['final_cluster'].nunique()
    
    k_values.append(current_k)
    mse_values.append(mse)
    final_k_values.append(final_clusters_count)
    
    print(f"-> MSE: {mse:.4f} | Połączono w {final_clusters_count} końcowych grup.\n")

fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(k_values, mse_values, marker='o', linestyle='-', color='green', linewidth=2, label='MSE (Wariancja KMeans)')
ax1.set_xlabel('Początkowa liczba klastrów (K_initial)')
ax1.set_ylabel('Mean Squared Error (MSE)', color='green', fontweight='bold')
ax1.tick_params(axis='y', labelcolor='green')
ax1.set_xticks(k_values)
ax1.grid(True, linestyle='--', alpha=0.7)

ax2 = ax1.twinx()  
ax2.plot(k_values, final_k_values, marker='s', linestyle=':', color='blue', linewidth=2, label='Ostateczna liczba klastrów')
ax2.set_ylabel('Ile klastrów zostało po przecięciach?', color='blue', fontweight='bold')
ax2.tick_params(axis='y', labelcolor='blue')

plt.title(f'Spatiotemporal KMeans (Próg czasu: {TIME_THRESHOLD}s)')
fig.tight_layout()

plot_path = 'clustering_ideas\\saint_arnoult_clustering\\mse_spatiotemporal_kmeans_plot.png'
plt.savefig(plot_path)
print(f"Zapisano wykres do: {plot_path}")

plt.show()
