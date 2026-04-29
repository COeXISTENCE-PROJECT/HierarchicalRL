"""
SPATIAL CLUSTERING & ELBOW METHOD SCRIPT
Evaluates multiple initial K values to find the optimal variance (MSE) 
while tracking the final number of clusters after geometric merging.
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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Wczytywanie danych
df = pd.read_csv('clustering_ideas\\provins_clustering\\provins_agents_coords.csv')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

df['vec_x'] = df['dest_x'] - df['origin_x']
df['vec_y'] = df['dest_y'] - df['origin_y']

features = ['origin_x', 'origin_y', 'vec_x', 'vec_y']
X = StandardScaler().fit_transform(df[features])

k_values = []
mse_values = []
final_k_values = []


for current_k in range(MIN_K, MAX_K + 1, STEP_K):
    print(f"Testowanie K_initial = {current_k}...")
    
    kmeans = KMeans(n_clusters=current_k, random_state=42)
    df['cluster'] = kmeans.fit_predict(X)
    
    mse = kmeans.inertia_ / len(X)
    
    lines = []
    for i in range(current_k):
        c = df[df['cluster'] == i]
        if not c.empty:
            lines.append(LineString([(c['origin_x'].mean(), c['origin_y'].mean()), 
                                     (c['dest_x'].mean(), c['dest_y'].mean())]))
        else:
            lines.append(LineString([(0,0), (0,0)]))

    final_map = {i: i for i in range(current_k)}
    for i in range(current_k):
        for j in range(i + 1, current_k):
            if lines[i].length > 0 and lines[j].length > 0 and lines[i].intersects(lines[j]):
                final_map[j] = final_map[i]

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
ax2.plot(k_values, final_k_values, marker='s', linestyle=':', color='purple', linewidth=2, label='Ostateczna liczba klastrów')
ax2.set_ylabel('Ile klastrów zostało po przecięciach linii?', color='purple', fontweight='bold')
ax2.tick_params(axis='y', labelcolor='purple')

plt.title('Spatial KMeans + Geometria Przecięć')
fig.tight_layout()

plot_path = 'clustering_ideas\\provins_clustering\\mse_spatial_kmeans_plot.png'
plt.savefig(plot_path)
print(f"Zapisano wykres do: {plot_path}")

plt.show()