"""
SPATIOTEMPORAL CLUSTERING & ELBOW METHOD SCRIPT
Evaluates multiple initial K values to find the optimal variance (MSE) 
while tracking the final number of clusters after geometric AND time-based merging.
Auto-selects optimal K using the perpendicular distance (Elbow) method.
"""
print("START")
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

CSV_PATH = 'clustering_ideas\\saint_arnoult_clustering\\saint_arnoult_agents_coords.csv'
PLOT_PATH = 'clustering_ideas\\saint_arnoult_clustering\\auto_elbow_spatiotemporal_plot.png'

def find_optimal_k(k_values, error_values):
    x = np.array(k_values)
    y = np.array(error_values)
    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])
    
    distances = []
    for i in range(len(x)):
        p_i = np.array([x[i], y[i]])
        numerator = np.abs(np.cross(p2 - p1, p1 - p_i))
        denominator = np.linalg.norm(p2 - p1)
        distances.append(numerator / denominator)
        
    optimal_index = np.argmax(distances)
    return x[optimal_index]

print("data loading")
df = pd.read_csv(CSV_PATH)

df['vec_x'] = df['dest_x'] - df['origin_x']
df['vec_y'] = df['dest_y'] - df['origin_y']

features = ['origin_x', 'origin_y', 'vec_x', 'vec_y', 'start_time']
X = StandardScaler().fit_transform(df[features])

k_values = []
mse_values = []
final_k_values = [] 

print("\nclustering (spatiotemporal) for different K values")
for current_k in range(MIN_K, MAX_K + 1, STEP_K):
    print(f"Testing K_initial = {current_k}...")
    
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
    
    print(f"-> MSE: {mse:.4f} | Combined into {final_clusters_count} final clusters.\n")


best_initial_k = find_optimal_k(k_values, mse_values)
opt_index = k_values.index(best_initial_k)
best_mse = mse_values[opt_index]
best_final_k = final_k_values[opt_index]

print(f">>> Optimal initial K selected (Elbow): {best_initial_k} <<<")
print(f">>> What results in the final number of clusters: {best_final_k} <<<")


fig, ax1 = plt.subplots(figsize=(10, 6))

#axis 1: MSE values
ax1.plot(k_values, mse_values, marker='o', linestyle='-', color='green', linewidth=2, label='MSE (KMeans Variance)')
ax1.plot(best_initial_k, best_mse, marker='o', markersize=12, color='red', label=f'Optimal initial K={best_initial_k}')

ax1.set_xlabel('Initial number of clusters (K_initial)')
ax1.set_ylabel('Mean Squared Error (MSE)', color='green', fontweight='bold')
ax1.tick_params(axis='y', labelcolor='green')
ax1.set_xticks(k_values)
ax1.grid(True, linestyle='--', alpha=0.7)

# axis 2: Final number of clusters after merging
ax2 = ax1.twinx()  
ax2.plot(k_values, final_k_values, marker='s', linestyle=':', color='blue', linewidth=2, label='Final number of clusters')
ax2.plot(best_initial_k, best_final_k, marker='s', markersize=10, color='magenta')

ax2.set_ylabel('Final number of clusters', color='blue', fontweight='bold')
ax2.tick_params(axis='y', labelcolor='blue')

plt.title(f'Spatiotemporal KMeans (Time threshold: {TIME_THRESHOLD}s)')

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

fig.tight_layout()

plt.savefig(PLOT_PATH)
print(f"Plot saved to: {PLOT_PATH}")

print("\nDone! Optimal parameters for spatiotemporal clustering:")
print(f"Initial number of clusters (K_initial): {int(best_initial_k)}")
print(f"Predicted final number of clusters (after merging): {int(best_final_k)}")

plt.show()