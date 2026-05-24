"""
SPATIAL CLUSTERING & ELBOW METHOD SCRIPT
Evaluates multiple initial K values to find the optimal variance (MSE) 
while tracking the final number of clusters after geometric merging.
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

CSV_PATH = 'clustering_ideas\\ingolstadt_custom_clustering\\ingolstadt_custom_agents_coords.csv'
PLOT_PATH = 'clustering_ideas\\ingolstadt_custom_clustering\\auto_elbow_spatial_plot.png'

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


print("Wczytywanie danych")
df = pd.read_csv(CSV_PATH)

df['vec_x'] = df['dest_x'] - df['origin_x']
df['vec_y'] = df['dest_y'] - df['origin_y']

features = ['origin_x', 'origin_y', 'vec_x', 'vec_y']
X = StandardScaler().fit_transform(df[features])

k_values = []
mse_values = []
final_k_values = []

print("Klastrowanie dla różnych K")
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

best_initial_k = find_optimal_k(k_values, mse_values)
opt_index = k_values.index(best_initial_k)
best_mse = mse_values[opt_index]
best_final_k = final_k_values[opt_index]

print(f">>> Optymalne początkowe K wybrane matematycznie (Elbow): {best_initial_k} <<<")
print(f">>> Co po przecięciach daje ostateczną liczbę klastrów: {best_final_k} <<<")

fig, ax1 = plt.subplots(figsize=(10, 6))

# Oś 1: MSE
ax1.plot(k_values, mse_values, marker='o', linestyle='-', color='green', linewidth=2, label='MSE (Wariancja KMeans)')
# Zaznaczenie Optymalnego K na zielonej linii
ax1.plot(best_initial_k, best_mse, marker='o', markersize=12, color='blue', label=f'Wybrane początkowe K={best_initial_k}')

ax1.set_xlabel('Początkowa liczba klastrów (K_initial)')
ax1.set_ylabel('Mean Squared Error (MSE)', color='green', fontweight='bold')
ax1.tick_params(axis='y', labelcolor='green')
ax1.set_xticks(k_values)
ax1.grid(True, linestyle='--', alpha=0.7)

# Oś 2: Ostateczna liczba klastrów
ax2 = ax1.twinx() 
ax2.plot(k_values, final_k_values, marker='s', linestyle=':', color='purple', linewidth=2, label='Ostateczna liczba klastrów')
# Zaznaczenie wyniku na fioletowej linii
ax2.plot(best_initial_k, best_final_k, marker='s', markersize=10, color='magenta')

ax2.set_ylabel('Ile klastrów zostało po przecięciach linii?', color='purple', fontweight='bold')
ax2.tick_params(axis='y', labelcolor='purple')

plt.title('Spatial KMeans + Geometria Przecięć')

# Łączenie legend z obu osi, żeby ładnie wyglądały w jednym oknie
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

fig.tight_layout()

plt.savefig(PLOT_PATH)
print(f"Zapisano wykres do: {PLOT_PATH}")

print("\nGotowe! Najlepsze parametry dla spatial:")
print(f"Początkowe num_clusters (K_initial): {int(best_initial_k)}")
print(f"Przewidywana Ostateczna liczba klastrów: {int(best_final_k)}")

plt.show()