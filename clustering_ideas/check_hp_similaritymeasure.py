"""
FULL GRID SEARCH K-MEDOIDS PIPELINE (EUCLIDEAN SPATIAL)
Tests ALL combinations of K and Weights
Finds the global optimal weights per K using Utopia distance, 
and then finds the optimal K using the Elbow method on those best distances
"""
print("START")
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import math

MIN_K = 5
MAX_K = 30
STEP_K = 2
MAX_ITERS = 50
print("Configuration set")

CSV_PATH = 'clustering_ideas\\saint_arnoult_clustering\\saint_arnoult_agents_coords.csv'
PLOT_K_PATH = 'clustering_ideas\\saint_arnoult_clustering\\auto_elbow_k_similaritymeasure_plot.png'
print("k plot")
PLOT_W_PATH = 'clustering_ideas\\saint_arnoult_clustering\\auto_weights_similaritymeasure_plot.png'
print("paths set")
print("Paths set")

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

print("\nWczytywanie danych")
df = pd.read_csv(CSV_PATH)

min_t, max_t = df['start_time'].min(), df['start_time'].max()
range_t = max_t - min_t if max_t != min_t else 1
df['t_norm'] = (df['start_time'] - min_t) / range_t

features = ['t_norm', 'origin_x', 'origin_y', 'dest_x', 'dest_y']
def calculate_euclidean_distance(agent_row, center_row, tw, sw):
    diff_time = abs(agent_row['t_norm'] - center_row['t_norm'])
    
    orig_dist = math.sqrt((agent_row['origin_x'] - center_row['origin_x'])**2 + 
                          (agent_row['origin_y'] - center_row['origin_y'])**2)
                          
    dest_dist = math.sqrt((agent_row['dest_x'] - center_row['dest_x'])**2 + 
                          (agent_row['dest_y'] - center_row['dest_y'])**2)
                          
    return (tw * diff_time) + (sw * (orig_dist + dest_dist))

def get_pure_spatial_error(agent_row, center_row):
    orig_dist = math.sqrt((agent_row['origin_x'] - center_row['origin_x'])**2 + 
                          (agent_row['origin_y'] - center_row['origin_y'])**2)
                          
    dest_dist = math.sqrt((agent_row['dest_x'] - center_row['dest_x'])**2 + 
                          (agent_row['dest_y'] - center_row['dest_y'])**2)
                          
    return orig_dist + dest_dist

def get_pure_temporal_error(agent_row, center_row):
    return abs(agent_row['t_norm'] - center_row['t_norm'])


print("\ngrid search")

time_weights_test = np.linspace(0.0, 1.0, 11)
space_weights_test = 1.0 - time_weights_test
k_list = list(range(MIN_K, MAX_K + 1, STEP_K))

total_runs = len(k_list) * len(time_weights_test)
current_run = 1

all_results = []

for current_k in k_list:
    for tw, sw in zip(time_weights_test, space_weights_test):
        print(f"[{current_run:03d}/{total_runs}] Trenowanie: K={current_k:02d} | TW={tw:.1f} | SW={sw:.1f}")
        
        centroids = df[features].sample(n=current_k, random_state=42).to_dict('records')
        
        for iteration in range(MAX_ITERS):
            clusters = []
            for index, row in df.iterrows():
                distances = [calculate_euclidean_distance(row, center, tw, sw) for center in centroids]
                clusters.append(distances.index(min(distances)))
            df['cluster'] = clusters
            
            new_centroids = []
            for i in range(current_k):
                cluster_cars = df[df['cluster'] == i]
                if not cluster_cars.empty:
                    ideal_t = cluster_cars['t_norm'].mean()
                    ideal_ox = cluster_cars['origin_x'].mean()
                    ideal_oy = cluster_cars['origin_y'].mean()
                    ideal_dx = cluster_cars['dest_x'].mean()
                    ideal_dy = cluster_cars['dest_y'].mean()
                    
                    virtual_center = {
                        't_norm': ideal_t, 
                        'origin_x': ideal_ox, 
                        'origin_y': ideal_oy, 
                        'dest_x': ideal_dx, 
                        'dest_y': ideal_dy
                    }
                    
                    best_car = None
                    min_dist = float('inf')
                    for _, car_row in cluster_cars.iterrows():
                        d = calculate_euclidean_distance(car_row, virtual_center, tw, sw)
                        if d < min_dist:
                            min_dist = d
                            best_car = car_row.to_dict()
                            
                    new_centroids.append({
                        't_norm': best_car['t_norm'], 
                        'origin_x': best_car['origin_x'], 
                        'origin_y': best_car['origin_y'],
                        'dest_x': best_car['dest_x'], 
                        'dest_y': best_car['dest_y']
                    })
                else:
                    new_centroids.append(centroids[i])
                    
            if centroids == new_centroids: break
            centroids = new_centroids

        total_spatial_sq = 0
        total_temporal_sq = 0
        N = len(df)
        for index, row in df.iterrows():
            assigned_center = centroids[int(row['cluster'])]
            s_err = get_pure_spatial_error(row, assigned_center)
            t_err = get_pure_temporal_error(row, assigned_center)
            total_spatial_sq += (s_err ** 2)
            total_temporal_sq += (t_err ** 2)
            
        all_results.append({
            'k': current_k,
            'tw': tw,
            'sw': sw,
            's_mse': total_spatial_sq / N,
            't_mse': total_temporal_sq / N
        })
        current_run += 1

df_results = pd.DataFrame(all_results)

s_min, s_max = df_results['s_mse'].min(), df_results['s_mse'].max()
t_min, t_max = df_results['t_mse'].min(), df_results['t_mse'].max()
s_range = s_max - s_min or 1.0
t_range = t_max - t_min or 1.0

df_results['s_norm'] = (df_results['s_mse'] - s_min) / s_range
df_results['t_norm'] = (df_results['t_mse'] - t_min) / t_range

# Obliczanie odległości od Utopii (0,0) dla każdej kombinacji
df_results['utopia_dist'] = np.sqrt(df_results['s_norm']**2 + df_results['t_norm']**2)

# Dla każdego K wybieramy wagi, które dały najmniejszą odległość od Utopii
best_per_k_idx = df_results.groupby('k')['utopia_dist'].idxmin()
best_per_k = df_results.loc[best_per_k_idx].sort_values('k').reset_index(drop=True)

#Szukamy "elbow point" na liście najlepszych wyników dla poszczególnych K
best_k = find_optimal_k(best_per_k['k'].tolist(), best_per_k['utopia_dist'].tolist())
print(f">>> Optymalne K wybrane matematycznie (Elbow): {best_k} <<<")

# Wyciągamy optymalne wagi dla wybranego K
optimal_row = best_per_k[best_per_k['k'] == best_k].iloc[0]
opt_tw = optimal_row['tw']
opt_sw = optimal_row['sw']
print(f">>> Optymalne Wagi dla K={best_k}: Czas={opt_tw:.1f}, Przestrzeń={opt_sw:.1f} <<<")

plt.figure(figsize=(10, 6))
plt.plot(best_per_k['k'], best_per_k['utopia_dist'], marker='o', linestyle='-', color='red', linewidth=2)
plt.plot(best_k, optimal_row['utopia_dist'], marker='o', markersize=12, color='blue', label=f'Wybrane K={best_k}')
plt.title('Znormalizowany Błąd (Utopia Distance) vs Liczba Klastrów (K)')
plt.xlabel('Liczba Klastrów (K)')
plt.ylabel('Kompozytowy Błąd Znormalizowany')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(k_list)
plt.legend()
plt.savefig(PLOT_K_PATH)

best_k_results = df_results[df_results['k'] == best_k].sort_values('tw')

fig, ax1 = plt.subplots(figsize=(12, 6))
x_positions = np.arange(len(time_weights_test))
x_labels = [f"{row.tw:.1f}:{row.sw:.1f}" for _, row in best_k_results.iterrows()]

color1 = 'tab:blue'
ax1.set_xlabel('Balans ( Czas : Przestrzeń )', fontweight='bold')
ax1.set_ylabel('Błąd Przestrzenny (MSE Euklidesowe)', color=color1, fontweight='bold')
ax1.plot(x_positions, best_k_results['s_mse'], marker='o', color=color1, linewidth=2.5)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_xticks(x_positions)
ax1.set_xticklabels(x_labels, rotation=45)

ax2 = ax1.twinx()
color2 = 'tab:orange'
ax2.set_ylabel('Błąd Czasowy (MSE Czasu)', color=color2, fontweight='bold')
ax2.plot(x_positions, best_k_results['t_mse'], marker='s', color=color2, linewidth=2.5, linestyle='--')
ax2.tick_params(axis='y', labelcolor=color2)

opt_w_index = list(time_weights_test).index(opt_tw)
plt.axvline(x=opt_w_index, color='green', linestyle=':', linewidth=3, label=f'Punkt Utopii ({opt_tw:.1f}:{opt_sw:.1f})')
plt.title(f'Kompromis Wag dla K-Medoids Euklidesowego (K={best_k})')
fig.tight_layout()
plt.legend()
plt.savefig(PLOT_W_PATH)

print("\nGotowe! Najlepsze parametry dla similarity measure (Euklides):")
print(f"num_clusters: {int(best_k)}")
print(f"TIME_WEIGHT: {opt_tw:.1f}")
print(f"SPACE_WEIGHT: {opt_sw:.1f}")