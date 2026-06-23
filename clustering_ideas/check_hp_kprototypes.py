"""
FULL GRID SEARCH K-PROTOTYPES PIPELINE (EUCLIDEAN SPATIAL)
Tests ALL combinations of K and Weights
Finds the global optimal weights per K using distance to (0,0) in normalized error space, 
and then finds the optimal K using the Elbow method on those best distances
"""
print("START")
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import math

# ================= CONFIGURATION =================
NUM_OF_ZONES = 5
MIN_K = 5
MAX_K = 30
STEP_K = 2
MAX_ITERS = 50
print("Configuration set")

CSV_PATH = 'clustering_ideas\\ingolstadt_custom_clustering\\ingolstadt_custom_agents_coords.csv'
print('csv')
PLOT_K_PATH = 'clustering_ideas\\ingolstadt_custom_clustering\\auto_elbow_kprototypes_plot.png'
print('plot k')
PLOT_W_PATH = 'clustering_ideas\\ingolstadt_custom_clustering\\auto_weights_kprototypes_plot.png'
print('plot w')
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

print("\nData loading")
df = pd.read_csv(CSV_PATH)

print(f"Creating initial clusters using K-Medoids (euclidean distance used)")
def simple_spatial_kmeans(coords_list, K=NUM_OF_ZONES, iters=30):
    centroids = random.sample(coords_list, K)
    
    for _ in range(iters):
        clusters = []
        for pt in coords_list:
            dists = [math.sqrt((pt[0]-c[0])**2 + (pt[1]-c[1])**2) for c in centroids]
            clusters.append(dists.index(min(dists)))
        
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

orig_coords = list(zip(df['origin_x'], df['origin_y']))
df['orig_cluster'] = simple_spatial_kmeans(orig_coords, K=NUM_OF_ZONES)

dest_coords = list(zip(df['dest_x'], df['dest_y']))
df['dest_cluster'] = simple_spatial_kmeans(dest_coords, K=NUM_OF_ZONES)

min_t, max_t = df['start_time'].min(), df['start_time'].max()
range_t = max_t - min_t if max_t != min_t else 1
df['t_norm'] = (df['start_time'] - min_t) / range_t

features = ['t_norm', 'orig_cluster', 'dest_cluster']

def calculate_similarity(agent, proto, tw, sw):
    diff_time = abs(agent[0] - proto[0])
    penalty = 0
    if agent[1] != proto[1]: penalty += 1
    if agent[2] != proto[2]: penalty += 1
    return (tw * diff_time) + (sw * penalty)

def get_mode(lst):
    return max(set(lst), key=lst.count)

print("\ngrid search")

time_weights_test = np.linspace(0.0, 1.0, 11)
space_weights_test = 1.0 - time_weights_test
k_list = list(range(MIN_K, MAX_K + 1, STEP_K))

total_runs = len(k_list) * len(time_weights_test)
current_run = 1

all_results = []

for current_k in k_list:
    for tw, sw in zip(time_weights_test, space_weights_test):
        print(f"[{current_run:03d}/{total_runs}] Training K-Prototypes: K={current_k:02d} | TW={tw:.1f} | SW={sw:.1f}")
        
        prototypes = df[features].sample(n=current_k, random_state=42).values.tolist()
        
        for iteration in range(MAX_ITERS):
            clusters = []
            for index, row in df[features].iterrows():
                agent = row.tolist()
                dists = [calculate_similarity(agent, p, tw, sw) for p in prototypes]
                clusters.append(dists.index(min(dists)))
            df['final_cluster'] = clusters
            
            new_prototypes = []
            for i in range(current_k):
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

        total_penalty_sq = 0
        total_time_sq = 0
        N = len(df)
        
        for index, row in df.iterrows():
            agent = [row['t_norm'], row['orig_cluster'], row['dest_cluster']]
            proto = prototypes[int(row['final_cluster'])]
            
            penalty = 0
            if agent[1] != proto[1]: penalty += 1
            if agent[2] != proto[2]: penalty += 1
            
            time_diff = abs(agent[0] - proto[0])
            
            total_penalty_sq += (penalty ** 2)
            total_time_sq += (time_diff ** 2)
            
        all_results.append({
            'k': current_k,
            'tw': tw,
            'sw': sw,
            's_mse': total_penalty_sq / N,
            't_mse': total_time_sq / N
        })
        current_run += 1

df_results = pd.DataFrame(all_results)

s_min, s_max = df_results['s_mse'].min(), df_results['s_mse'].max()
t_min, t_max = df_results['t_mse'].min(), df_results['t_mse'].max()
s_range = s_max - s_min or 1.0
t_range = t_max - t_min or 1.0

df_results['s_norm'] = (df_results['s_mse'] - s_min) / s_range
df_results['t_norm'] = (df_results['t_mse'] - t_min) / t_range

#Calculating distance from (0,0) for each combination
df_results['dist_to_origin'] = np.sqrt((df_results['s_norm'])**2 + (df_results['t_norm'])**2)
#For each K we choose the weights that gave the smallest distance
best_per_k_idx = df_results.groupby('k')['dist_to_origin'].idxmin()
best_per_k = df_results.loc[best_per_k_idx].sort_values('k').reset_index(drop=True)

#Looking for the elbow point on the list of best results for each K
best_k = find_optimal_k(best_per_k['k'].tolist(), best_per_k['dist_to_origin'].tolist())
print(f">>> Optimal K (Elbow): {best_k} <<<")

# Getting optimal weights for the chosen K
optimal_row = best_per_k[best_per_k['k'] == best_k].iloc[0]
opt_tw = optimal_row['tw']
opt_sw = optimal_row['sw']
print(f">>> Optimal Weights for K={best_k}: Time={opt_tw:.1f}, Space={opt_sw:.1f} <<<")

plt.figure(figsize=(10, 6))
plt.plot(best_per_k['k'], best_per_k['dist_to_origin'], marker='o', linestyle='-', color='red', linewidth=2)
plt.plot(best_k, optimal_row['dist_to_origin'], marker='o', markersize=12, color='blue', label=f'Chosen K={best_k}')
plt.title('Normalized Error (distance to (0,0)) vs Number of Clusters (K)')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Normalized Composite Error')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(k_list)
plt.legend()
plt.savefig(PLOT_K_PATH)

# Trade-off (only for chosen, best K)
best_k_results = df_results[df_results['k'] == best_k].sort_values('tw')

fig, ax1 = plt.subplots(figsize=(12, 6))
x_positions = np.arange(len(time_weights_test))
x_labels = [f"{row.tw:.1f}:{row.sw:.1f}" for _, row in best_k_results.iterrows()]

color1 = 'tab:blue'
ax1.set_xlabel('Balance ( Time : Space )', fontweight='bold')
ax1.set_ylabel('Spatial Error (MSE)', color=color1, fontweight='bold')
ax1.plot(x_positions, best_k_results['s_mse'], marker='o', color=color1, linewidth=2.5)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_xticks(x_positions)
ax1.set_xticklabels(x_labels, rotation=45)

ax2 = ax1.twinx()
color2 = 'tab:orange'
ax2.set_ylabel('Temporal Error (MSE)', color=color2, fontweight='bold')
ax2.plot(x_positions, best_k_results['t_mse'], marker='s', color=color2, linewidth=2.5, linestyle='--')
ax2.tick_params(axis='y', labelcolor=color2)

opt_w_index = list(time_weights_test).index(opt_tw)
plt.axvline(x=opt_w_index, color='green', linestyle=':', linewidth=3, label=f'Best point ({opt_tw:.1f}:{opt_sw:.1f})')
plt.title(f'Weights for K-Prototypes (K={best_k})')
fig.tight_layout()
plt.legend()
plt.savefig(PLOT_W_PATH)

print("\nBest Parameters for K-Prototypes:")
print(f"num_clusters (K): {int(best_k)}")
print(f"TIME_WEIGHT: {opt_tw:.1f}")
print(f"SPACE_WEIGHT: {opt_sw:.1f}")