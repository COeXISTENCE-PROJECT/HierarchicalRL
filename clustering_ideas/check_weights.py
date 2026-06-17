import numpy as np
import pandas as pd
import json
import os
import copy


K = 12
MAX_ITER = 100
EPISODES = 250  

PATHS_JSON = 'ingolstadt_custom_clustering/agent_paths.json'
AGENTS_CSV = 'ingolstadt_custom_clustering/ingolstadt_custom_agents_coords.csv'
OUTPUT_DIR_CSV = 'ingolstadt_custom_clustering'

OUTPUT_DIR_JSON = '../config/algo_config/feudal_hrl'

BASE_CONFIG = {
    "training_eps": 5000,
    "desc": "Initial feudal HRL config: discrete manager over route families and PPO-style worker updates.",
    "manager_period": 100,
    "use_cluster_embedding": True,
    "cluster_source": "path",
    "cluster_csv_path": "clustering_ideas/ingolstadt_custom_clustering/agents_clustered_by_path.csv",
    "cluster_key_columns": ["start_time", "origin", "destination"],
    "num_subgoals": 4,
    "num_clusters": 0,
    "cluster_embed_dim": 8,
    "manager_hidden_dims": [128, 128],
    "controller_hidden_dims": [128, 128],
    "subgoal_embed_dim": 16,
    "manager_lr": 0.0003,
    "controller_lr": 0.0003,
    "manager_epochs": 3,
    "controller_epochs": 3,
    "batch_size": 64,
    "update_every": 2,
    "manager_clip_eps": 0.2,
    "controller_clip_eps": 0.2,
    "manager_entropy_coef": 0.01,
    "controller_entropy_coef": 0.01,
    "normalize_advantage": True,
    "intrinsic_reward_weight": 0.15,
    "manager_reward_weight": 1.0,
    "goal_switch_penalty": 0.0,
    "worker_pretrain_epochs": 0,
    "action_mask_strategy": "uniform_bins",
    "save_subgoal_records": True
}

weight_experiments = []
for i in range(11):
    w_path = round(1.0 - (i * 0.1), 1)
    w_time = round(i * 0.1, 1)
    name = f"path{int(w_path*100):02d}_time{int(w_time*100):02d}"
    weight_experiments.append({"name": name, "path_w": w_path, "time_w": w_time})

print("Wczytywanie danych...")
if not os.path.exists(PATHS_JSON) or not os.path.exists(AGENTS_CSV):
    print(f"BŁĄD: Nie znaleziono plików wejściowych w {OUTPUT_DIR_CSV}")
    exit(1)

with open(PATHS_JSON, 'r') as f:
    agent_paths = json.load(f)
df = pd.read_csv(AGENTS_CSV)

agent_ids = sorted([int(k) for k in agent_paths.keys()])
n = len(agent_ids)
times = df.set_index('id').loc[agent_ids, 'start_time'].values
max_time_diff = times.max() - times.min()

print("Obliczenie bazowych macierzy dystansu (Trasa i Czas)...")
D_path = np.zeros((n, n))
D_time = np.zeros((n, n))

for i in range(n):
    path_i = set(agent_paths[str(agent_ids[i])])
    time_i = times[i]
    for j in range(i, n):
        path_j = set(agent_paths[str(agent_ids[j])])
        time_j = times[j]
        
        p_dist = 1.0 if not path_i or not path_j else 1.0 - (len(path_i.intersection(path_j)) / len(path_i.union(path_j)))
        t_dist = abs(time_i - time_j) / max_time_diff
        
        D_path[i, j] = D_path[j, i] = p_dist
        D_time[i, j] = D_time[j, i] = t_dist

def run_kmedoids(dist_mat, K, max_iter=100):
    medoids = np.random.choice(dist_mat.shape[0], K, replace=False)
    for _ in range(max_iter):
        clusters = np.argmin(dist_mat[medoids, :], axis=0)
        new_medoids = np.copy(medoids)
        for k in range(K):
            cluster_indices = np.where(clusters == k)[0]
            if len(cluster_indices) == 0: continue
            cluster_dist = dist_mat[np.ix_(cluster_indices, cluster_indices)]
            new_medoids[k] = cluster_indices[np.argmin(np.sum(cluster_dist, axis=1))]
        if np.array_equal(medoids, new_medoids): break
        medoids = new_medoids
    return np.argmin(dist_mat[medoids, :], axis=0)

print(f"\nRozpoczynam pętlę dla {len(weight_experiments)} kombinacji wag...")
os.makedirs(OUTPUT_DIR_JSON, exist_ok=True)

for exp in weight_experiments:
    print(f" -> Klastrowanie dla wag: PATH={exp['path_w']}, TIME={exp['time_w']}")
    dist_matrix = (exp['path_w'] * D_path) + (exp['time_w'] * D_time)
    clusters = run_kmedoids(dist_matrix, K, MAX_ITER)
    
    df_out = pd.read_csv(AGENTS_CSV)
    df_out['cluster'] = clusters
    csv_filename = f"agents_clustered_{exp['name']}.csv"
    out_csv_path = os.path.join(OUTPUT_DIR_CSV, csv_filename)
    df_out.to_csv(out_csv_path, index=False)
    
    new_config = copy.deepcopy(BASE_CONFIG)
    new_config['cluster_csv_path'] = f"clustering_ideas/ingolstadt_custom_clustering/{csv_filename}"
    new_config['num_subgoals'] = K
    new_config['num_clusters'] = K
    new_config['training_eps'] = EPISODES
    
    json_filename = f"config_ing_{exp['name']}.json"
    out_json_path = os.path.join(OUTPUT_DIR_JSON, json_filename)
    
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(new_config, f, indent=4)

print(f"\nSUKCES! Wygenerowano CSV w {OUTPUT_DIR_CSV} oraz 11 plików .json w {OUTPUT_DIR_JSON}.")