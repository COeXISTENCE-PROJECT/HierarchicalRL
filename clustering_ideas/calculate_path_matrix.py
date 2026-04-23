import json
import numpy as np

INPUT_PATH = 'ingolstadt_custom_clustering/agent_paths.json'
MATRIX_OUTPUT = 'ingolstadt_custom_clustering/path_distance_matrix.npy'

with open(INPUT_PATH, 'r') as f:
    agent_paths = json.load(f)

agent_ids = sorted([int(k) for k in agent_paths.keys()])
n = len(agent_ids)
dist_matrix = np.zeros((n, n))

print(f"Obliczanie macierzy Jaccarda dla {n} agentów ({n*n} operacji)...")

for i in range(n):
    path_i = set(agent_paths[str(agent_ids[i])])
    for j in range(i, n):
        path_j = set(agent_paths[str(agent_ids[j])])
        
        if not path_i or not path_j:
            dist = 1.0
        else:
            intersection = len(path_i.intersection(path_j))
            union = len(path_i.union(path_j))
            dist = 1.0 - (intersection / union)
        
        dist_matrix[i, j] = dist
        dist_matrix[j, i] = dist

np.save(MATRIX_OUTPUT, dist_matrix)
print("Macierz odległości zapisana.")