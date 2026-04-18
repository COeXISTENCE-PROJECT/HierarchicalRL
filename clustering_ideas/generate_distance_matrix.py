import sumolib
import pandas as pd
import json
import itertools

print("Start")
net = sumolib.net.readNet('networks\\ingolstadt_custom\\ingolstadt_custom.net.xml')
df = pd.read_csv('clustering_ideas\\ingolstadt_custom_clustering\\ingolstadt_custom_agents_coords.csv')

unique_origins = set(df['origin_real_id'].dropna())
unique_dests = set(df['dest_real_id'].dropna())
all_unique_edges = list(unique_origins.union(unique_dests))

distance_matrix = {}
max_dist = 0.0
print("Edges loaded")
for e in all_unique_edges:
    distance_matrix[e] = {}

print("Filling distance matrix")
for e1 in all_unique_edges:
    edge_from = net.getEdge(e1)
    for e2 in all_unique_edges:
        if e1 == e2:
            distance_matrix[e1][e2] = 0.0
            continue
            
        edge_to = net.getEdge(e2)
        
        # getShortestPath returns (list of edges, cost as length in meters)
        path, cost = net.getShortestPath(edge_from, edge_to)
        
        if path is None:
            # no path found ??
            cost = 99999.0
            
        distance_matrix[e1][e2] = cost
        if cost < 99999.0 and cost > max_dist:
            max_dist = cost

# Normalize the matrix (min max scaling to 0-1)
print(f"Normalizing")
for e1 in distance_matrix:
    for e2 in distance_matrix[e1]:
        distance_matrix[e1][e2] = distance_matrix[e1][e2] / max_dist

with open('clustering_ideas\\ingolstadt_custom_clustering\\shortest_path_metric_matrix.json', 'w') as f:
    json.dump(distance_matrix, f)

print("Saved as shortest_path_metric_matrix.json")