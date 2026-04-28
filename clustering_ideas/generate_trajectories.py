import sumolib
import pandas as pd
import json
import os

# NET_PATH = 'networks/ingolstadt_custom/ingolstadt_custom.net.xml'
# AGENTS_PATH = 'clustering_ideas/ingolstadt_custom_clustering/ingolstadt_custom_agents_coords.csv'
# OUTPUT_PATH = 'clustering_ideas/ingolstadt_custom_clustering/agent_paths.json'

NET_PATH = 'networks/provins/provins.net.xml'
AGENTS_PATH = 'clustering_ideas/provins_clustering/provins_agents_coords.csv'
OUTPUT_PATH = 'clustering_ideas/provins_clustering/agent_paths.json'

net = sumolib.net.readNet(NET_PATH)
df = pd.read_csv(AGENTS_PATH)

agent_paths = {}

print(f"Generate trajectories for {len(df)} agents")
for idx, row in df.iterrows():
    try:
        e_from = net.getEdge(row['origin_real_id'])
        e_to = net.getEdge(row['dest_real_id'])
        
        path, _ = net.getShortestPath(e_from, e_to)
        
        if path:
            agent_paths[int(row['id'])] = [edge.getID() for edge in path]
        else:
            agent_paths[int(row['id'])] = []
    except:
        agent_paths[int(row['id'])] = []

with open(OUTPUT_PATH, 'w') as f:
    json.dump(agent_paths, f)

print(f"Saved in {OUTPUT_PATH}")