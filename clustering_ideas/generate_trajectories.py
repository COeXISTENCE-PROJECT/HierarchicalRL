import sumolib
import pandas as pd
import json
import os

# Parametry wejściowe
NET_PATH = '../networks/ingolstadt_custom/ingolstadt_custom.net.xml'
AGENTS_PATH = 'ingolstadt_custom_clustering/ingolstadt_custom_agents_coords.csv'
OUTPUT_PATH = 'ingolstadt_custom_clustering/agent_paths.json'

print("Wczytywanie sieci i danych...")
net = sumolib.net.readNet(NET_PATH)
df = pd.read_csv(AGENTS_PATH)

agent_paths = {}

print(f"Generowanie trajektorii dla {len(df)} agentów...")
for idx, row in df.iterrows():
    # Pobieramy krawędzie startowe i końcowe po ich realnym ID
    try:
        e_from = net.getEdge(row['origin_real_id'])
        e_to = net.getEdge(row['dest_real_id'])
        
        # Wyznaczamy najkrótszą ścieżkę (lista krawędzi)
        path, _ = net.getShortestPath(e_from, e_to)
        
        if path:
            # Zapisujemy tylko ID krawędzi jako listę stringów
            agent_paths[int(row['id'])] = [edge.getID() for edge in path]
        else:
            agent_paths[int(row['id'])] = [] # Brak połączenia
    except:
        agent_paths[int(row['id'])] = []

with open(OUTPUT_PATH, 'w') as f:
    json.dump(agent_paths, f)

print(f"Sukces. Trajektorie zapisane w {OUTPUT_PATH}")