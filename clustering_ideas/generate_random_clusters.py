import pandas as pd
import numpy as np
import os


# skrypt generuje losowe klastrowanie agentów z pliku CSV i zapisuje wyniki do nowych plików CSV w określonym katalogu.
# ma przygotować 3 scenariusze:
# 1. Mało klastrów (np. 3)
# 2. Przeciętna, optymalna ilość (np. 12 - żeby porównać fair 1:1 z By-Path)
# 3. Bardzo dużo klastrów (np. 3-4 auta na klaster)


INPUT_CSV = 'clustering_ideas/ingolstadt_custom_clustering/ingolstadt_custom_agents_coords.csv'
OUTPUT_DIR = 'clustering_ideas/ingolstadt_custom_clustering'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

df = pd.read_csv(INPUT_CSV)
num_agents = len(df)

K_SCENARIOS = {
    "small": 3,
    "medium": 12,
    "huge": num_agents // 3  
}

print(f"Wczytano {num_agents} agentów. Rozpoczynam losowe klastrowanie...")

for scenario_name, k in K_SCENARIOS.items():
    df_random = df.copy()
    
    df_random['cluster'] = np.random.randint(0, k, size=num_agents)
    
    filename = f"agents_clustered_random_{scenario_name}_K{k}.csv"
    output_path = os.path.join(OUTPUT_DIR, filename)
    df_random.to_csv(output_path, index=False)
    
    print(f" -> Scenario saved '{scenario_name}' (K={k}) in: {filename}")

print("\nReady!")