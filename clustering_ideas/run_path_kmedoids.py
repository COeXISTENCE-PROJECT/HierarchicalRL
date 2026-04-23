import numpy as np
import pandas as pd

# Parametry algorytmu
K = 15  # Liczba klastrów (Twoja decyzja, mentor sugeruje 15-20)
MAX_ITER = 100
MATRIX_PATH = 'ingolstadt_custom_clustering/path_distance_matrix.npy'
AGENTS_PATH = 'ingolstadt_custom_clustering/ingolstadt_custom_agents_coords.csv'
OUTPUT_CSV = 'ingolstadt_custom_clustering/agents_clustered_by_path.csv'

dist_matrix = np.load(MATRIX_PATH)
n = dist_matrix.shape[0]

# 1. Inicjalizacja: wybieramy losowe indeksy jako początkowe medoidy
medoids = np.random.choice(n, K, replace=False)

print("Rozpoczynam pętlę K-Medoids...")
for m_iter in range(MAX_ITER):
    # Przypisz agentów do najbliższego medoida
    # argmin po kolumnach macierzy odległości dla wybranych wierszy (medoidów)
    clusters = np.argmin(dist_matrix[medoids, :], axis=0)
    
    new_medoids = np.copy(medoids)
    
    for k in range(K):
        # Znajdź wszystkich agentów należących do klastra k
        cluster_indices = np.where(clusters == k)[0]
        if len(cluster_indices) == 0:
            continue
            
        # Wewnątrz klastra szukamy agenta, który ma najmniejszą sumę odległości do innych
        # To jest nowy medoid
        cluster_dist = dist_matrix[np.ix_(cluster_indices, cluster_indices)]
        total_dists = np.sum(cluster_dist, axis=1)
        new_medoids[k] = cluster_indices[np.argmin(total_dists)]
    
    if np.array_equal(medoids, new_medoids):
        print(f"Konwergencja osiągnięta w iteracji {m_iter}")
        break
    medoids = new_medoids

# Zapis wyników
df = pd.read_csv(AGENTS_PATH)
df['cluster'] = clusters
df.to_csv(OUTPUT_CSV, index=False)
print(f"Wyniki zapisane w {OUTPUT_CSV}. Klastry wyznaczone na podstawie przepływu tras.")