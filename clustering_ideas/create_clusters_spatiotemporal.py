import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from shapely.geometry import LineString

# 1. Wczytanie danych z Twoimi nowymi współrzędnymi
df = pd.read_csv('clustering_ideas\\ingolstadt_custom_clustering\\ingolstadt_custom_agents_coords.csv')

# 2. Obliczanie wektorów (Twój Pomysł 7: Gdzie jadą?)
df['vec_x'] = df['dest_x'] - df['origin_x']
df['vec_y'] = df['dest_y'] - df['origin_y']

# 3. Klastrowanie po Pozycji + Kierunku
features = ['origin_x', 'origin_y', 'vec_x', 'vec_y', 'start_time']
X = StandardScaler().fit_transform(df[features])

K_initial = 20 # Zaczynamy od większej liczby grup
df['cluster'] = KMeans(n_clusters=K_initial, random_state=42).fit_predict(X)

# 4. Wyliczanie centroidów czasowo-przestrzennych dla klastrów
cluster_info = []
for i in range(K_initial):
    c = df[df['cluster'] == i]
    if len(c) == 0: continue
    
    # Tworzymy linię z uśrednionych współrzędnych
    line = LineString([(c['origin_x'].mean(), c['origin_y'].mean()), 
                       (c['dest_x'].mean(), c['dest_y'].mean())])
    
    # Zapamiętujemy średni czas startu klastra
    mean_time = c['start_time'].mean()
    
    cluster_info.append({
        'id': i,
        'line': line,
        'mean_time': mean_time
    })

TIME_THRESHOLD = 60

final_map = {i: i for i in range(K_initial)}

for i in range(len(cluster_info)):
    for j in range(i + 1, len(cluster_info)):
        c1 = cluster_info[i]
        c2 = cluster_info[j]
        
        # Warunek 1: Przecięcie wektorów
        intersects = c1['line'].intersects(c2['line'])
        
        # Warunek 2: Bliskość czasowa
        time_diff = abs(c1['mean_time'] - c2['mean_time'])
        close_in_time = time_diff < TIME_THRESHOLD
        
        if intersects and close_in_time:
            # Łączymy - przypisujemy j do grupy i
            final_map[c2['id']] = final_map[c1['id']]

df['cluster'] = df['cluster'].map(final_map)
df.to_csv('clustering_ideas\\ingolstadt_custom_clustering\\agents_clustered_with_spatiotemporal.csv', index=False)
print(f"Gotowe! Klastry (przestrzeń+kierunek+czas) zapisane. Ostateczna liczba klastrów: {df['cluster'].nunique()}")