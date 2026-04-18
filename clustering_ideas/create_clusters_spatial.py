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
features = ['origin_x', 'origin_y', 'vec_x', 'vec_y']
X = StandardScaler().fit_transform(df[features])

K_initial = 10 # Zaczynamy od większej liczby grup
df['cluster'] = KMeans(n_clusters=K_initial, random_state=42).fit_predict(X)

# 4. Łączenie klastrów, których trasy się przecinają
lines = []
for i in range(K_initial):
    c = df[df['cluster'] == i]
    lines.append(LineString([(c['origin_x'].mean(), c['origin_y'].mean()), 
                             (c['dest_x'].mean(), c['dest_y'].mean())]))

final_map = {i: i for i in range(K_initial)}
for i in range(K_initial):
    for j in range(i + 1, K_initial):
        if lines[i].intersects(lines[j]):
            final_map[j] = final_map[i]

df['cluster'] = df['cluster'].map(final_map)
df.to_csv('clustering_ideas\\ingolstadt_custom_clustering\\agents_clustered_with_spatial.csv', index=False)
print("Gotowe! Klastry wektorowe zapisane.")