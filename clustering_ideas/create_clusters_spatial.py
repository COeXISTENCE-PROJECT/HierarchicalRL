"""
This script implements a spatial clustering methodology for traffic agents based on their origin coordinates and movement vectors.
It employs KMeans to group agents with similar starting points and directions, followed by a geometric merge of clusters whose mean trajectories intersect.
The resulting output identifies distinct spatial traffic flows across the network, providing a structural basis for path-based agent categorization.
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from shapely.geometry import LineString

df = pd.read_csv('clustering_ideas\\ingolstadt_custom_clustering\\ingolstadt_custom_agents_coords.csv')

df['vec_x'] = df['dest_x'] - df['origin_x']
df['vec_y'] = df['dest_y'] - df['origin_y']

features = ['origin_x', 'origin_y', 'vec_x', 'vec_y']
X = StandardScaler().fit_transform(df[features])

K_initial = 10 
df['cluster'] = KMeans(n_clusters=K_initial, random_state=42).fit_predict(X)

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