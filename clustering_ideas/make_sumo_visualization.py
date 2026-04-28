# SUMO VISUALIZATION GENERATOR SCRIPT

# This script takes the clustered agent data (where each vehicle has been assigned to a specific group) 
# and converts it into a route file (.rou.xml) that the SUMO traffic simulator can read

import pandas as pd

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ADJUST THIS PATH IF NEEDED
# df = pd.read_csv('clustering_ideas\\ingolstadt_custom_clustering\\agents_clustered_k_prototypes_shortest_path.csv')
#df = pd.read_csv('clustering_ideas\\ingolstadt_custom_clustering\\agents_clustered_similarity_measure_shortest_path.csv')
# df = pd.read_csv('clustering_ideas\\ingolstadt_custom_clustering\\agents_clustered_with_spatial.csv')
# df = pd.read_csv('clustering_ideas\\saint_arnoult_clustering\\agents_clustered_with_spatiotemporal.csv')
# df = pd.read_csv('clustering_ideas\\provins_clustering\\agents_clustered_k_prototypes_shortest_path.csv')
# df = pd.read_csv('clustering_ideas\\ingolstadt_custom_clustering\\agents_clustered_by_path.csv')
df = pd.read_csv('clustering_ideas\\provins_clustering\\agents_clustered_by_path.csv')
#df = pd.read_csv('clustering_ideas\\ingolstadt_custom_clustering\\agents_clustered_with_spatiotemporal.csv')
# df = pd.read_csv('clustering_ideas\\ingolstadt_custom_clustering\\agents_clustered_with_similarity_measure.csv')
# df = pd.read_csv('clustering_ideas\\ingolstadt_custom_clustering\\agents_clustered_with_k_prototypes.csv')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

df = df.sort_values(by='start_time')

# 20 possible clusters, each with a unique color (R,G,B format)
cluster_colors = {
    0: "255,0,0",       # Red
    1: "0,255,0",       # Green
    2: "0,0,255",       # Blue
    3: "255,255,0",     # Yellow
    4: "255,0,255",     # Fuchsia (Magenta)
    5: "0,255,255",     # Cyan (Light Blue)
    6: "255,165,0",     # Orange
    7: "128,0,128",     # Purple
    8: "255,20,147",    # Deep Pink
    9: "50,205,50",     # Lime Green
    10: "0,128,128",    # Teal
    11: "165,42,42",    # Brown
    12: "0,0,128",      # Navy Blue
    13: "255,215,0",    # Gold
    14: "255,127,80",   # Coral
    15: "173,216,230",  # Light Blue
    16: "250,128,114",  # Salmon
    17: "154,205,50",   # Yellow Green
    18: "218,112,214",  # Orchid
    19: "210,105,30"    # Chocolate
}

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ADJUST THIS PATH IF NEEDED
#with open('clustering_ideas\\ingolstadt_custom_clustering\\visualization_similarity_measure_shortest_path.rou.xml', 'w') as f:
#with open('clustering_ideas\\ingolstadt_custom_clustering\\visualization_with_spatiotemporal.rou.xml', 'w') as f:
# with open('clustering_ideas\\ingolstadt_custom_clustering\\visualization_with_spatial.rou.xml', 'w') as f:
# with open('clustering_ideas\\saint_arnoult_clustering\\visualization_with_spatiotemporal.rou.xml', 'w') as f:
# with open('clustering_ideas\\provins_clustering\\visualization_k_prototypes_shortest_path.rou.xml', 'w') as f:
# with open('clustering_ideas\\ingolstadt_custom_clustering\\visualization_by_path.rou.xml', 'w') as f:
with open('clustering_ideas\\provins_clustering\\visualization_by_path.rou.xml', 'w') as f:

# with open('clustering_ideas\\ingolstadt_custom_clustering\\clustered_with_time_k_prototypes.rou.xml', 'w') as f:
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    f.write('<routes>\n')
    f.write('    <vType id="klaster_car" vClass="passenger"/>\n\n')
    
    for index, row in df.iterrows():
        veh_id = f"auto_{index}"
        depart_time = int(row['start_time'])
        
        origin_edge = row['origin_real_id']
        dest_edge = row['dest_real_id']
        
        cluster_id = int(row['cluster'])
        color = cluster_colors.get(cluster_id, "255,255,255")
        
        f.write(f'    <trip id="{veh_id}" type="klaster_car" depart="{depart_time}" '
                f'from="{origin_edge}" to="{dest_edge}" color="{color}"/>\n')

    f.write('</routes>\n')

# print("Generated file visualization_k_prototypes.rou.xml")
# print("Generated file visualization_k_prototypes_shortest_path.rou.xml")
print("Generated file visualization_by_path.rou.xml")
# print("Generated file visualization_with_spatial.rou.xml")
#print("Generated file visualization_with_spatiotemporal.rou.xml")