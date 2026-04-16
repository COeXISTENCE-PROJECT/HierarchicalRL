# DATA PREPARATION AND COORDINATE EXTRACTION SCRIPT

# This script acts as a bridge between your simulation's agent data and the physical SUMO road network
# In order to cluster vehicles based on distance, we need their actual X and Y coordinates on the map, not just their street names
import pandas as pd
import ast
import sumolib

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ADJUST THIS PATH IF NEEDED
# chose .txt file with origins and destinations
with open('networks\\ingolstadt_custom\\od_ingolstadt_custom.txt', 'r') as f:
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    data_content = f.read()
    od_dict = ast.literal_eval(data_content)

origins_list = od_dict['origins']
destinations_list = od_dict['destinations']

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ADJUST THIS PATH IF NEEDED
# Chose agents.csv file from your simulation 
agents_df = pd.read_csv('networks\\ingolstadt_custom\\agents.csv')
# Chose .net.xml file from your simulation 
net = sumolib.net.readNet('networks\\ingolstadt_custom\\ingolstadt_custom.net.xml')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

edge_coords = {}
for edge in net.getEdges():
    edge_id = edge.getID()
    # getShape() returns a list of all points that make up the street: [(x1, y1), (x2, y2), ...]
    shape = edge.getShape() 
    
    if shape:
        # Grab the very first point of the street (start) and the very last point (end)
        start_x, start_y = shape[0]
        end_x, end_y = shape[-1]
        
        edge_coords[edge_id] = {
            'start_x': start_x, 'start_y': start_y,
            'end_x': end_x, 'end_y': end_y
        }

# Find the exact X or Y coordinate for a specific agent
def get_coord_for_agent(agent_idx, is_origin, coord_axis):
    try:
        if is_origin:
            edge_id = origins_list[agent_idx]
            return edge_coords[edge_id][f'start_{coord_axis}']
        else:
            edge_id = destinations_list[agent_idx]
            return edge_coords[edge_id][f'end_{coord_axis}']
    except (IndexError, KeyError):
        return None

# Map the internal integer IDs (0, 1, 2) back to the real string IDs (e.g., '315358244')
agents_df['origin_real_id'] = [origins_list[i] for i in agents_df['origin']]
agents_df['dest_real_id'] = [destinations_list[i] for i in agents_df['destination']]

agents_df['origin_x'] = agents_df['origin'].apply(lambda idx: get_coord_for_agent(idx, True, 'x'))
agents_df['origin_y'] = agents_df['origin'].apply(lambda idx: get_coord_for_agent(idx, True, 'y'))
agents_df['dest_x'] = agents_df['destination'].apply(lambda idx: get_coord_for_agent(idx, False, 'x'))
agents_df['dest_y'] = agents_df['destination'].apply(lambda idx: get_coord_for_agent(idx, False, 'y'))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ADJUST THIS PATH IF NEEDED
agents_df.to_csv('clustering_ideas\\ingolstadt_custom_clustering\\ingolstadt_custom_agents_coords.csv', index=False)
print("Saved to ingolstadt_custom_agents_coords.csv")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print(agents_df.head())