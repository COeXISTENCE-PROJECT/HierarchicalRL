import pandas as pd

# Input files
agents_file = "sa_agents.csv"
coords_file = "saint_arnoult_agents_coords.csv"

agents = pd.read_csv(agents_file)
coords = pd.read_csv(coords_file)

av_ids = agents.loc[agents["kind"] == "AV", "id"].unique()
av_coords = coords[coords["id"].isin(av_ids)].copy()
av_coords = av_coords.sort_values("id")

av_coords.to_csv("sa_AVs_coords.csv", index=False)

print(av_coords)
print(f"Number of AVs: {len(av_ids)}")
print(f"Number of matched coordinate rows: {len(av_coords)}")
