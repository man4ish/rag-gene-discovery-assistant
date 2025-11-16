import os
import glob
import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Path to the output folder
output_folder = "output"
json_files = glob.glob(os.path.join(output_folder, "*.json"))

G = nx.DiGraph()  # Directed graph

# Loop through all JSON files
for json_file in json_files:
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Build graph safely
    for entry in data:
        # Disease associations
        gene = entry.get("gene")
        disease = entry.get("disease")
        if gene and disease:
            G.add_node(gene, type="gene")
            G.add_node(disease, type="disease")
            G.add_edge(gene, disease, relation="associated_with")

        # Pathways
        pathway = entry.get("pathway")
        if gene and pathway:
            G.add_node(gene, type="gene")
            G.add_node(pathway, type="pathway")
            G.add_edge(gene, pathway, relation="part_of")

        # Drugs
        drug = entry.get("drug")
        target = entry.get("target")
        if drug and target:
            G.add_node(drug, type="drug")
            G.add_node(target, type="target")
            G.add_edge(drug, target, relation="targets")

        # Tools
        tool = entry.get("tool")
        if tool and disease:
            G.add_node(tool, type="tool")
            G.add_node(disease, type="disease")
            G.add_edge(tool, disease, relation="used_for")

# Assign node colors based on type
color_map = {
    "gene": "skyblue",
    "disease": "lightgreen",
    "drug": "orange",
    "pathway": "pink",
    "tool": "yellow"
}
node_colors = [color_map.get(G.nodes[n].get('type', 'other'), 'grey') for n in G.nodes()]

# Layout
pos = nx.spring_layout(G, k=0.5, seed=42)

# Draw graph
plt.figure(figsize=(15, 12))
nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=1500, font_size=10, arrows=True)
edge_labels = nx.get_edge_attributes(G, 'relation')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=8)

# Add legend
legend_elements = [
    mpatches.Patch(color='skyblue', label='Gene'),
    mpatches.Patch(color='lightgreen', label='Disease'),
    mpatches.Patch(color='orange', label='Drug'),
    mpatches.Patch(color='pink', label='Pathway'),
    mpatches.Patch(color='yellow', label='Tool')
]
plt.legend(handles=legend_elements, loc='best')

# Save and show
plt.savefig(os.path.join(output_folder, "network_graph.png"), dpi=300, bbox_inches='tight')
plt.show()
