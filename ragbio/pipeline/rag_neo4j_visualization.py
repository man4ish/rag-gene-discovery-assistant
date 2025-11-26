#!/usr/bin/pyhton3

"""
rag_neo4j_streamlit.py

A Streamlit-based interactive application to visualize and explore 
gene-target-drug-disease networks using Neo4j as the backend graph database.

This module provides functionality to:
1. Connect to a local Neo4j database.
2. Load JSON files containing target-drug-disease-PMID relationships into Neo4j.
3. Fetch nodes and edges from Neo4j with optional filtering by node type or search term.
4. Render an interactive Cytoscape network visualization in a Streamlit UI.

JSON Input Format:
Each JSON file should contain a list of dictionaries with the following optional keys:
- "target" : str, the gene or target name
- "drug" : str, drug interacting with the target
- "disease" / "cancer" / "cancer_association" : str, associated diseases (comma-separated)
- "pmid" : str, PubMed ID referencing the association

Neo4j Node Labels:
- Target
- Drug
- Disease
- PMID

Neo4j Relationships:
- Target INTERACTS_WITH Drug
- Target ASSOCIATED_WITH Disease
- Target CITED_IN PMID

Streamlit Features:
- Node type selection filter (Target, Drug, Disease, PMID)
- Search by node name (partial match, case-insensitive)
- Button to load all JSON files from a specified output directory into Neo4j
- Interactive Cytoscape graph rendering with color-coded nodes

Dependencies:
- streamlit
- py2neo
- json
- os
- glob
- Cytoscape.js (via CDN in HTML component)

Usage:
    streamlit run rag_neo4j_streamlit.py

Author:
    Manish Kumar
"""

import streamlit as st
from py2neo import Graph, Node, Relationship
import json
import os
import glob

# ----------------------------
# Neo4j connection
# ----------------------------
uri = "bolt://localhost:7687"
user = "neo4j"
password = "@#DataScientist007"
graph = Graph(uri, auth=(user, password))

# ----------------------------
# Configuration
# ----------------------------
OUTPUT_DIR = "output"  # Folder containing JSON files

# ----------------------------
# Load all JSON files into Neo4j
# ----------------------------
def load_json_to_neo4j(json_file):
    if not os.path.exists(json_file):
        st.warning(f"JSON file not found: {json_file}")
        return
    with open(json_file, "r") as f:
        data = json.load(f)
    for entry in data:
        target_name = entry.get("target")
        if not target_name:
            continue
        target_node = Node("Target", name=target_name)
        graph.merge(target_node, "Target", "name")
        # Drug
        drug_name = entry.get("drug")
        if drug_name:
            drug_node = Node("Drug", name=drug_name)
            graph.merge(drug_node, "Drug", "name")
            graph.merge(Relationship(target_node, "INTERACTS_WITH", drug_node))
        # Disease / Cancer
        disease = entry.get("disease") or entry.get("cancer") or entry.get("cancer_association")
        if disease:
            for d in disease.split(","):
                d = d.strip()
                disease_node = Node("Disease", name=d)
                graph.merge(disease_node, "Disease", "name")
                graph.merge(Relationship(target_node, "ASSOCIATED_WITH", disease_node))
        # PMID
        pmid = entry.get("pmid")
        if pmid:
            pmid_node = Node("PMID", name=pmid)
            graph.merge(pmid_node, "PMID", "name")
            graph.merge(Relationship(target_node, "CITED_IN", pmid_node))

def load_all_jsons_to_neo4j(output_dir=OUTPUT_DIR):
    json_files = glob.glob(os.path.join(output_dir, "*.json"))
    if not json_files:
        st.warning(f"No JSON files found in {output_dir}")
        return
    for jf in json_files:
        load_json_to_neo4j(jf)
    st.success(f"Loaded {len(json_files)} JSON files into Neo4j!")

# ----------------------------
# Fetch nodes and edges from Neo4j
# ----------------------------
def fetch_cy_elements(node_types=None, search=None):
    # Base query
    if search:
        # Match nodes connected to the search term
        query = f"""
        MATCH (n)-[r]-(m)
        WHERE toLower(n.name) CONTAINS "{search.lower()}" 
           OR toLower(m.name) CONTAINS "{search.lower()}"
        RETURN n, r, m
        """
    else:
        query = "MATCH (n)-[r]->(m) RETURN n, r, m"

    results = graph.run(query).data()
    nodes_dict = {}
    edges = []

    for record in results:
        n = record['n']
        m = record['m']
        r = record['r']

        for node in [n, m]:
            node_type = list(node.labels)[0].lower()
            if node_types and node_type not in node_types:
                continue
            if node['name'] not in nodes_dict:
                nodes_dict[node['name']] = {
                    "data": {"id": node['name'], "label": node['name'], "type": node_type}
                }

        if n['name'] in nodes_dict and m['name'] in nodes_dict:
            edges.append({
                "data": {"source": n['name'], "target": m['name'], "label": r.__class__.__name__}
            })

    return list(nodes_dict.values()) + edges

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("RAG Network Visualization with Search")

# Node type filter
all_node_types = ["target", "drug", "disease", "pmid"]
selected_types = st.multiselect("Select node types to display:", options=all_node_types, default=all_node_types)

# Search box
search_term = st.text_input("Search target, drug, or disease:", "")

# Load JSONs button
if st.button("Refresh Graph from all JSONs"):
    load_all_jsons_to_neo4j(OUTPUT_DIR)

# Fetch filtered elements
cy_elements = fetch_cy_elements(node_types=selected_types, search=search_term)

# Render Cytoscape
st.components.v1.html(f"""
<html>
<head>
<script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.25.0/cytoscape.min.js"></script>
</head>
<body>
<div id="cy" style="width:100%; height:850px; border:1px solid #ccc"></div>
<script>
var cy = cytoscape({{
  container: document.getElementById('cy'),
  elements: {json.dumps(cy_elements)},
  layout: {{ name: 'cose', animate: true }},
  style: [
    {{ selector: 'node[type="target"]', style: {{ 'background-color': '#61bffc', 'label': 'data(label)' }} }},
    {{ selector: 'node[type="drug"]', style: {{ 'background-color': '#ff6666', 'label': 'data(label)' }} }},
    {{ selector: 'node[type="disease"]', style: {{ 'background-color': '#ffcc66', 'label': 'data(label)' }} }},
    {{ selector: 'node[type="pmid"]', style: {{ 'background-color': '#99cc99', 'label': 'data(label)' }} }},
    {{ selector: 'edge', style: {{ 'width': 2, 'line-color': '#9dbaea', 'target-arrow-color': '#9dbaea', 'target-arrow-shape': 'triangle', 'curve-style': 'bezier' }} }}
  ]
}});
</script>
</body>
</html>
""", height=950)
