import streamlit as st
from py2neo import Graph, Node, Relationship
import json
import os

# ----------------------------
# Neo4j connection
# ----------------------------
uri = "bolt://localhost:7687"
user = "neo4j"
password = "@#DataScientist007"
graph = Graph(uri, auth=(user, password))

# ----------------------------
# Function to load JSON into Neo4j incrementally
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

# ----------------------------
# Function to fetch Cytoscape elements from Neo4j
# ----------------------------
def fetch_cy_elements():
    query = "MATCH (n)-[r]->(m) RETURN n, r, m"
    results = graph.run(query).data()

    nodes_dict = {}
    edges = []

    for record in results:
        n = record['n']
        m = record['m']
        r = record['r']

        for node in [n, m]:
            if node['name'] not in nodes_dict:
                nodes_dict[node['name']] = {
                    "data": {
                        "id": node['name'],
                        "label": node['name'],
                        "type": list(node.labels)[0].lower()
                    }
                }

        edges.append({
            "data": {
                "source": n['name'],
                "target": m['name'],
                "label": r.__class__.__name__  # optional
            }
        })

    return list(nodes_dict.values()) + edges

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("RAG Network: Oxidative Stress in Alzheimer's Disease")

json_file = "output/which_genes_are_linked_to_oxidative_stress_in_alzheimer_s_disease__output.json"

# Refresh button
if st.button("Refresh Graph"):
    load_json_to_neo4j(json_file)
    st.success("Graph updated from JSON!")

# Fetch elements from Neo4j
cy_elements = fetch_cy_elements()

# Render Cytoscape.js
st.components.v1.html(f"""
<html>
<head>
<script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.25.0/cytoscape.min.js"></script>
</head>
<body>
<div id="cy" style="width:100%; height:850px; border:1px solid #ccc;"></div>
<script>
var cy = cytoscape({{
  container: document.getElementById('cy'),
  elements: {json.dumps(cy_elements)},
  layout: {{ name: 'cose', animate: true }},
  style: [
    {{
      selector: 'node[type="target"]',
      style: {{ 'background-color': '#61bffc', 'label': 'data(label)' }}
    }},
    {{
      selector: 'node[type="drug"]',
      style: {{ 'background-color': '#ff6666', 'label': 'data(label)' }}
    }},
    {{
      selector: 'node[type="disease"]',
      style: {{ 'background-color': '#ffcc66', 'label': 'data(label)' }}
    }},
    {{
      selector: 'node[type="pmid"]',
      style: {{ 'background-color': '#99cc99', 'label': 'data(label)' }}
    }},
    {{
      selector: 'edge',
      style: {{
        'width': 2,
        'line-color': '#9dbaea',
        'target-arrow-color': '#9dbaea',
        'target-arrow-shape': 'triangle',
        'curve-style': 'bezier'
      }}
    }}
  ]
}});
</script>
</body>
</html>
""", height=950)
