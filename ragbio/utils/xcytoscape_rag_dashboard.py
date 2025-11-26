import streamlit as st
import json

# Load JSON
json_file = "output/which_genes_are_linked_to_oxidative_stress_in_alzheimer_s_disease__output.json"
with open(json_file, "r") as f:
    data = json.load(f)

nodes_dict = {}  # use dict to avoid duplicates
edges = []

for entry in data:
    target = entry.get("target")
    if target:
        nodes_dict[target] = {"data": {"id": target, "label": target, "type": "target"}}
    
    drug = entry.get("drug")
    if drug:
        nodes_dict[drug] = {"data": {"id": drug, "label": drug, "type": "drug"}}
        if target:
            edges.append({"data": {"source": target, "target": drug}})
    
    disease = entry.get("disease") or entry.get("cancer") or entry.get("cancer_association")
    if disease:
        for d in disease.split(","):
            d = d.strip()
            nodes_dict[d] = {"data": {"id": d, "label": d, "type": "disease"}}
            if target:
                edges.append({"data": {"source": target, "target": d}})
    
    pmid = entry.get("pmid")
    if pmid:
        nodes_dict[pmid] = {"data": {"id": pmid, "label": pmid, "type": "pmid"}}
        if target:
            edges.append({"data": {"source": target, "target": pmid}})

nodes = list(nodes_dict.values())
cy_elements = nodes + edges

# Render Cytoscape.js
st.title("RAG Network: Oxidative Stress in Alzheimer's Disease")

st.components.v1.html(f"""
<html>
<head>
<script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.25.0/cytoscape.min.js"></script>
</head>
<body>
<div id="cy" style="width:100%; height:650px; border:1px solid #ccc;"></div>
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
""", height=1050)
