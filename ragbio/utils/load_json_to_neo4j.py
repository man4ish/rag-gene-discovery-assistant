# load_json_to_neo4j_safe.py
import json
from neo4j import GraphDatabase

# -------------------------
# Neo4j connection
# -------------------------
uri = "bolt://localhost:7687"  # Docker Neo4j
user = "neo4j"
password = "@#DataScientist007"  # Replace with your Neo4j password

driver = GraphDatabase.driver(uri, auth=(user, password))

# -------------------------
# Load JSON
# -------------------------
json_file = "output/which_genes_are_linked_to_oxidative_stress_in_alzheimer_s_disease__output.json"
with open(json_file) as f:
    data = json.load(f)

# -------------------------
# Function to create nodes and relationships
# -------------------------
def create_graph(tx, entry):
    try:
        # Create nodes if non-null
        if entry.get("drug"):
            tx.run("MERGE (d:Drug {name:$drug})", drug=entry["drug"])
        if entry.get("gene"):
            tx.run("MERGE (g:Gene {name:$gene})", gene=entry["gene"])
        if entry.get("target"):
            tx.run("MERGE (t:Target {name:$target})", target=entry["target"])
        if entry.get("cancer"):
            tx.run("MERGE (c:Cancer {name:$cancer})", cancer=entry["cancer"])
        if entry.get("disease"):
            tx.run("MERGE (d:Disease {name:$disease})", disease=entry["disease"])
        if entry.get("pathway"):
            for p in entry["pathway"].split(","):
                tx.run("MERGE (pw:Pathway {name:$pathway})", pathway=p.strip())

        # Relationships
        if entry.get("drug") and entry.get("target"):
            tx.run("""
                MATCH (d:Drug {name:$drug}), (t:Target {name:$target})
                MERGE (d)-[:TARGETS]->(t)
            """, drug=entry["drug"], target=entry["target"])

        if entry.get("gene") and entry.get("target"):
            tx.run("""
                MATCH (g:Gene {name:$gene}), (t:Target {name:$target})
                MERGE (g)-[:ASSOCIATED_WITH]->(t)
            """, gene=entry["gene"], target=entry["target"])

        if entry.get("gene") and entry.get("cancer_association"):
            for c in entry["cancer_association"].split(","):
                tx.run("""
                    MERGE (c:Cancer {name:$cancer})
                    WITH c
                    MATCH (g:Gene {name:$gene})
                    MERGE (g)-[:ASSOCIATED_WITH]->(c)
                """, gene=entry["gene"], cancer=c.strip())

        if entry.get("target") and entry.get("pathway"):
            for pw in entry["pathway"].split(","):
                tx.run("""
                    MATCH (t:Target {name:$target}), (pw:Pathway {name:$pathway})
                    MERGE (t)-[:PARTICIPATES_IN]->(pw)
                """, target=entry["target"], pathway=pw.strip())

    except Exception as e:
        print(f"Error inserting entry {entry}: {e}")

# -------------------------
# Insert all entries into Neo4j
# -------------------------
with driver.session(database="neo4j") as session:
    for entry in data:
        session.execute_write(create_graph, entry)

print("Graph loaded into Neo4j! Open Neo4j Browser and run `MATCH (n) RETURN n LIMIT 25` to see the nodes.")
