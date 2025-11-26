#!/usr/bin/pyhton3

"""
structured_drug_kg.py

Neo4j Knowledge Graph Utilities for Structured Drug-Target-Cancer Data.

This module provides functions to:
1. Connect to a Neo4j database.
2. Add drugs, targets, and cancer types as nodes.
3. Create relationships:
   - Drug TARGETS Target (with optional mechanism and PMID)
   - Drug USED_FOR Cancer (with PMID reference)
4. Query drugs targeting a specific protein in a given cancer type.

Functions:
- create_kg_entry(tx, drug, target, cancer, pmid, mechanism=""):
    Adds nodes and relationships for a single drug-target-cancer entry in a transaction.

- add_structured_data_to_kg(structured_data: list):
    Adds multiple structured drug-target-cancer entries to the Neo4j knowledge graph.

- query_drugs_by_target(target_name: str, cancer_name: str) -> list:
    Returns a list of drugs targeting a specified protein in a given cancer type.

Example Usage:
    data = [
        {
            "drug": "Trastuzumab",
            "targets": [
                {"target": "HER2", "cancer": "Breast Cancer", "mechanism": "Inhibits HER2 signaling"}
            ],
            "pmid": "12345678"
        }
    ]
    
    add_structured_data_to_kg(data)
    drugs = query_drugs_by_target("HER2", "Breast Cancer")
    print(drugs)

Configuration:
- NEO4J_URI      : Bolt URI of the Neo4j instance (default: bolt://localhost:7687)
- NEO4J_USER     : Neo4j username
- NEO4J_PASSWORD : Neo4j password

Dependencies:
- neo4j Python driver

Author:
    Manish Kumar
"""


from neo4j import GraphDatabase

# ---------------------------
# Neo4j connection config
# ---------------------------
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "MyNewSecurePassword123"

# ---------------------------
# Neo4j driver
# ---------------------------
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# ---------------------------
# Functions
# ---------------------------
def create_kg_entry(tx, drug, target, cancer, pmid, mechanism=""):
    """
    Add nodes and edges to Neo4j knowledge graph.
    """
    tx.run("""
        MERGE (d:Drug {name: $drug})
        MERGE (t:Target {name: $target})
        MERGE (c:Cancer {name: $cancer})
        MERGE (d)-[:TARGETS {mechanism: $mechanism, pmid: $pmid}]->(t)
        MERGE (d)-[:USED_FOR {pmid: $pmid}]->(c)
    """, drug=drug, target=target, cancer=cancer, pmid=pmid, mechanism=mechanism)

def add_structured_data_to_kg(structured_data: list):
    """
    Add structured drug-target-cancer data to Neo4j.
    
    structured_data: List of dictionaries like:
    [
        {
            "drug": "Trastuzumab",
            "targets": [
                {"target": "HER2", "cancer": "Breast Cancer", "mechanism": "Inhibits HER2 signaling"},
                {"target": "HER2", "cancer": "Gastric Cancer", "mechanism": "Blocks receptor dimerization"}
            ],
            "pmid": "12345678"
        },
        ...
    ]
    """
    with driver.session() as session:
        for entry in structured_data:
            drug = entry.get("drug")
            pmid = entry.get("pmid", "Unknown")
            for t in entry.get("targets", []):
                target = t.get("target", "Unknown")
                cancer = t.get("cancer", "Unknown")
                mechanism = t.get("mechanism", "")
                session.execute_write(create_kg_entry, drug, target, cancer, pmid, mechanism)
    print(f"Added {len(structured_data)} drugs to the knowledge graph.")

def query_drugs_by_target(target_name: str, cancer_name: str) -> list:
    """
    Query Neo4j for drugs targeting a given protein in a cancer type.
    """
    def _query(tx):
        result = tx.run("""
            MATCH (d:Drug)-[:TARGETS]->(t:Target {name: $target})
            MATCH (d)-[:USED_FOR]->(c:Cancer {name: $cancer})
            RETURN d.name AS drug
        """, target=target_name, cancer=cancer_name)
        return [record["drug"] for record in result]

    with driver.session() as session:
        return session.execute_read(_query)  # <-- changed from read_transaction


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    # Example structured input
    data = [
        {
            "drug": "Trastuzumab",
            "targets": [
                {"target": "HER2", "cancer": "Breast Cancer", "mechanism": "Inhibits HER2 signaling"}
            ],
            "pmid": "12345678"
        },
        {
            "drug": "Lapatinib",
            "targets": [
                {"target": "HER2", "cancer": "Breast Cancer", "mechanism": "Tyrosine kinase inhibitor"}
            ],
            "pmid": "23456789"
        }
    ]
    
    # Add to Neo4j
    add_structured_data_to_kg(data)
    
    # Query example
    drugs = query_drugs_by_target("HER2", "Breast Cancer")
    print("Drugs targeting HER2 in Breast Cancer:", drugs)
