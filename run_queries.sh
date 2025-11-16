#!/bin/bash

# Array of queries
QUERIES=(
    "genes linked to Parkinson's disease"
    "drugs targeting BRCA1 in breast cancer"
    "key metabolic pathways in type 2 diabetes"
)

# Loop through queries and run the Python script
for QUERY in "${QUERIES[@]}"; do
    echo "Running query: $QUERY"
    python -m src.rag_pipeline_langchain --query "$QUERY" --structured
    echo "-------------------------------------------"
done

