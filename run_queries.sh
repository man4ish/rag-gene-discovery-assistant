#!/bin/bash

# Array of queries
QUERIES=(
    "TP53 variants and cancer prognosis"
    "KRAS inhibitors clinical trials"
    "metformin effects on metabolic syndrome"
    "BRCA1 mutations in breast cancer"
    "EGFR-targeted therapies in lung cancer"
    "PD-1 inhibitors and immunotherapy response"
    "APOE variants and Alzheimer's disease risk"
    "VEGF inhibitors in diabetic retinopathy"
    "AKT pathway activation in glioblastoma"
    "statins effect on cardiovascular outcomes"
)


# Loop through queries and run the Python script
for QUERY in "${QUERIES[@]}"; do
    echo "Running query: $QUERY"
    python -m ragbio.pipeline.rag_pipeline --query "$QUERY" --structured
    echo "-------------------------------------------"
done

