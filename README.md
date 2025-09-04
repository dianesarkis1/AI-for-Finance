# AI For Finance Project
## Overview
This project aims to demonstrate how AI can be applied in finance to automate key workflows. The goal is to design and implement an MVP that ingests financial data/documents, extracts meaningful insights, and generates structured "memo-like" outputs.

## Planned MVP Scope
### Planned Input:
Public financial documents (e.g., SEC filings, earnings reports, legal documents) or synthetic financial datasets.

### Planned Features:
Data ingestion and parsing.
Key metric extraction (e.g., revenue, EBITDA, guidance changes).
Automated generation of draft financial summaries.
Benchmarking for accuracy, cost, and latency.

### Planned Output: 
A draft financial memo with executive summary, key data, strengths/weaknesses of investment.

### Planned Benchmarks:
Extraction Accuracy: % of correct metrics extracted.
Summary Quality: Manual evaluation or LLM-as-judge scoring.
Cost & Latency: Measured per-document.
Optional Robustness: Sensitivity to noisy or incomplete data.

### Folder organization
Data folder: 
- urls.txt has all the url links to the entire dataset.
- eval_urls.txt has all the url links to the eval set (this list is pre-set so that it stays the same across iterations/runs)
- cleaned_data.jsonl has the entire dataset after preprocessing/cleaning, which is done by running data_cleaning.py
- eval.jsonl has the eval split of the data
- train.jsonl has the training split
- sample_memo.md has a template memo