# AI For Finance Project
## Overview
This project aims to test out how AI can be applied in finance to automate key workflows. The goal is to:
- design and implement an MVP that ingests financial documents, extracts meaningful insights, and generates a structured output in the form of a memo
- build a harness to compare various models based on an organized set of evals / a benchmark

## Planned MVP Scope
### Planned Input:
Public financial documents (e.g., SEC filings, earnings reports, legal documents) scraped from the web.

### Planned Features:
Data ingestion and parsing.
Key metric extraction (e.g., revenue, EBITDA, guidance changes).
Automated generation of financial memo (executive summary, key data, strengths/weaknesses of investment).
Benchmarking pipeline involving several key metrics below. (accuracy, completeness, consistency across runs, quality of presentation...)

### Planned Benchmarks:
Accuracy: are there any terms or numbers in the memo that were not in the inputted document? 
Completeness: are any key terms missing from the memo?
Quality of presentation: is the total length/tone appropriate? Is the structure consistent with the template?
Consistency (intra-memo): does the memo contradict itself anywhere (e.g. listing a stated weakness as a strenght of the investment)? 
Consistency across runs
(TBD: latency / cost-like measures)
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
