# AI For Finance Project
## Overview
This project aims to test out how AI can be applied in finance to automate key workflows. The goal is twofold:
- Design and implement an MVP that ingests financial documents, extracts meaningful insights, and generates a structured output in the form of a memo
- Build a harness to compare various models based on an organized set of evals / a benchmark

## Planned Scope
### Input:
Publicly available financial documents (e.g., SEC filings, earnings reports, legal documents) scraped from the web.

### Features of MVP (document > IC memo):
- Web scraper to obtain a dataset of input documents from SEC website.
- Automated data processing / cleaning.
- Generation of investment memo (executive summary, key data, strengths/weaknesses of investment) through thoughtful prompting (incl. a memo template).
- Benchmarking pipeline involving several key metrics below.

### Benchmarking Metrics:
- Accuracy: are there any terms or numbers in the memo that were not in the inputted document? (_could be evaluated throught a strict semantic matcher or by getting a consensus from answers to yes/no prompts [e.g. average of 1s and 0s]_)
- Completeness: are any key terms missing from the memo? (_ strict semantic matcher or consensus of yes/no answers_)
- Quality of presentation: is the total length/tone appropriate? Is the structure consistent with the template?
- Consistency (intra-memo): does the memo contradict itself anywhere (e.g. listing a stated weakness as a strenght of the investment)? 
- Consistency across runs
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
