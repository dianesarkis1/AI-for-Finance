# AI For Finance Project
## Overview
This project aims to test out how AI can be applied in finance to automate key workflows. The goal is twofold:
- Design and implement an MVP that ingests financial documents, extracts meaningful insights, and generates a structured output in the form of a memo
- Build a harness to compare various models based on an organized set of evals / a benchmark

## Planned Scope
### Input:
Publicly available financial documents (e.g., SEC filings, earnings reports, legal documents) scraped from the web.

### Features of MVP (document > IC memo):
- Web scraper to obtain a dataset of input documents from SEC website
- Automated data processing / cleaning
- Generation of investment memo (executive summary, key data, strengths/weaknesses of investment) through thoughtful prompting (incl. a memo template)
- Benchmarking pipeline involving several key metrics below

### Benchmarking Metrics:
- **Accuracy**: Are there any terms in the memo that were not in the inputted document? (_evaluated throught a strict semantic matcher or by getting a consensus from answers to yes/no prompts [e.g. average of 1s and 0s]_)
- **Completeness**: Are any key terms missing from the memo? (_strict semantic matcher or consensus of yes/no answers_)
- **Quality of presentation**: Is the total length/tone appropriate? Is the structure consistent with the template? (_Total score will be an average of the following subscores: LLM as a judge with rubric e.g. score 1-5 on clarity, conciseness, tone, yes/no answer (1/0 score) to whether the output length is within a given word count range, yes/no to whether structure is in the correct order (parser)_) 
- **Consistency (intra-memo)**: Does the memo contradict itself anywhere (e.g. listing a stated weakness as a strenght of the investment)? (_get consensus from handful of AI models [thoughtful prompt asking to check for any inconsistencies]_)
- **Consistency across runs** (_Summary score for the output of each model call based on an average/sum of the above scores, can then measure variance and worst score of k calls of a given model with a given input_)
- **(TBD: latency / cost-like measures)**

### Folder organization
Data folder: 
- urls.txt has all the url links to the full dataset.
- eval_urls.txt has all the url links to the eval set (this list is pre-set so that it stays the same across iterations/runs)
- cleaned_data.jsonl has the entire dataset after preprocessing/cleaning, which is done by running data_cleaning.py
- eval.jsonl has the eval split of the data
- train.jsonl has the training split
- sample_memo.md has a template memo

Project Scripts folder:
- "main" functions run "exploratory" functions to generate investment memos using several models (used for initial selection of what the baseline [benchmark] model will be).

Project Logs folder: contains logs on each step of the project
