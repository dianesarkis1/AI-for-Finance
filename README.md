# AI For Finance Project
## Overview
This project explores how AI can be applied in finance to automate key workflows. The goals are twofold:
- Build an MVP pipeline that ingests financial documents, extracts meaningful insights, and generates a structured output in the form of an investment memo
- Develop an evaluation harness to evaluate models (gpt-5, claude opus 4, gemini 2.5 pro...) on a consistent set of metrics, while testing whether prompt optimization techniques (few shot examples, iterative refinement...) can systematically improve performance

## Initial Scope of Project
My original plan was to fine-tune a model on a dataset of financial documents and evaluate whether the fine-tuned model generated better investment memos. However, I identified several challenges that made this approach less viable:
- Limited compute resources
- Lack of labeled data: While it was straightforward to scrape a large corpus of credit agreements, creating high-quality, paired input–output labels (investment memos) at scale would have required significant manual effort.
- Risk of overfitting: Training on a small “gold standard” labeled dataset might have produced outputs too closely tied to that dataset’s style, without improving generalization.
- Limited real-world applicability: Fine-tuning at this stage would have been premature. Instead of customizing a model, I thought it would be more valuable to establish a pipeline that demonstrates end-to-end utility and produces outputs evaluable against objective criteria.

Instead of fine-tuning, I therefore focused on building a practical, evaluation-driven system.

## Planned Scope
### Input:
Publicly available financial documents scraped from the SEC EDGAR database, specifically credit agreements. These agreements are filed by large, public companies (2023-onward); contain a common set of transaction details (interest rates, maturities, covenants, collateral, etc.); and share broadly similar structures across issuers, making them a good testbed for memo generation and evaluation.

### Features of MVP (document > IC memo):
- Web scraper to obtain a dataset of input documents from SEC website (credit agreements)
- Preprocessing pipeline: convert raw filings into clean plain text, store in jsonl format, splits (train/eval) are made deterministically via a URL hash, ensuring reproducibility across iterations.
- Generation of investment memo (executive summary, key data, strengths/weaknesses of investment) through thoughtful prompting (incl. a memo template)
- (Then: benchmarking harness involving several key metrics below, prompt optimizing)

### Benchmarking Metrics:
- **Accuracy**: Are there any key financial terms in the memo that were not in the inputted document?
  - _Implementation: LLM consensus approach with yes/no prompts across 3 models (gpt-5, claude-sonnet-4-20250514, gemini-2.5-pro)_
  - _Rationale: I considered a strict semantic matcher but rejected it because: (1) credit agreements don't all follow the exact same format, making it difficult to programmatically identify what constitutes "key" terms, (2) strict matching is brittle to paraphrasing (e.g., "5.25% interest rate" vs "525 basis points"), and (3) requires exact entity extraction which is error-prone. LLM consensus is preferred because it: (1) understands financial context and can distinguish key terms from filler words, (2) handles synonyms and reformulations naturally, (3) is flexible across document formats, and (4) consensus across multiple models reduces individual model biases._
- **Completeness**: Are any key terms missing from the memo? (_strict semantic matcher or consensus of yes/no answers_)
- **Quality of presentation**: Is the total length/tone appropriate? Is the structure consistent with the template...? (_Total score will be an average of the following subscores: LLM as a judge with rubric e.g. score 1-5 on clarity, tone, yes/no answer (1/0 score) to whether the output length is within a given word count range, yes/no to whether structure is in the correct order (parser)_) 
- **Consistency (intra-memo)**: Does the memo contradict itself anywhere (e.g. listing a stated weakness as a strenght of the investment)? (_get consensus from handful of AI models [thoughtful prompt asking to check for any inconsistencies]_)
- **Consistency across runs** (_Summary score for the output of each model call based on an average/sum of the above scores, can then measure variance and worst score of k calls of a given model with a given input_)
- **(TBD: latency / cost-like measures)**

### Folder Organization
Data folder: 
- urls.txt has all the url links to the full dataset.
- eval_urls.txt has all the url links to the eval set (this list is pre-set so that it stays the same across iterations/runs)
- cleaned_data.jsonl has the entire dataset after preprocessing/cleaning, which is done by running data_cleaning.py
- eval.jsonl has the eval split of the data
- train.jsonl has the training split
- sample_memo.md has a template memo

Project Scripts folder:
- "main_exploratory" functions run "model_run" functions to generate investment memos using several models (used for initial selection of what the baseline [benchmark] model will be).

Evals folder:
- **metrics.py**: Core evaluation functions implementing all 5 metrics:
  - `evaluate_accuracy()`: Detects hallucinated financial terms using LLM consensus (3 models vote YES/NO)
  - `evaluate_completeness()`: Detects missing key terms using LLM consensus
  - `evaluate_consistency()`: Detects intra-memo contradictions using LLM consensus with JSON output
  - `evaluate_quality()`: Scores presentation quality across 4 dimensions (clarity, tone, length, structure) on 0-100 scale
  - `calculate_summary_score()`: Aggregates all 4 metrics into single summary score (0-100) with configurable weights
- **evaluator.py**: Main evaluation harness with two key functions:
  - `evaluate_memo()`: Runs all 4 metrics on a single memo and returns summary score
  - `worst_at_k()`: Generates k memos from same input (via model_run.py) and returns worst-case score to test consistency across runs
- **utils.py**: Helper functions for LLM-as-judge evaluation:
  - `call_llm_for_eval()`: Unified interface to call OpenAI, Anthropic, or Google models for evaluation
  - Handles API differences, retries, and response parsing
- **helper_tests/**: Test scripts for each metric that run on exploratory outputs as sanity checks:
  - `test_accuracy.py`: Tests accuracy metric on record_01
  - `test_completeness.py`: Tests completeness metric on record_01
  - `test_consistency.py`: Tests consistency metric on record_01
  - `test_quality.py`: Tests quality metric (all 4 sub-dimensions) on record_01
  - `test_summary_score.py`: Runs all 4 metrics and tests aggregation into summary score
  - All tests include rate limit handling (35s delays for Gemini free tier) and save results to JSON

### Future Avenues for Exploration:
- Human-in-the-loop feedback: adding an interface to collect user ratings on generated memos
- Transaction-specific rubrics: inspired by the HealthBench paper, design bespoke evaluation criteria tailored to each credit agreement (e.g., covenant depth, collateral clarity).
- Model generalization: test transferability by applying the pipeline to adjacent but slightly document types (e.g. legal agreements such as indentures and term sheets that contain similar information on a credit transaction but are structured differently).
- RAG: explore whether augmenting LLMs with an external knowledge base (on the company, similar transactions, or other) improves performance.
