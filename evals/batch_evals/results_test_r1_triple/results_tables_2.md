# Comprehensive Evaluation Results (Version 2)

**Iterative Refinement: 2 rounds**

## Summary Statistics

| Metric                | Round 0 | Round 1 |
| --------------------- | ------- | ------- |
| Total Memos Evaluated | 3       | 3       |
| Total Evaluations     | 9       | 9       |
| Total Quality Scores  | 36      | 36      |
| Mean Score            | 60.24   | 57.46   |
| Median Score          | 68.94   | 52.38   |
| Min Score             | 42.77   | 51.25   |
| Max Score             | 69.00   | 68.75   |
| Std Dev               | 15.13   | 9.79    |

## Statistics by Evaluator

| Evaluator                | Count | Mean - Round 0 | Mean - Round 1 | Median - Round 0 | Median - Round 1 |
| ------------------------ | ----- | -------------- | -------------- | ---------------- | ---------------- |
| claude-sonnet-4-20250514 | 3     | 67.06          | 75.21          | 67.69            | 67.69            |
| gemini-2.5-pro           | 3     | 53.25          | 45.06          | 45.25            | 45.00            |
| gpt-5                    | 3     | 60.39          | 52.11          | 69.06            | 44.44            |

## Statistics by Metric

| Metric            | Count | Mean/% - R0   | Mean/% - R1   | Median/Details - R0 | Median/Details - R1 |
| ----------------- | ----- | ------------- | ------------- | ------------------- | ------------------- |
| quality_clarity   | 9     | 84.22         | 85.33         | 84.00               | 86.00               |
| quality_length    | 9     | 79.44         | 81.11         | 85.00               | 86.00               |
| quality_structure | 9     | 43.33         | 41.11         | 45.00               | 40.00               |
| quality_tone      | 9     | 90.11         | 89.56         | 93.00               | 88.00               |
| accuracy          | 9     | 11.1% halluc. | 22.2% halluc. | 1/9                 | 2/9                 |
| completeness      | 9     | 88.9% incomp. | 77.8% incomp. | 8/9                 | 7/9                 |
| consistency       | 9     | 33.3% issues  | 44.4% issues  | 3/9                 | 4/9                 |

## Results by Index (Summary Scores)

| Index | Overall - R0 | Overall - R1 | Gpt - R0 | Gpt - R1 | Claude - R0 | Claude - R1 | Gemini - R0 | Gemini - R1 |
| ----- | ------------ | ------------ | -------- | -------- | ----------- | ----------- | ----------- | ----------- |
| 0     | 69.00        | 68.75        | 69.06    | 68.88    | 92.69       | 67.50       | 45.25       | 69.88       |
| 1     | 68.94        | 52.38        | 69.12    | 44.44    | 67.69       | 67.69       | 70.00       | 45.00       |
| 2     | 42.77        | 51.25        | 43.00    | 43.00    | 40.81       | 90.44       | 44.50       | 20.31       |

