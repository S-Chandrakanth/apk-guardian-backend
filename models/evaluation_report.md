# Model evaluation

Train rows: 32  |  Test rows: 8  |  Test classes: {'MALWARE': 5, 'BENIGN': 3}

## Test-set metrics (threshold 0.5, MALWARE = positive class)

| Model | Accuracy | Precision | Recall | F1 | FPR | FNR | PR-AUC |
|---|---|---|---|---|---|---|---|
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 |
| majority_class | 0.6250 | 0.6250 | 1.0000 | 0.7692 | 1.0000 | 0.0000 | 0.6250 |
| logistic_regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 |

## Random Forest confusion matrix

| | Predicted BENIGN | Predicted MALWARE |
|---|---|---|
| **Actually BENIGN** | 3 | 0 |
| **Actually MALWARE** | 0 | 5 |

## Per original category

- Benign (n=3, labelled BENIGN): false_positive_rate = 0.0000
- Malware (n=5, labelled MALWARE): recall = 1.0000

## NEEDS REVIEW band [0.4, 0.6]

0 of 8 test rows fall in the band (0.0000). Metrics on the remaining rows:

`{'n': 8, 'confusion_matrix': {'tn': 3, 'fp': 0, 'fn': 0, 'tp': 5}, 'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'false_positive_rate': 0.0, 'false_negative_rate': 0.0, 'roc_auc': 1.0, 'pr_auc': 1.0}`

## Permutation importance (drop in F1 when shuffled)

- api_calls_count: 0.1861 (std 0.1017)
- strings_count: 0.0556 (std 0.0556)
- permissions_count: 0.0000 (std 0.0000)
- activities_count: 0.0000 (std 0.0000)
- services_count: 0.0000 (std 0.0000)
- receivers_count: 0.0000 (std 0.0000)
- dangerous_permissions: 0.0000 (std 0.0000)
- suspicious_api_calls: 0.0000 (std 0.0000)

## Limits of these results

- These numbers were measured on ONE held-out test split of THIS dataset only.
- They do not estimate performance on other datasets, on newer malware, or at real-world malware/benign ratios.
- Model probabilities are scores, not guaranteed certainty. They are not calibrated.
- Static analysis with eight count features cannot detect every malware sample.
