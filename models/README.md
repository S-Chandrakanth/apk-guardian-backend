# models/

Trained model artifacts are written here by `ml_training/train_model.py`.

No model exists yet. Nothing in this folder is pre-trained or fabricated.

Expected outputs after a real training run:

- `rf_baseline.joblib` - the trained Random Forest pipeline
- `feature_schema.json` - ordered feature names and extractor version
- `training_report.json` - metrics from the held-out test set, split method, dataset hash

Model files are git-ignored. Re-create them by running the training pipeline on a validated dataset.
