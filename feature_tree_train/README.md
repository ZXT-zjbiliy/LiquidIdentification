# Legacy Feature-Tree Scripts

This folder keeps the original teammate scripts for reference. The maintained project entry point is:

```bash
python scripts/train_classical_liquid_ml.py
```

Renamed legacy files:

```text
train_binary_full_features.py              binary full-feature extraction and RF/SVM training
compare_classical_models.py                model comparison from an existing extracted_features.csv
iterative_binary_split.py                  iterative binary split adjustment experiment
train_lcdtc_and_small_classical_ml.py      LCDTC plus small-dataset binary/four-class experiment
```

The legacy scripts still contain the original absolute paths and experiment assumptions. Use them only to trace the old method; use `scripts/train_classical_liquid_ml.py` for new runs.
