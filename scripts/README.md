# Scripts

All project entry-point scripts live in this directory.

Run them from the repository root so default paths continue to point at the
project data and outputs. For example:

```bash
python scripts/train_obb.py --label-set labels_0123 --prepare-data-only
python scripts/train_classical_liquid_ml.py --sources bottle-dataset --feature-set full --models all --tasks amount
python scripts/train_tree_classifier.py --features runs/tree_segments/features.csv --algorithm all
```

The scripts compute the repository root as the parent of this directory, so they
can still find `bottleDataset/`, `.dataset_views/`, `LCDTC/`, local model
weights, and `runs/` after the move.
