from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import csv
import json

from train_obb import DEFAULT_DATASET, LABEL_SET_ALIASES, LABEL_SETS


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LCDTC_ROOT = ROOT / "LCDTC"
DEFAULT_OUTPUT = ROOT / "runs" / "classical_liquid_ml"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
METADATA_COLUMNS = [
    "source",
    "split",
    "image_name",
    "crop_source",
    "class_id",
    "class_name",
    "has_liquid",
    "amount_label",
]
MODEL_CHOICES = ("decision-tree", "random-forest", "gradient-boosting", "knn", "svm", "xgboost")
TASK_CHOICES = ("binary", "amount")

cv2 = None
joblib = None
np = None
DecisionTreeClassifier = None
GradientBoostingClassifier = None
RandomForestClassifier = None
accuracy_score = None
classification_report = None
confusion_matrix = None
KNeighborsClassifier = None
Pipeline = None
StandardScaler = None
SVC = None
export_text = None


def import_runtime_dependencies():
    global cv2, joblib, np
    global DecisionTreeClassifier, GradientBoostingClassifier, RandomForestClassifier
    global accuracy_score, classification_report, confusion_matrix
    global KNeighborsClassifier, Pipeline, StandardScaler, SVC, export_text

    if cv2 is not None:
        return

    try:
        import cv2 as cv2_module
        import joblib as joblib_module
        import numpy as np_module
        from sklearn.ensemble import GradientBoostingClassifier as gradient_boosting_classifier
        from sklearn.ensemble import RandomForestClassifier as random_forest_classifier
        from sklearn.metrics import accuracy_score as accuracy_score_function
        from sklearn.metrics import classification_report as classification_report_function
        from sklearn.metrics import confusion_matrix as confusion_matrix_function
        from sklearn.neighbors import KNeighborsClassifier as knn_classifier
        from sklearn.pipeline import Pipeline as sklearn_pipeline
        from sklearn.preprocessing import StandardScaler as standard_scaler
        from sklearn.svm import SVC as svc_classifier
        from sklearn.tree import DecisionTreeClassifier as decision_tree_classifier
        from sklearn.tree import export_text as export_tree_text
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing classical-ML dependencies. Install them with: pip install -r requirements.txt"
        ) from exc

    cv2 = cv2_module
    joblib = joblib_module
    np = np_module
    DecisionTreeClassifier = decision_tree_classifier
    GradientBoostingClassifier = gradient_boosting_classifier
    RandomForestClassifier = random_forest_classifier
    accuracy_score = accuracy_score_function
    classification_report = classification_report_function
    confusion_matrix = confusion_matrix_function
    KNeighborsClassifier = knn_classifier
    Pipeline = sklearn_pipeline
    StandardScaler = standard_scaler
    SVC = svc_classifier
    export_text = export_tree_text


def parse_args():
    parser = ArgumentParser(
        description=(
            "Train classical machine-learning models for bottle liquid recognition. "
            "This is the cleaned replacement for the feature_tree_train scripts."
        )
    )
    parser.add_argument(
        "--features",
        default=None,
        help="Existing feature CSV. If set, feature extraction is skipped.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=("lcdtc", "bottle-dataset", "labels-picture"),
        default=("lcdtc", "bottle-dataset"),
        help="Datasets used to build the feature table.",
    )
    parser.add_argument("--lcdtc-root", default=str(DEFAULT_LCDTC_ROOT), help="LCDTC dataset root.")
    parser.add_argument("--bottle-dataset", default=str(DEFAULT_DATASET), help="Small bottleDataset root.")
    parser.add_argument(
        "--label-set",
        choices=sorted(LABEL_SET_ALIASES),
        default="labels_0123",
        help="Small-dataset label set used for amount labels.",
    )
    parser.add_argument(
        "--labels-picture-root",
        default=None,
        help="Optional legacy labels_picture root containing crop_info.csv and cropped images.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output directory.")
    parser.add_argument("--feature-set", choices=("light", "full"), default="full", help="Feature set size.")
    parser.add_argument("--overwrite-features", action="store_true", help="Rebuild feature CSV if it already exists.")
    parser.add_argument(
        "--models",
        default="decision-tree,random-forest,gradient-boosting,knn,svm",
        help="Comma-separated models to train, or all.",
    )
    parser.add_argument("--tasks", choices=(*TASK_CHOICES, "all"), default="all", help="Task to train.")
    parser.add_argument("--max-depth", type=int, default=10, help="Maximum tree depth for tree-based models.")
    parser.add_argument("--n-estimators", type=int, default=200, help="Estimator count for ensemble models.")
    parser.add_argument("--neighbors", type=int, default=5, help="KNN neighbor count.")
    parser.add_argument("--svm-c", type=float, default=1.5, help="SVM regularization strength.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--top-k-features",
        type=int,
        default=0,
        help="Keep only the top K random-forest features for training. 0 disables selection.",
    )
    return parser.parse_args()


def normalized_label_set(label_set: str) -> str:
    return LABEL_SET_ALIASES[label_set]


def load_image(path: Path):
    image = cv2.imread(str(path))
    if image is None:
        print(f"Skipping unreadable image: {path}")
    return image


def crop_bbox(image, bbox) -> np.ndarray | None:
    x, y, w, h = [int(round(value)) for value in bbox]
    x = max(0, x)
    y = max(0, y)
    w = max(0, min(w, image.shape[1] - x))
    h = max(0, min(h, image.shape[0] - y))
    if w <= 0 or h <= 0:
        return None
    return image[y : y + h, x : x + w]


def crop_obb_polygon(image, coords: list[float]) -> np.ndarray | None:
    height, width = image.shape[:2]
    points = np.asarray(
        [[coords[idx] * width, coords[idx + 1] * height] for idx in range(0, 8, 2)],
        dtype=np.float32,
    )
    x1 = max(0, int(np.floor(points[:, 0].min())))
    y1 = max(0, int(np.floor(points[:, 1].min())))
    x2 = min(width, int(np.ceil(points[:, 0].max())))
    y2 = min(height, int(np.ceil(points[:, 1].max())))
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


def amount_from_category(category_id: int) -> int:
    if category_id == 0:
        return 0
    return min(int(category_id), 3)


def add_feature_row(rows: list[dict], metadata: dict, crop, feature_set: str):
    from liquid_level_features import extract_liquid_features

    features = extract_liquid_features(crop, feature_set=feature_set, is_already_cropped=True)
    if not features:
        return
    row = dict(metadata)
    row.update(features)
    rows.append(row)


def build_lcdtc_rows(lcdtc_root: Path, feature_set: str) -> list[dict]:
    rows: list[dict] = []
    for split, annotation_name, image_subdir in (
        ("train", "instances_train2017.json", "train2017"),
        ("val", "instances_val2017.json", "val2017"),
    ):
        annotation_path = lcdtc_root / "annotations" / annotation_name
        image_dir = lcdtc_root / "images" / image_subdir
        if not annotation_path.exists() or not image_dir.exists():
            print(f"Skipping LCDTC {split}: missing {annotation_path} or {image_dir}")
            continue

        with annotation_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        category_names = {int(item["id"]): item.get("name", str(item["id"])) for item in data.get("categories", [])}
        image_lookup = {int(item["id"]): item["file_name"] for item in data.get("images", [])}

        for index, annotation in enumerate(data.get("annotations", []), start=1):
            if index % 1000 == 0:
                print(f"LCDTC {split}: {index}/{len(data.get('annotations', []))}")
            image_name = image_lookup.get(int(annotation["image_id"]))
            if not image_name:
                continue
            image_path = image_dir / image_name
            image = load_image(image_path)
            if image is None:
                continue
            crop = crop_bbox(image, annotation["bbox"])
            if crop is None:
                continue
            category_id = int(annotation["category_id"])
            amount_label = amount_from_category(category_id)
            add_feature_row(
                rows,
                {
                    "source": "lcdtc",
                    "split": split,
                    "image_name": image_name,
                    "crop_source": str(image_path.resolve()).replace("\\", "/"),
                    "class_id": category_id,
                    "class_name": category_names.get(category_id, str(category_id)),
                    "has_liquid": int(amount_label != 0),
                    "amount_label": amount_label,
                },
                crop,
                feature_set,
            )

    return rows


def iter_images(image_dir: Path):
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() in IMAGE_EXTENSIONS:
            yield image_path


def read_first_label(label_path: Path) -> tuple[int, list[float]] | None:
    if not label_path.exists():
        return None
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.strip().split()
        if len(parts) >= 9:
            return int(parts[0]), [float(value) for value in parts[1:9]]
    return None


def build_bottle_dataset_rows(dataset_root: Path, label_set: str, feature_set: str) -> list[dict]:
    rows: list[dict] = []
    label_key = normalized_label_set(label_set)
    label_config = LABEL_SETS[label_key]
    label_root = dataset_root / label_config["label_dir"]
    if not label_root.exists():
        print(f"Skipping bottleDataset: missing label directory {label_root}")
        return rows

    for split in ("train", "val", "test"):
        image_dir = dataset_root / "images" / split
        if not image_dir.exists():
            continue
        for image_path in iter_images(image_dir):
            label = read_first_label(label_root / split / f"{image_path.stem}.txt")
            if label is None:
                continue
            class_id, coords = label
            image = load_image(image_path)
            if image is None:
                continue
            crop = crop_obb_polygon(image, coords)
            if crop is None:
                continue
            amount_label = amount_from_category(class_id)
            class_name = label_config["names"][class_id] if class_id < len(label_config["names"]) else str(class_id)
            add_feature_row(
                rows,
                {
                    "source": "bottle-dataset",
                    "split": split,
                    "image_name": image_path.name,
                    "crop_source": str(image_path.resolve()).replace("\\", "/"),
                    "class_id": class_id,
                    "class_name": class_name,
                    "has_liquid": int(amount_label != 0),
                    "amount_label": amount_label,
                },
                crop,
                feature_set,
            )

    return rows


def build_labels_picture_rows(labels_picture_root: Path, feature_set: str) -> list[dict]:
    rows: list[dict] = []
    info_csv = labels_picture_root / "crop_info.csv"
    if not info_csv.exists():
        print(f"Skipping labels_picture: missing {info_csv}")
        return rows

    with info_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for item in reader:
            relative_image = item.get("cropped_image_path", "")
            image_path = labels_picture_root / relative_image
            image = load_image(image_path)
            if image is None:
                continue
            split = item.get("new_split") or item.get("split") or "train"
            has_liquid = int(float(item.get("has_liquid", 0)))
            amount_label = int(float(item.get("amount_label", 0 if has_liquid == 0 else 3)))
            add_feature_row(
                rows,
                {
                    "source": "labels-picture",
                    "split": split,
                    "image_name": relative_image,
                    "crop_source": str(image_path.resolve()).replace("\\", "/"),
                    "class_id": amount_label,
                    "class_name": str(amount_label),
                    "has_liquid": has_liquid,
                    "amount_label": amount_label,
                },
                image,
                feature_set,
            )

    return rows


def write_feature_csv(rows: list[dict], output_path: Path) -> Path:
    if not rows:
        raise ValueError("No feature rows were built. Check dataset paths and labels.")

    feature_columns = sorted({key for row in rows for key in row if key not in METADATA_COLUMNS})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*METADATA_COLUMNS, *feature_columns])
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, 0.0) for column in writer.fieldnames})
    print(f"Feature CSV: {output_path} ({len(rows)} rows, {len(feature_columns)} features)")
    return output_path


def build_feature_csv(args) -> Path:
    import_runtime_dependencies()

    output_dir = Path(args.output).resolve()
    feature_csv = output_dir / f"features_{args.feature_set}.csv"
    if args.features:
        feature_csv = Path(args.features).resolve()
        if not feature_csv.exists():
            raise FileNotFoundError(f"Feature CSV not found: {feature_csv}")
        return feature_csv

    if feature_csv.exists() and not args.overwrite_features:
        print(f"Reusing feature CSV: {feature_csv}")
        return feature_csv

    rows: list[dict] = []
    if "lcdtc" in args.sources:
        rows.extend(build_lcdtc_rows(Path(args.lcdtc_root).resolve(), args.feature_set))
    if "bottle-dataset" in args.sources:
        rows.extend(build_bottle_dataset_rows(Path(args.bottle_dataset).resolve(), args.label_set, args.feature_set))
    if "labels-picture" in args.sources:
        if not args.labels_picture_root:
            raise ValueError("--labels-picture-root is required when --sources includes labels-picture")
        rows.extend(build_labels_picture_rows(Path(args.labels_picture_root).resolve(), args.feature_set))

    return write_feature_csv(rows, feature_csv)


def read_feature_csv(feature_csv: Path):
    rows = []
    with feature_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        feature_columns = [column for column in reader.fieldnames or [] if column not in METADATA_COLUMNS]
        for row in reader:
            split = row.get("split", "train")
            if split == "predict":
                continue
            rows.append(
                {
                    "split": split,
                    "has_liquid": int(float(row["has_liquid"])),
                    "amount_label": int(float(row["amount_label"])),
                    "features": [float(row.get(column, 0.0) or 0.0) for column in feature_columns],
                }
            )
    if not rows:
        raise ValueError(f"No usable feature rows found in {feature_csv}")
    x = np.nan_to_num(np.asarray([row["features"] for row in rows], dtype=np.float32))
    splits = np.asarray([row["split"] for row in rows])
    labels = {
        "binary": np.asarray([row["has_liquid"] for row in rows]),
        "amount": np.asarray([row["amount_label"] for row in rows]),
    }
    return x, labels, splits, feature_columns


def select_models(models_arg: str) -> list[str]:
    if models_arg == "all":
        return list(MODEL_CHOICES)
    selected = [item.strip() for item in models_arg.split(",") if item.strip()]
    unknown = [item for item in selected if item not in MODEL_CHOICES]
    if unknown:
        raise ValueError(f"Unsupported models: {', '.join(unknown)}")
    return selected


def build_model(name: str, args):
    if name == "decision-tree":
        return DecisionTreeClassifier(max_depth=args.max_depth, class_weight="balanced", random_state=args.random_state)
    if name == "random-forest":
        return RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            class_weight="balanced",
            random_state=args.random_state,
            n_jobs=-1,
        )
    if name == "gradient-boosting":
        return GradientBoostingClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth if args.max_depth is not None else 3,
            random_state=args.random_state,
        )
    if name == "knn":
        return Pipeline([("scale", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=args.neighbors))])
    if name == "svm":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", SVC(kernel="rbf", C=args.svm_c, gamma="scale", class_weight="balanced")),
            ]
        )
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth if args.max_depth is not None else 6,
            learning_rate=0.1,
            eval_metric="mlogloss",
            random_state=args.random_state,
        )
    raise ValueError(f"Unsupported model: {name}")


def unwrap_model(model):
    if isinstance(model, Pipeline):
        return model.steps[-1][1]
    return model


def split_indices(splits: np.ndarray):
    return {
        "train": np.where(splits == "train")[0],
        "val": np.where(splits == "val")[0],
        "test": np.where(splits == "test")[0],
    }


def maybe_select_features(x_train, y_train, x_all, feature_names: list[str], top_k: int, random_state: int):
    if top_k <= 0 or top_k >= len(feature_names):
        return x_all, feature_names, None

    selector = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=random_state, n_jobs=-1)
    selector.fit(x_train, y_train)
    selected_idx = np.argsort(selector.feature_importances_)[::-1][:top_k]
    selected_names = [feature_names[idx] for idx in selected_idx]
    selected_importances = [(feature_names[idx], float(selector.feature_importances_[idx])) for idx in selected_idx]
    return x_all[:, selected_idx], selected_names, selected_importances


def evaluate_model(model, x, y, indices: dict[str, np.ndarray], label_names: list[str]):
    results = {}
    for split in ("train", "val", "test"):
        split_idx = indices[split]
        if len(split_idx) == 0:
            continue
        y_true = y[split_idx]
        y_pred = model.predict(x[split_idx])
        labels = sorted(set(y.tolist()))
        results[split] = {
            "samples": int(len(split_idx)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "classification_report": classification_report(
                y_true,
                y_pred,
                labels=labels,
                target_names=[label_names[int(label)] if str(label).isdigit() and int(label) < len(label_names) else str(label) for label in labels],
                zero_division=0,
                output_dict=True,
            ),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        }
    return results


def write_importances(model, feature_names: list[str], output_path: Path):
    estimator = unwrap_model(model)
    if not hasattr(estimator, "feature_importances_"):
        return None
    rows = sorted(zip(feature_names, estimator.feature_importances_), key=lambda item: item[1], reverse=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature", "importance"])
        for feature, importance in rows:
            writer.writerow([feature, f"{float(importance):.10f}"])
    return output_path


def train_and_evaluate(feature_csv: Path, args):
    import_runtime_dependencies()

    x, labels_by_task, splits, feature_names = read_feature_csv(feature_csv)
    indices = split_indices(splits)
    if len(indices["train"]) == 0:
        raise ValueError("No train split rows found in the feature CSV.")

    output_root = Path(args.output).resolve()
    models = select_models(args.models)
    tasks = TASK_CHOICES if args.tasks == "all" else (args.tasks,)
    summary_rows = []

    for task in tasks:
        y = labels_by_task[task]
        task_label_names = ["empty", "liquid"] if task == "binary" else ["empty", "little", "mid", "much"]
        x_task, selected_features, selected_importances = maybe_select_features(
            x[indices["train"]],
            y[indices["train"]],
            x,
            feature_names,
            args.top_k_features,
            args.random_state,
        )
        for model_name in models:
            model_output = output_root / task / model_name
            model_output.mkdir(parents=True, exist_ok=True)

            print(f"\nTraining {task}/{model_name} ...")
            try:
                model = build_model(model_name, args)
                model.fit(x_task[indices["train"]], y[indices["train"]])
                metrics = evaluate_model(model, x_task, y, indices, task_label_names)
            except Exception as exc:
                print(f"FAILED {task}/{model_name}: {exc}")
                summary_rows.append(
                    {
                        "task": task,
                        "model": model_name,
                        "split": "",
                        "accuracy": "",
                        "accuracy_percent": "",
                        "status": f"failed: {exc}",
                        "output_dir": str(model_output).replace("\\", "/"),
                    }
                )
                continue

            model_path = model_output / f"{model_name.replace('-', '_')}.joblib"
            joblib.dump(
                {
                    "model": model,
                    "task": task,
                    "model_name": model_name,
                    "feature_names": selected_features,
                    "feature_csv": str(feature_csv.resolve()).replace("\\", "/"),
                },
                model_path,
            )
            (model_output / "metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            write_importances(model, selected_features, model_output / "feature_importances.csv")
            if model_name == "decision-tree":
                (model_output / "tree_rules.txt").write_text(
                    export_text(unwrap_model(model), feature_names=selected_features),
                    encoding="utf-8",
                )
            if selected_importances:
                with (model_output / "selected_features.csv").open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["feature", "selection_importance"])
                    writer.writerows(selected_importances)

            best_split = "val" if "val" in metrics else ("test" if "test" in metrics else "train")
            best_accuracy = metrics[best_split]["accuracy"]
            print(f"{task}/{model_name}: {best_split} accuracy={best_accuracy:.4f}")
            summary_rows.append(
                {
                    "task": task,
                    "model": model_name,
                    "split": best_split,
                    "accuracy": f"{best_accuracy:.10f}",
                    "accuracy_percent": f"{best_accuracy * 100:.2f}%",
                    "status": "ok",
                    "output_dir": str(model_output).replace("\\", "/"),
                }
            )

    summary_path = output_root / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["task", "model", "split", "accuracy", "accuracy_percent", "status", "output_dir"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    successful_rows = [row for row in summary_rows if row["status"] == "ok" and row["accuracy"]]
    if not successful_rows:
        raise RuntimeError("No classical ML model finished successfully.")
    best = max(successful_rows, key=lambda row: float(row["accuracy"]))
    print("\nBest result:")
    print(f"{best['task']}/{best['model']} on {best['split']} accuracy={float(best['accuracy']):.4f} ({best['accuracy_percent']})")
    print(f"Summary CSV: {summary_path}")


def main():
    args = parse_args()
    feature_csv = build_feature_csv(args)
    train_and_evaluate(feature_csv, args)


if __name__ == "__main__":
    main()
