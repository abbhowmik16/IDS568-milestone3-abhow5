import os
import json
import hashlib
import argparse
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


PROCESSED_DIR = os.getenv("PROCESSED_DIR", "data/processed")
MODEL_DIR = os.getenv("MODEL_DIR", "models")

# IMPORTANT:
# Use a portable default for MLflow that works on Linux + GitHub Actions
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "ids568_milestone3")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def train_model(C: float, max_iter: int, test_size: float, random_state: int) -> dict:
    os.makedirs(MODEL_DIR, exist_ok=True)

    data_path = os.path.join(PROCESSED_DIR, "iris_processed.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed data missing: {data_path}. Run preprocess first.")

    df = pd.read_csv(data_path)
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")

    model_path = os.path.join(MODEL_DIR, "model.pkl")
    joblib.dump(model, model_path)

    model_hash = sha256_file(model_path)

    data_version_path = os.path.join(PROCESSED_DIR, "data_version.json")
    data_version = {}
    if os.path.exists(data_version_path):
        with open(data_version_path, "r") as f:
            data_version = json.load(f)

    return {
        "accuracy": float(acc),
        "f1_macro": float(f1),
        "model_path": model_path,
        "model_hash": model_hash,
        "data_version": data_version,
    }


def _normalize_mlflow_uri(uri: str) -> str:
    """
    Make tracking uri safe across OS.
    - If user passes a plain path, convert to file:<abs_posix_path>
    - Avoid Windows-drive paths causing '/C:' errors on Linux runners
    """
    if uri.startswith("file:"):
        return uri

    # If it's a Windows path like C:\something, convert to file:///C:/something
    if len(uri) >= 3 and uri[1:3] == ":\\":
        uri = uri.replace("\\", "/")
        return f"file:///{uri}"

    # Otherwise treat as local path
    p = Path(uri).resolve()
    return f"file:{p.as_posix()}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=200)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    tracking_uri = _normalize_mlflow_uri(MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("[train] MLFLOW_TRACKING_URI =", tracking_uri)

    with mlflow.start_run():
        # log params
        mlflow.log_param("C", args.C)
        mlflow.log_param("max_iter", args.max_iter)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("data_path", os.path.join(PROCESSED_DIR, "iris_processed.csv"))

        results = train_model(args.C, args.max_iter, args.test_size, args.random_state)

        # log metrics
        mlflow.log_metric("accuracy", results["accuracy"])
        mlflow.log_metric("f1_macro", results["f1_macro"])

        # tags for checklist
        mlflow.set_tag("model_hash_sha256", results["model_hash"])
        if results["data_version"]:
            mlflow.set_tag("data_version_rows", results["data_version"].get("rows"))
            mlflow.set_tag("data_version_cols", results["data_version"].get("cols"))
            mlflow.set_tag("dataset", results["data_version"].get("dataset"))

        # log model
        mlflow.sklearn.log_model(
            sk_model=joblib.load(results["model_path"]),
            artifact_path="model",
        )

        # log raw model file
        mlflow.log_artifact(results["model_path"], artifact_path="model_file")

        # write metrics for CI validation
        os.makedirs("reports", exist_ok=True)
        metrics_out = {"accuracy": results["accuracy"], "f1_macro": results["f1_macro"]}
        with open("reports/metrics.json", "w") as f:
            json.dump(metrics_out, f, indent=2)

        mlflow.log_artifact("reports/metrics.json", artifact_path="reports")

        print("[train] metrics:", metrics_out)
        print("[train] model_hash_sha256:", results["model_hash"])


if __name__ == "__main__":
    main()