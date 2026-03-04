import os
import json
import pandas as pd
from sklearn.datasets import load_iris

RAW_DIR = os.getenv("RAW_DIR", "data/raw")
PROCESSED_DIR = os.getenv("PROCESSED_DIR", "data/processed")

def preprocess_data(output_path: str = None) -> str:
    """
    Idempotent: if output already exists, re-use it.
    Produces a single deterministic dataset file.
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    if output_path is None:
        output_path = os.path.join(PROCESSED_DIR, "iris_processed.csv")

    if os.path.exists(output_path):
        print(f"[preprocess] Reusing existing processed file: {output_path}")
        return output_path

    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    # deterministic column names
    df.columns = [c.replace(" (cm)", "").replace(" ", "_") for c in df.columns]

    df.to_csv(output_path, index=False)

    meta = {
        "dataset": "sklearn_iris",
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "output_path": output_path,
    }
    with open(os.path.join(PROCESSED_DIR, "data_version.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[preprocess] Wrote processed data: {output_path}")
    return output_path

if __name__ == "__main__":
    preprocess_data()