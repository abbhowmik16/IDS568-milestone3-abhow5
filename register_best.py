import os
import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "ids568_milestone3")
MODEL_NAME = os.getenv("MODEL_NAME", "ids568_milestone3_model")

def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise RuntimeError(f"Experiment not found: {EXPERIMENT_NAME}")

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.accuracy DESC"],
        max_results=1
    )
    if not runs:
        raise RuntimeError("No runs found. Run train.py multiple times first.")

    best = runs[0]
    run_id = best.info.run_id

    # model logged at artifact path "model" in train.py
    model_uri = f"runs:/{run_id}/model"
    print("Registering model from:", model_uri)

    mv = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
    print("Registered version:", mv.version)

    # tag + describe version
    client.set_model_version_tag(MODEL_NAME, mv.version, "candidate", "true")
    client.update_model_version(
        name=MODEL_NAME,
        version=mv.version,
        description=f"Auto-registered from best accuracy run_id={run_id}"
    )

    # stage progression: None -> Staging -> Production (document in report + screenshots)
    client.transition_model_version_stage(MODEL_NAME, mv.version, stage="Staging", archive_existing_versions=False)
    client.transition_model_version_stage(MODEL_NAME, mv.version, stage="Production", archive_existing_versions=False)

    print("Promoted to Production.")

if __name__ == "__main__":
    main()