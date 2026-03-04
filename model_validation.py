import json
import os
import sys

THRESHOLD_ACC = float(os.getenv("MIN_ACCURACY", "0.8"))
THRESHOLD_F1 = float(os.getenv("MIN_F1", "0.8"))

metrics_file = "reports/metrics.json"

if not os.path.exists(metrics_file):
    print("metrics.json not found")
    sys.exit(1)

with open(metrics_file) as f:
    metrics = json.load(f)

accuracy = metrics.get("accuracy", 0)
f1 = metrics.get("f1_macro", 0)

print("Accuracy:", accuracy)
print("F1 Macro:", f1)

if accuracy >= THRESHOLD_ACC and f1 >= THRESHOLD_F1:
    print("MODEL VALIDATION PASSED")
else:
    print("MODEL VALIDATION FAILED")
    sys.exit(1)