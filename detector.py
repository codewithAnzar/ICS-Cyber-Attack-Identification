"""
-
Runs anomaly detection on sensor_data.csv using two approaches:
  1. Z-score rule  – simple, explainable, good baseline
  2. Isolation Forest – ML model, catches subtler patterns

Prints a clear report and saves results to results.csv
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv("sensor_data.csv", parse_dates=["timestamp"])
print(f"[+] Loaded {len(df)} rows from sensor_data.csv\n")


# METHOD 1: Z-SCORE (rule-based)
# For each sensor, compute how many standard deviations a reading is from
# that sensor's mean. If |z| > threshold, flag it as anomaly.

print("── Method 1: Z-score (rule-based) ───────────────────────────────────")

Z_THRESHOLD = 2.5   # readings beyond 2.5 std are suspicious

# Compute per-sensor mean and std from the data
sensor_stats = df.groupby("sensor")["value"].agg(["mean", "std"]).reset_index()
sensor_stats.columns = ["sensor", "sensor_mean", "sensor_std"]

df = df.merge(sensor_stats, on="sensor")
df["z_score"] = (df["value"] - df["sensor_mean"]) / df["sensor_std"]
df["z_score"] = df["z_score"].abs().round(3)

df["zscore_pred"] = df["z_score"].apply(lambda z: "attack" if z > Z_THRESHOLD else "normal")

# Evaluate
y_true = df["label"]
y_pred_z = df["zscore_pred"]

print(classification_report(y_true, y_pred_z, target_names=["normal", "attack"]))


# METHOD 2: ISOLATION FOREST (ML-based)

# Isolation Forest isolates anomalies by randomly partitioning data.
# Anomalies are isolated in fewer splits → lower anomaly score.
# No labels needed for training (unsupervised).

print("── Method 2: Isolation Forest (ML) ──────────────────────────────────")

# Build feature matrix: one row per reading
# Features: value, z_score, and one-hot encoded sensor name
features_raw = df[["value", "z_score"]].copy()

# Add sensor as a numeric feature (encode as integer)
sensor_map = {s: i for i, s in enumerate(df["sensor"].unique())}
features_raw["sensor_id"] = df["sensor"].map(sensor_map)

scaler = StandardScaler()
X = scaler.fit_transform(features_raw)

# contamination = expected fraction of anomalies (we set 15%)
iso_forest = IsolationForest(contamination=0.15, random_state=42, n_estimators=100)
iso_preds_raw = iso_forest.fit_predict(X)

# IsolationForest returns: -1 = anomaly, 1 = normal
df["iforest_pred"] = ["attack" if p == -1 else "normal" for p in iso_preds_raw]
df["anomaly_score"] = iso_forest.decision_function(X).round(4)

y_pred_if = df["iforest_pred"]
print(classification_report(y_true, y_pred_if, target_names=["normal", "attack"]))


# COMBINED VERDICT: flag if EITHER method says attack
df["final_pred"] = df.apply(
    lambda r: "attack" if r["zscore_pred"] == "attack" or r["iforest_pred"] == "attack" else "normal",
    axis=1
)

# ── Summary report ────────────────────────────────────────────────────────────
print("── Combined verdict ──────────────────────────────────────────────────")
total    = len(df)
detected = (df["final_pred"] == "attack").sum()
actual   = (df["label"] == "attack").sum()
correct  = ((df["final_pred"] == "attack") & (df["label"] == "attack")).sum()

print(f"  Total readings:    {total}")
print(f"  Actual attacks:    {actual}")
print(f"  Flagged:           {detected}")
print(f"  True positives:    {correct}")
print(f"  Detection rate:    {correct/actual*100:.1f}%\n")

# Show top 10 most suspicious readings
print("── Top 10 most anomalous readings ───────────────────────────────────")
top = (
    df[df["final_pred"] == "attack"]
    .sort_values("z_score", ascending=False)
    .head(10)[["timestamp", "sensor", "value", "unit", "z_score", "attack_type", "final_pred"]]
)
print(top.to_string(index=False))

# Save results
df.to_csv("results.csv", index=False)
print("\n[+] Full results saved → results.csv")
