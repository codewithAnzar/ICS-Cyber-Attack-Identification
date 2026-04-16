"""
Generates 3 plots from results.csv:
  1. Sensor readings over time – normal vs attack highlighted
  2. Z-score distribution per sensor
  3. Confusion matrix (z-score vs Isolation Forest)

Saves all plots to the plots/ folder.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (works without a display)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

os.makedirs("plots", exist_ok=True)

df = pd.read_csv("results.csv", parse_dates=["timestamp"])
print("[+] Loaded results.csv — generating plots...\n")

# ── Color scheme ──────────────────────────────────────────────────────────────
COLOR_NORMAL  = "#4A90D9"
COLOR_ATTACK  = "#E24B4A"
COLOR_DRIFT   = "#EF9F27"
BG            = "#F8F8F7"
GRID          = "#E0DED8"

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "axes.facecolor": BG,
    "figure.facecolor": "white",
    "axes.grid":      True,
    "grid.color":     GRID,
    "grid.linewidth": 0.5,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})


# PLOT 1: Temperature & Pressure over time with anomalies highlighted
fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
fig.suptitle("ICS Sensor Readings — Anomaly Detection Results", fontsize=14, fontweight="bold", y=0.98)

for ax, sensor in zip(axes, ["temperature", "pressure"]):
    sub = df[df["sensor"] == sensor].copy().sort_values("timestamp")

    # Plot all points
    normal  = sub[sub["final_pred"] == "normal"]
    attacks = sub[sub["final_pred"] == "attack"]

    ax.plot(sub["timestamp"], sub["value"], color=COLOR_NORMAL, linewidth=0.8, alpha=0.6, zorder=1)
    ax.scatter(normal["timestamp"],  normal["value"],  color=COLOR_NORMAL, s=18, zorder=2, label="Normal")
    ax.scatter(attacks["timestamp"], attacks["value"], color=COLOR_ATTACK,  s=40, marker="X", zorder=3, label="Detected anomaly")

    # Baseline band (±2 std)
    mean = sub["value"].mean()
    std  = sub["value"].std()
    ax.axhspan(mean - 2*std, mean + 2*std, alpha=0.08, color=COLOR_NORMAL, label="±2σ band")
    ax.axhline(mean, color=COLOR_NORMAL, linewidth=0.8, linestyle="--", alpha=0.5)

    unit = sub["unit"].iloc[0]
    ax.set_ylabel(f"{sensor.title()} ({unit})", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")

axes[1].set_xlabel("Time", fontsize=10)
plt.tight_layout()
plt.savefig("plots/01_sensor_timeline.png", dpi=150, bbox_inches="tight")
plt.close()
print("[+] Saved plots/01_sensor_timeline.png")


# PLOT 2: Z-score distribution – normal vs attack
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_title("Z-score Distribution: Normal vs Attack Readings", fontsize=13, fontweight="bold")

normal_z  = df[df["label"] == "normal"]["z_score"]
attack_z  = df[df["label"] == "attack"]["z_score"]

bins = np.linspace(0, df["z_score"].max() + 0.5, 40)
ax.hist(normal_z, bins=bins, color=COLOR_NORMAL, alpha=0.7, label=f"Normal (n={len(normal_z)})")
ax.hist(attack_z, bins=bins, color=COLOR_ATTACK,  alpha=0.7, label=f"Attack (n={len(attack_z)})")

# Threshold line
ax.axvline(x=2.5, color="#333", linewidth=1.5, linestyle="--", label="Threshold (z=2.5)")
ax.text(2.6, ax.get_ylim()[1] * 0.85, "Detection\nthreshold", fontsize=9, color="#333")

ax.set_xlabel("Z-score (standard deviations from mean)", fontsize=10)
ax.set_ylabel("Count", fontsize=10)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("plots/02_zscore_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("[+] Saved plots/02_zscore_distribution.png")


# PLOT 3: Anomalies by attack type + per-sensor breakdown
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Attack Analysis", fontsize=13, fontweight="bold")

# Left: by attack type
attack_df = df[df["label"] == "attack"]
type_counts = attack_df["attack_type"].value_counts()
colors_bar = [COLOR_ATTACK, COLOR_DRIFT, "#9B59B6"]
bars = ax1.bar(type_counts.index, type_counts.values, color=colors_bar[:len(type_counts)], edgecolor="white", linewidth=0.5)
ax1.set_title("Attacks by Type", fontsize=11)
ax1.set_ylabel("Count")
for bar, val in zip(bars, type_counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 1, str(val), ha="center", fontsize=9)

# Right: detection rate per sensor
detected_per_sensor = df[df["final_pred"] == "attack"].groupby("sensor").size()
actual_per_sensor   = df[df["label"]      == "attack"].groupby("sensor").size()
det_rate = (detected_per_sensor / actual_per_sensor * 100).fillna(0).round(1)

colors_s = [COLOR_ATTACK if v >= 70 else COLOR_DRIFT if v >= 40 else COLOR_NORMAL for v in det_rate.values]
bars2 = ax2.barh(det_rate.index, det_rate.values, color=colors_s, edgecolor="white", linewidth=0.5)
ax2.set_title("Detection Rate per Sensor (%)", fontsize=11)
ax2.set_xlabel("Detection rate (%)")
ax2.axvline(x=70, color="#333", linewidth=1, linestyle="--", alpha=0.5)
for bar, val in zip(bars2, det_rate.values):
    ax2.text(val + 1, bar.get_y() + bar.get_height()/2, f"{val}%", va="center", fontsize=9)

plt.tight_layout()
plt.savefig("plots/03_attack_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("[+] Saved plots/03_attack_analysis.png")

print("\n[+] All plots saved to plots/ folder.")
