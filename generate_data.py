"""
Simulates ICS (Industrial Control System) sensor readings.
Generates normal traffic + injects 3 types of attacks.
"""

import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

# --- Sensor definitions ---
# Each sensor has: name, normal range (mean, std), unit
SENSORS = {
    "temperature":  {"mean": 320.0, "std": 5.0,  "unit": "°C"},
    "pressure":     {"mean": 150.0, "std": 3.0,  "unit": "PSI"},
    "flow_rate":    {"mean": 240.0, "std": 8.0,  "unit": "L/min"},
    "voltage":      {"mean": 480.0, "std": 4.0,  "unit": "V"},
    "vibration":    {"mean": 0.5,   "std": 0.05, "unit": "mm/s"},
}

# --- Attack types ---
# 1. Spike attack   – sudden large jump (e.g. sabotage setpoint)
# 2. Freeze attack  – value stuck (sensor replay / spoofing)
# 3. Gradual drift  – slow creep toward danger zone (hard to notice)
ATTACK_TYPES = ["spike", "freeze", "gradual_drift"]


def generate_dataset(n_readings=200, attack_ratio=0.15):
    """
    Returns a DataFrame with columns:
      timestamp, sensor, value, unit, label (normal/attack), attack_type
    """
    rows = []
    timestamp = pd.Timestamp("2024-01-01 08:00:00")

    for i in range(n_readings):
        timestamp += pd.Timedelta(seconds=5)

        for sensor_name, cfg in SENSORS.items():
            # Decide if this reading is an attack
            is_attack = random.random() < attack_ratio
            attack_type = random.choice(ATTACK_TYPES) if is_attack else "none"

            # Generate the value
            normal_value = np.random.normal(cfg["mean"], cfg["std"])

            if not is_attack:
                value = normal_value

            elif attack_type == "spike":
                # Jump 4–7 standard deviations
                direction = random.choice([-1, 1])
                value = cfg["mean"] + direction * cfg["std"] * random.uniform(4, 7)

            elif attack_type == "freeze":
                # Value suspiciously identical to last reading (or exact mean)
                value = cfg["mean"]   # simplified: always the exact mean

            elif attack_type == "gradual_drift":
                # Drift 2–4 std above normal (sneaky)
                value = cfg["mean"] + cfg["std"] * random.uniform(2, 4)

            rows.append({
                "timestamp":   timestamp,
                "sensor":      sensor_name,
                "value":       round(value, 3),
                "unit":        cfg["unit"],
                "label":       "attack" if is_attack else "normal",
                "attack_type": attack_type,
            })

    df = pd.DataFrame(rows)
    df.to_csv("sensor_data.csv", index=False)
    print(f"[+] Generated {len(df)} readings → sensor_data.csv")
    print(f"    Normal:  {(df.label == 'normal').sum()}")
    print(f"    Attack:  {(df.label == 'attack').sum()}")
    return df


if __name__ == "__main__":
    generate_dataset()
