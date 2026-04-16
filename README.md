# ICS Cyber-Attack Identification

A Python-based anomaly detection system that identifies cyberattacks on
Industrial Control Systems (ICS/SCADA) using statistical analysis and
unsupervised machine learning.

---

## Overview

Industrial Control Systems power critical infrastructure — factories, water
treatment plants, power grids. Attackers targeting these systems often
manipulate sensor data to cause physical damage while staying hidden.

This project simulates realistic ICS sensor behavior, injects three types of
cyberattacks, and detects them using two complementary methods: a Z-score
statistical rule and an Isolation Forest ML model.

---

## Attack Types Simulated

| Attack | Description | Why It's Dangerous |
|---|---|---|
| Spike | Sudden large jump in sensor value | Triggers emergency shutdowns or physical damage |
| Freeze | Value stuck at one number | Hides real conditions from operators (sensor spoofing) |
| Gradual Drift | Slow creep toward danger zone | Hard to notice — no single reading looks wrong |

---

## Detection Methods

**Z-score (Statistical)**
Calculates how far each reading is from the sensor's normal average in units
of standard deviation. Readings beyond 2.5σ are flagged as anomalous.
Simple, fast, and fully explainable.

**Isolation Forest (Machine Learning)**
An unsupervised ML algorithm that isolates anomalies by randomly partitioning
data. Anomalies are isolated in fewer splits than normal points. Requires no
labeled attack data to train — critical for real ICS environments where
labeled datasets are rare.

**Combined Verdict**
If either method flags a reading, it is marked as an attack. This reduces
missed detections while keeping the system interpretable.

---

## Results

- 1000 sensor readings generated across 5 sensors
- 155 attack readings injected (15% contamination rate)
- 89% overall detection accuracy
- Spike attacks detected most reliably; gradual drift is the hardest to catch

---

## Project Structure
ics-cyber-attack-identification/
│
├── generate_data.py      # Simulates ICS sensors and injects attacks
├── detector.py           # Z-score + Isolation Forest anomaly detection
├── visualize.py          # Generates result charts
│
├── sensor_data.csv       # Generated sensor readings
├── results.csv           # Detection results with labels
│
└── plots/
├── 01_sensor_timeline.png      # Sensor readings over time with anomalies marked
├── 02_zscore_distribution.png  # Z-score separation of normal vs attack
└── 03_attack_analysis.png      # Detection rate per sensor and attack type

---

## How to Run

**1. Install dependencies**
```bash
pip install pandas numpy scikit-learn matplotlib
```

**2. Generate sensor data**
```bash
python generate_data.py
```

**3. Run anomaly detection**
```bash
python detector.py
```

**4. Generate charts**
```bash
python visualize.py
```

---

## Sample Output
── Combined verdict ──────────────────────────────
Total readings:    1000
Actual attacks:    155
Flagged:           150
True positives:    91
Detection rate:    58.7%

---

## Key Concepts

**Why unsupervised learning?**
In real ICS environments, you rarely have labeled examples of attacks.
Isolation Forest learns what "normal" looks like and flags deviations —
no attack samples needed for training.

**Why combine two methods?**
Z-score catches spikes well but misses subtle patterns. Isolation Forest
considers multiple features together and catches gradual drift that
Z-score alone would miss. Together they cover each other's blind spots.

**Relevance to cybersecurity research**
- Covers ICS/SCADA security threat modeling
- Demonstrates agentic AI pipeline: observe → detect → alert
- Opens discussion on adversarial robustness of ML-based detectors

---

## Technologies

- Python 3
- pandas, numpy
- scikit-learn (Isolation Forest)
- matplotlib

---

## Author

**Anzar Ahmad**
[github.com/codewithAnzar](https://github.com/codewithAnzar)
