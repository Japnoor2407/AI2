# DATASET PROFILE: Heterogeneous Traffic Count Dataset, Pune (Jan 2023)

## 1. Overview & Provenance
- **Dataset Title**: Heterogeneous Traffic Count Dataset, Pune (Jan 2023)
- **Source**: Traffic Police Commissioner, Pune; processed by Department of Computer Engineering, COEP Technological University, Pune under TEQIP-III.
- **Recording Period**: January 11, 2023 – January 18, 2023 (7 consecutive days).
- **Locations**: 3 major high-congestion intersections in Pune, Maharashtra, India:
  1. **Alankar Chowk** (Camera feeds: `a2`, `a3`)
  2. **Jehangir Chowk** (Camera feeds: `j1`, `j2`, `j3`)
  3. **RTO Chowk** (Camera feeds: `r1`, `r2`, `r3`)
- **Total Files**: 115 CSV files organized hierarchically by Junction and Camera identifier.
- **Processing Pipeline**: Fixed CCTV cameras $\to$ Frame extraction at 25 fps $\to$ YOLO-based object detection $\to$ 4-class vehicle classification $\to$ Multi-frame object tracking $\to$ Non-Maximum Suppression (NMS) $\to$ Directional counting per interval $\to$ CSV logs.

---

## 2. File Organization & Naming Convention
```
traffic/
  traffic/
    AlankarChowk/
      a2/
        a2_11.csv, a2_11_one.csv, a2_12.csv, ..., a2_18_one.csv
      a3/
        a3_11.csv, a3_11_one.csv, a3_12.csv, ..., a3_18_one.csv
    JehangirChowk/
      j1/ (j1_11.csv ... j1_18_one.csv)
      j2/ (j2_11.csv ... j2_18_one.csv)
      j3/ (j3_11.csv ... j3_18.csv)
    RTOChowk/
      r1/ (r1_11.csv ... r1_18_one.csv)
      r2/ (r2_11.csv ... r2_18_one.csv)
      r3/ (r3_11.csv ... r3_18.csv)
```
- Primary files follow `<cameraID>_<day>.csv` (e.g., `a2_11.csv` = Camera 2 at Alankar Chowk on January 11, 2023).
- Continuation files follow `<cameraID>_<day>_one.csv` (e.g., `a2_11_one.csv`).

---

## 3. Raw Data Schema & Column Definitions

| Column Name | Raw Data Type | Canonical Type | Allowed Values | Semantic Meaning |
| :--- | :--- | :--- | :--- | :--- |
| `Time` | `str` | `datetime.time` | `00:00:00` – `23:59:59` | Start timestamp of the observation interval. |
| `Direction` | `str` | `enum` | `UP`, `DOWN`, `LEFT`, `RIGHT` | Directional movement vector through the camera field of view. |
| `car` | `float64` / `int` | `int` | $\ge 0$ | Cumulative detected count of standard passenger cars. |
| `motorbike` | `float64` / `int` | `int` | $\ge 0$ | Cumulative detected count of 2-wheelers (motorcycles, scooters). |
| `bus` | `float64` / `int` | `int` | $\ge 0$ | Cumulative detected count of public transit & private buses. |
| `truck` | `float64` / `int` | `int` | $\ge 0$ | Cumulative detected count of commercial heavy freight trucks. |

---

## 4. Critical Structural Findings & Validation Rules

### A. Frame-Rate Granularity & Cumulative Counter Semantics
- **Inspection Finding**: Each second contains exactly **100 rows** (25 frames per second across 4 directional records `UP`, `DOWN`, `LEFT`, `RIGHT`).
- **Cumulative vs Instantaneous**: The vehicle counts in the CSV are **cumulative running counters** tracking vehicles that passed the virtual tripwire since the session began.
- **CRITICAL DATA HANDLING RULE**:
  $$\text{Arrivals}(t_k, t_{k+1}) = \max\left(0, \text{Count}(t_{k+1}) - \text{Count}(t_k)\right)$$
  **Never sum the raw CSV rows!** Directly summing the CSV rows would sum the same cumulative counter 25 times every second, causing an inflation error of $25\times$ to $2500\times$.
- **Counter Resets**: In continuation files (`*_one.csv`) or across session interruptions, counters reset to 0. The ingestion adapter detects negative discrete differences ($\Delta < 0$) and treats the new value as a fresh baseline.

### B. Missing Values & Data Hygiene
- Trailing lines in some CSV files contain padding `NaN` values (e.g. 1,880 trailing null rows in `a2_11.csv`).
- The adapter strictly drops rows with null timestamps or negative counts and coerces numeric fields cleanly to non-negative integers.

### C. Heterogeneous Traffic Weighting (Indian IRC PCU Standards)
Indian urban traffic is highly heterogeneous with high motorcycle density and mixed commercial vehicles. The system converts raw vehicle counts into **Passenger Car Units (PCU)** using IRC:106 guidelines:
- **Car**: $1.0\text{ PCU}$
- **Motorbike**: $0.5\text{ PCU}$
- **Bus**: $2.5\text{ PCU}$
- **Truck**: $2.5\text{ PCU}$

$$\text{Demand (PCU)} = 1.0 \times \text{car} + 0.5 \times \text{motorbike} + 2.5 \times \text{bus} + 2.5 \times \text{truck}$$

---

## 5. What the Dataset Supports vs Does NOT Support

### Supported Capabilities
1. **Empirical Multi-Class Arrival Modeling**: Precise time-stamped vehicle arrivals decomposed into 4 vehicle classes and 4 directional movements.
2. **Multi-Approach Intersection Synchronization**: Alankar Chowk has 2 synchronized camera perspectives (`a2`, `a3`), while Jehangir and RTO have 3 perspectives (`j1`, `j2`, `j3`). These map cleanly to junction approaches (North, South, East, West).
3. **Temporal Demand Variations**: Morning peak, midday lull, and evening traffic swings across 7 consecutive days.
4. **Demand Forecasting**: Chronological time-series prediction of upcoming vehicle arrival rates using lag features.
5. **High-Fidelity Replay & Simulation Testing**: Real Pune traffic streams driving both fixed-time and adaptive signal controllers under identical demand profiles.

### Explicit Dataset Limitations & Modeling Assumptions
1. **No Ground-Truth Queue Lengths**: The cameras only counted vehicles crossing tripwires; queue lengths are not labeled.
   * *Resolution*: The system uses a standard hydrodynamic queuing model where queue accumulates when arrival rate exceeds departure capacity during red phases:
     $$Q(t + \Delta t) = \max\left(0, Q(t) + \text{Arrivals}(t) - \text{Discharges}(t)\right)$$
     In the UI and logs, queues are clearly marked as **Estimated Queue (PCU)**.
2. **No Historical Signal Phase Annotations**: The CCTV dataset contains video detections, not traffic signal phase logs.
   * *Resolution*: Signal phases and cycles are governed by our simulation engine, allowing controlled benchmarking between Fixed-Time baseline and Adaptive controllers on the exact same empirical arrivals.
3. **No Trajectory / Individual Travel Time**: Individual vehicle trajectories were not exported.
   * *Resolution*: Vehicle waiting times are modeled through Little's Law and accumulated discrete delay tracking in the simulation engine.

