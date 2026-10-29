# System Architecture: SIH1607 Adaptive Traffic Signal Optimization Agent

## 1. System Overview
The SIH1607 Adaptive Traffic Signal Optimization Agent is a locally runnable, end-to-end intelligent traffic signal control and monitoring system. It processes empirical 25 fps CCTV traffic detection logs collected by the Traffic Police Commissioner, Pune, across major urban intersections in Pune, Maharashtra.

The platform includes:
- A high-throughput **Cumulative-to-Arrival Ingestion Engine** that converts raw cumulative detections into discrete vehicle arrivals across vehicle classes.
- A **Passenger Car Unit (PCU) Converter** adhering to Indian IRC:106 urban standards.
- A **Safety-Enforced Dual-Strategy Signal Controller** comparing pre-timed fixed cycles with a multi-factor demand-responsive adaptive controller.
- A **Discrete-Time Junction Queuing Simulator** that replays empirical arrivals to evaluate queue build-up, throughput, and delay.
- An **Automated Comparative Evaluation Framework** outputting reproducible metrics (mean queue, maximum delay, throughput improvement, starvation prevention).
- A **FastAPI REST API** exposing operational endpoints for live control, historical telemetry, and data export.
- A **React + Vite + TypeScript Operations Dashboard** featuring real-time junction visualization, time-series charts, and evaluation reporting.

```
+----------------------------------------------------------------------------------+
|                            PUNE CCTV DATASET (Jan 2023)                         |
|   115 CSVs: Alankar Chowk, Jehangir Chowk, RTO Chowk (25 fps Cumulative Tripwire)|
+----------------------------------------------------------------------------------+
                                         |
                                         v
+----------------------------------------------------------------------------------+
|                              DATA INGESTION LAYER                                |
|        (backend/data/loader.py: Timestamp deduplication & Discrete Delta)        |
|        Delta_arrivals = max(0, count(t) - count(t-1))                            |
|        IRC:106 PCU = 1.0*car + 0.5*bike + 2.5*bus + 2.5*truck                     |
+----------------------------------------------------------------------------------+
                                         |
                                         v
+----------------------------------------------------------------------------------+
|                       DISCRETE-TIME SIMULATION & REPLAY                          |
|        (backend/simulation/replay_engine.py & junction_layout.py)                 |
|        Q(t+1) = max(0, Q(t) + Arrivals(t) - Discharge(t) * Green(t))             |
|        W(t) = Q(t) / max(Arrival_Rate, epsilon)   [Little's Law Proxy]            |
+----------------------------------------------------------------------------------+
                    |                                          |
                    v                                          v
+---------------------------------------+  +---------------------------------------+
|        FIXED-TIME BASELINE            |  |          ADAPTIVE CONTROLLER          |
|  - 30s Fixed Green Split              |  |  - Demand Score = wq*Q + wa*A + ...   |
|  - 3s Yellow + 2s All-Red Clearance   |  |  - Dynamic Green: [10s, 55s]          |
|  - Strict Non-Conflicting Sequences   |  |  - Anti-Starvation Override (>90s)    |
|                                       |  |  - Strict Clearance Intervals         |
+---------------------------------------+  +---------------------------------------+
                    \                                          /
                     \--------------------+-------------------/
                                          |
                                          v
+----------------------------------------------------------------------------------+
|                            EVALUATION FRAMEWORK                                  |
|        (backend/evaluation/evaluator.py)                                         |
|        - Side-by-side run on identical empirical time series                     |
|        - Metrics: Mean Queue, Max Wait, Total Throughput, Starvation Count        |
|        - Report generation & JSON/CSV telemetry exports                          |
+----------------------------------------------------------------------------------+
                                          |
                                          v
+----------------------------------------------------------------------------------+
|                             FASTAPI REST BACKEND                                 |
|        (backend/main.py: Port 8000)                                              |
|        /api/traffic/current, /api/controller/step, /api/evaluation/run, etc.     |
+----------------------------------------------------------------------------------+
                                          |
                                          v
+----------------------------------------------------------------------------------+
|                           REACT OPERATIONS DASHBOARD                             |
|        (frontend: Vite + React + TypeScript + Tailwind CSS, Port 5173)           |
|        - 4-Way Animated Junction & Signal Heads                                  |
|        - Live Queue, Rate & Wait KPI Cards                                       |
|        - Real-time Recharts (Queue, Delay, Phase Timeline)                       |
|        - Comparative Evaluation Suite & Decision Explanation Log                 |
+----------------------------------------------------------------------------------+
```

---

## 2. Mathematical Formulations & Safety Invariants

### A. Cumulative Counter Delta Rule
The dataset contains frame-by-frame cumulative counts. To obtain instantaneous arrivals $\Delta C(t)$ without inflation:
$$\Delta C(t) = \max\left(0, C(t) - C(t - \Delta t)\right)$$
If a counter resets (continuation file `_one.csv`), $\Delta C(t)$ treats the reset cleanly to prevent artificial spikes.

### B. Heterogeneous Traffic Weighting (IRC:106 PCU)
$$\text{Demand}_{\text{PCU}}(t) = 1.0 \times \text{car}(t) + 0.5 \times \text{motorbike}(t) + 2.5 \times \text{bus}(t) + 2.5 \times \text{truck}(t)$$

### C. Adaptive Demand Scoring
For each candidate phase $P$, the controller evaluates:
$$\text{Score}(P) = \sum_{a \in P} \left( w_q \cdot \frac{Q_a}{\max_i Q_i} + w_a \cdot \frac{\lambda_a}{\max_i \lambda_i} + w_w \cdot \frac{W_a}{\max_i W_i} + w_t \cdot \frac{T_a}{\max_i T_i} \right)$$
Default weights: $w_q = 0.40$, $w_a = 0.25$, $w_w = 0.20$, $w_t = 0.15$.

### D. Safety Invariants
1. **Clearance Enforceability**: Every phase transition strictly executes:
   - **Yellow Clearance**: 3 seconds
   - **All-Red Clearance**: 2 seconds
2. **Conflict Prevention**: Conflicting movements (e.g. North-South Arterial vs East-West Crossway) can never display green simultaneously.
3. **Starvation Prevention**: Any approach unserved for $T \ge 90\text{s}$ triggers an emergency priority switch as soon as the active phase completes its minimum green ($10\text{s}$).
4. **Green Bounds**: Allocated green durations are clamped to $[10\text{s}, 55\text{s}]$.

