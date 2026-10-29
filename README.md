# SIH1607 — AI-Based Adaptive Traffic Signal Optimization Agent

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/Frontend-React_19_Vite_TS-61DAFB?logo=react)](https://react.dev)
[![TailwindCSS](https://img.shields.io/badge/Styling-Tailwind_CSS_v4-38B2AC?logo=tailwind-css)](https://tailwindcss.com)
[![Tests](https://img.shields.io/badge/Tests-50%20Passed-success)](https://github.com/Japnoor2407/AI2)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## 1. Problem Statement (SIH1607)
Urban intersections in Indian cities suffer from severe congestion, vehicle delay, and excessive emissions due to outdated pre-timed fixed traffic signals that cannot adapt to dynamic, heterogeneous traffic conditions.

This system implements an end-to-end intelligent traffic signal controller operating directly on empirical traffic surveillance data from Pune, Maharashtra (Pune Traffic Police Commissioner CCTV dataset, Jan 2023). It dynamically allocates green durations based on real-time demand, enforces traffic safety clearance intervals, prevents vehicle starvation, and rigorously benchmarks against both fixed-time and actuated baselines.

---

## 2. Key Capabilities & System Architecture

### 2.1 Multi-Strategy Benchmarking Engine (Phase 1)
- **Fixed-Time Baseline**: Cycle-based signal controller with configurable phase splits, yellow clearance (3s), and all-red clearance (2s).
- **Actuated Baseline**: Standard demand-responsive gap-out and max-out controller with passage time extensions and safe phase transitions.
- **Adaptive Controller (AI)**: Multi-objective scoring model optimizing queue length, arrival rate, waiting delay, and starvation prevention.
- **Scientific Metrics**: Computes mean queue, peak queue, mean delay, median wait (P50), 95th-percentile wait (P95), throughput (PCU), residual queues (unfinished trips), controller execution latency, and relative percentage improvements.

### 2.2 Independent Safety Supervisor Boundary (Phase 2)
- **Zero-Conflict Invariants**: Guarantees no two conflicting movements receive green simultaneously.
- **Clearance Interval Protection**: Preempts any premature termination of yellow or all-red safety clearance.
- **Green Boundary Clamping**: Enforces minimum ($10\text{s}$) and maximum ($55\text{s}$) bounds regardless of optimization scores.
- **Audit Logging**: Logs every supervisory intervention with active constraints and causal explanations at `/api/safety/events`.

### 2.3 Machine Learning Demand Forecaster (Phase 3)
- **Autoregressive Forecaster**: Gradient Boosting & Random Forest models predicting short-term arrivals.
- **Chronological Validation**: Strict 70/15/15 time-series split preventing data leakage.
- **Baseline Comparative Validation**: Evaluates model MAE and RMSE against naive persistence ($y_t = y_{t-1}$) and rolling moving-average baselines.
- **Feature Attribution**: Quantifies predictive weight across lag intervals and velocity differentials.

### 2.4 Indian Traffic Standards (IRC:106)
Converts heterogeneous vehicle classes into standardized Passenger Car Units:
- Two-Wheeler / Motorbike: `0.5 PCU`
- Passenger Car / Auto: `1.0 PCU`
- Bus (PMPML / Interstate): `2.5 PCU`
- Commercial Heavy Truck: `2.5 PCU`

---

## 3. Technology Stack

- **Backend**: Python 3.11+, FastAPI, Uvicorn, Pydantic v2, Pandas, NumPy, Scikit-learn, Joblib, OpenCV Headless, Pytest.
- **Frontend**: React 19, TypeScript, Vite, Tailwind CSS v4, Recharts, Lucide Icons.
- **Data**: Pune Heterogeneous Traffic Count Dataset (Jan 11–18, 2023, 25 fps cumulative tripwire counters).

---

## 4. Quick Start & Local Run

### Prerequisites
- Python 3.11+
- Node.js v20+ and npm

### Terminal 1 — Backend API Server
```powershell
cd D:\AI2
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
- Interactive Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- API Health Status: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

### Terminal 2 — Frontend Operations Center
```powershell
cd D:\AI2\frontend
npm run dev
```
- Dashboard UI: [http://localhost:5173](http://localhost:5173)

---

## 5. Automated Verification & Testing

Run the full pytest suite (33 unit & integration tests):
```powershell
cd D:\AI2
python -m pytest backend/tests/test_system.py -v
```

Tests verify:
- PCU calculations according to Indian IRC:106 standard
- Fixed-Time, Actuated, and Adaptive controller state machines
- Clearance interval transitions (Yellow & All-Red)
- Minimum green and maximum green bounds clamping
- Starvation prevention priority overrides
- Actuated gap-out and max-out termination logic
- Safety supervisor invariant verification and conflict rejection
- Statistical percentile calculations (P50, P95)

---

## 6. REST API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Service health status and simulation step counter |
| `GET` | `/api/controller/modes` | List available controller modes (Fixed, Actuated, Adaptive) |
| `POST` | `/api/controller/mode` | Set active controller mode |
| `POST` | `/api/controller/step` | Advance discrete simulation by 1 second |
| `POST` | `/api/simulation/start` | Start replay simulation session |
| `POST` | `/api/simulation/pause` | Pause / resume replay |
| `POST` | `/api/simulation/reset` | Reset simulation state |
| `GET` | `/api/traffic/current` | Real-time approach queues, arrival rates, signal indications |
| `GET` | `/api/traffic/history` | High-frequency time-series history |
| `GET` | `/api/traffic/aggregated` | 10-minute interval aggregated PCU demand analytics |
| `POST` | `/api/evaluation/run` | Execute 3-controller evaluation (Fixed vs Actuated vs Adaptive) |
| `GET` | `/api/evaluation/full-summary`| Full 3-controller comparative performance matrix |
| `GET` | `/api/evaluation/export` | Download benchmark report as JSON or CSV |
| `GET` | `/api/safety/events` | Audit log of safety supervisor override events |
| `POST` | `/api/ml/train` | Train traffic demand forecasting model on CCTV data |
| `GET` | `/api/ml/metrics` | Forecaster metrics vs persistence baselines |
| `GET` | `/api/ml/predict` | Short-term prediction for next $N$ steps |
| `GET` | `/api/decision-log` | Log of controller phase decisions and rationale |
