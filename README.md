# SIH1607 — AI-Based Adaptive Traffic Signal Optimization Agent

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/Frontend-React_19_Vite_TS-61DAFB?logo=react)](https://react.dev)
[![TailwindCSS](https://img.shields.io/badge/Styling-Tailwind_CSS_v4-38B2AC?logo=tailwind-css)](https://tailwindcss.com)
[![Tests](https://img.shields.io/badge/Tests-141%20Passed-success)](https://github.com/Japnoor2407/AI2)
[![Gymnasium](https://img.shields.io/badge/RL-Gymnasium_PPO_SB3-blueviolet)](https://gymnasium.farama.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x_CPU-ee4c2c?logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## 1. Problem Statement (SIH1607)
Urban intersections in Indian cities suffer from severe congestion, vehicle delay, and excessive emissions due to outdated pre-timed fixed traffic signals that cannot adapt to dynamic, heterogeneous traffic conditions.

This system implements an end-to-end intelligent traffic signal controller operating directly on empirical traffic surveillance data from Pune, Maharashtra (Pune Traffic Police Commissioner CCTV dataset, Jan 2023). It dynamically allocates green durations based on real-time demand, enforces traffic safety clearance intervals, prevents vehicle starvation, and rigorously benchmarks against both fixed-time and actuated baselines.

---

## 2. Key Capabilities & System Architecture

### 2.1 Multi-Strategy Benchmarking Engine
The system supports 5 independently selectable and testable traffic control strategies:
- **Fixed-Time Baseline**: Cycle-based signal controller with configurable phase splits, yellow clearance (3s), and all-red clearance (2s).
- **Actuated Baseline**: Standard demand-responsive gap-out and max-out controller with passage time extensions and safe phase transitions.
- **Weighted Adaptive (AI)**: Multi-objective scoring model optimizing queue length, arrival rate, waiting delay, and starvation prevention.
- **Max-Pressure Controller**: Advanced feedback control evaluating input vs output queue pressure differences and prioritizing maximal congestion relief.
- **Deep Reinforcement Learning (PPO)**: Neural network policy trained via Proximal Policy Optimization across repeated simulation episodes.

### 2.2 Strategy Comparison Lab
- **Guaranteed Identical Demand**: Evaluates all selected controllers sequentially against the exact same vehicle arrival sequence, timestamps, and random seed.
- **Configurable Durations**: 5m (300s), 10m (600s), 15m (900s), 30m (1800s), 45m (2700s), 60m (3600s).
- **Exact Time-Saved Calculation**:
  - Average waiting time saved per vehicle: $\Delta W = W_{\text{fixed}} - W_{\text{strategy}}$
  - Total travel time saved: $T_{\text{saved}} = \Delta W \times N_{\text{vehicles}}$
  - Percentage wait time reduction: $\frac{W_{\text{fixed}} - W_{\text{strategy}}}{W_{\text{fixed}}} \times 100\%$
- **Multi-Metric Composite Ranking**: Ranks controllers from #1 to #5 based on balanced delay reduction, queue reduction, throughput, and starvation penalties, accompanied by readable plain-English justifications.
- **Session Persistence & Export**: Automatically saves reports to disk with one-click CSV and JSON downloads.

### 2.3 Emergency Vehicle Priority (EVP) System
- **Conflict-Free Preemption**: Detects approaching emergency vehicles (Ambulance, Fire Engine, Police) and prioritizes their movement.
- **Safe Clearance Protocols**: Transitions smoothly through yellow (3s) and all-red (2s) clearance intervals before granting priority green.
- **Green Extension**: Holds green until the emergency vehicle clears, then returns control smoothly to the active background controller.
- **Audit Logging**: Full audit trail of EVP dispatches, approach paths, and clearance latencies.

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

### 2.4 Deep Reinforcement Learning Controller (PPO)
- **Custom Gymnasium Environment**: Standard-compliant `TrafficSignalEnv` driving a four-way intersection with real Pune CCTV arrival dynamics.
- **21-Dimensional Observation Space**: Normalized queue lengths (4), arrival rates (4), mean delays (4), starvation timers (4), active phase one-hot (2), elapsed phase duration (1), intersection density (1), and episode progress (1).
- **16-Discrete Action Space**: Combinatorial phase selection ($\text{Phase 1: NS}, \text{Phase 2: EW}$) paired with 8 green durations ($10\text{s}, 15\text{s}, 20\text{s}, 25\text{s}, 30\text{s}, 40\text{s}, 50\text{s}, 55\text{s}$).
- **Multi-Objective Reward Function**:
  $$R = 0.35 \cdot \Delta Q + 0.30 \cdot T - 0.20 \cdot W - 0.10 \cdot S - 0.05 \cdot P_{\text{switch}} - 0.05 \cdot P_{\text{balance}}$$
  where $\Delta Q$ is queue improvement, $T$ is throughput discharged, $W$ is mean wait penalty, $S$ is starvation overage, $P_{\text{switch}}$ discourages rapid oscillations, and $P_{\text{balance}}$ minimizes queue variance.
- **Safety Guarantee**: Every neural inference output is validated and clamped by the `SafetySupervisor` before signal actuation.
- **SUMO / TraCI Interoperability**: `SimulationAdapter` auto-detects local SUMO binaries on PATH and seamlessly falls back to `JunctionSimulator` on any standard CPU.

### 2.5 Indian Traffic Standards (IRC:106)
Converts heterogeneous vehicle classes into standardized Passenger Car Units:
- Two-Wheeler / Motorbike: `0.5 PCU`
- Passenger Car / Auto: `1.0 PCU`
- Bus (PMPML / Interstate): `2.5 PCU`
- Commercial Heavy Truck: `2.5 PCU`

---

## 3. Technology Stack

- **Backend**: Python 3.11+, FastAPI, Uvicorn, Pydantic v2, Pandas, NumPy, Scikit-learn, Joblib, PyTorch (CPU), Gymnasium, Stable-Baselines3, TraCI, Sumolib, Pytest.
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

Run the full pytest suite (141 unit & integration tests):
```powershell
cd D:\AI2
python -m pytest backend/tests/test_system.py backend/tests/test_rl_environment.py backend/tests/test_sumo_integration.py backend/tests/test_emergency_lab.py -v
```

Tests verify:
- PCU calculations according to Indian IRC:106 standard
- Fixed-Time, Actuated, Adaptive, and Deep RL controller state machines
- Clearance interval transitions (Yellow & All-Red)
- Minimum green and maximum green bounds clamping
- Starvation prevention priority overrides
- Actuated gap-out and max-out termination logic
- Safety supervisor invariant verification and conflict rejection
- Statistical percentile calculations (P50, P95)
- Gymnasium `check_env` conformance for `TrafficSignalEnv`
- Observation space bounds, action decoding/encoding roundtrips, reward calculation breakdowns
- Stable-Baselines3 PPO initialization, prediction, and reproducibility across seeds
- SUMO network generation, nodes, edges, connections, traffic light programs, and `.sumocfg` configuration
- Pune CCTV demand conversion to individual vehicle route definitions with heterogeneous Indian traffic classes
- `SimulationAdapter` dual-engine execution and transparent fallback lifecycle management

---

## 6. Deep Reinforcement Learning Training

Train the PPO agent from the command line:
```powershell
cd D:\AI2
python -m backend.rl.train --timesteps 100000 --episode-duration 900 --checkpoint-freq 10000
```
Trained model checkpoints are saved to `backend/models/rl/ppo_traffic_latest.zip`.

---

## 7. REST API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Service health status and simulation step counter |
| `GET` | `/api/controller/modes` | List available controller modes (Fixed, Actuated, Adaptive, AI (PPO)) |
| `POST` | `/api/controller/mode` | Set active controller mode |
| `POST` | `/api/controller/step` | Advance discrete simulation by 1 second |
| `POST` | `/api/simulation/start` | Start replay simulation session |
| `POST` | `/api/simulation/pause` | Pause / resume replay |
| `POST` | `/api/simulation/reset` | Reset simulation state |
| `GET` | `/api/traffic/current` | Real-time approach queues, arrival rates, signal indications |
| `GET` | `/api/traffic/history` | High-frequency time-series history |
| `GET` | `/api/traffic/aggregated` | 10-minute interval aggregated PCU demand analytics |
| `POST` | `/api/evaluation/run` | Execute 4-controller evaluation (Fixed vs Actuated vs Adaptive vs Deep RL) |
| `GET` | `/api/evaluation/full-summary`| Full comparative performance matrix |
| `GET` | `/api/evaluation/export` | Download benchmark report as JSON or CSV |
| `GET` | `/api/safety/events` | Audit log of safety supervisor override events |
| `POST` | `/api/ml/train` | Train traffic demand forecasting model on CCTV data |
| `GET` | `/api/ml/metrics` | Forecaster metrics vs persistence baselines |
| `GET` | `/api/ml/predict` | Short-term prediction for next $N$ steps |
| `GET` | `/api/decision-log` | Log of controller phase decisions and rationale |
| `GET` | `/api/rl/status` | Deep RL model status, availability, and metadata |
| `POST` | `/api/rl/train` | Trigger background PPO model training |
| `GET` | `/api/rl/training/status` | Real-time training progress, timesteps, and rewards |
| `POST` | `/api/rl/predict` | Single-step PPO policy action inference |
| `GET` | `/api/rl/metrics` | RL training history and checkpoint directory |
| `POST` | `/api/rl/evaluate` | Comparative evaluation benchmark with PPO policy |
| `GET` | `/api/rl/comparison` | Strategy comparison summary including RL performance |
| `POST` | `/api/emergency/request` | Dispatch emergency vehicle preemption (Ambulance, Fire Truck, Police) |
| `POST` | `/api/emergency/clear` | Clear active emergency priority and restore background controller |
| `GET` | `/api/emergency/status` | Current active emergency vehicle priority status |
| `GET` | `/api/emergency/events` | Audit log of past emergency vehicle preemption events |
| `POST` | `/api/lab/run` | Execute multi-strategy benchmark under strictly identical demand |
| `GET` | `/api/lab/history` | Historical comparison lab sessions list |
| `GET` | `/api/lab/report/{id}` | Detailed strategy benchmark report with time-saved metrics |
| `GET` | `/api/lab/export/{id}` | Download comparison lab session as CSV or JSON |
