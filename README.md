# AI2 — Intelligent Adaptive Traffic Signal Optimization System

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/Frontend-React_19_Vite_TS-61DAFB?logo=react)](https://react.dev)
[![TailwindCSS](https://img.shields.io/badge/Styling-Tailwind_CSS_v4-38B2AC?logo=tailwind-css)](https://tailwindcss.com)
[![Tests](https://img.shields.io/badge/Tests-141%20Passed-success)](https://github.com/Japnoor2407/AI2)
[![Gymnasium](https://img.shields.io/badge/RL-Gymnasium_PPO_SB3-blueviolet)](https://gymnasium.farama.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x_CPU-ee4c2c?logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An intelligent, safety-verified urban traffic signal optimization platform powered by Deep Reinforcement Learning (PPO), mathematical network algorithms, and real empirical traffic camera surveillance data from Pune, India.

---

## 🚦 Overview

Traditional traffic lights operate on rigid, pre-timed timers. They keep motorists waiting at red lights even when cross-streets are completely empty, leading to unnecessary delays, heavy congestion, fuel wastage, and slowed emergency responses.

**AI2** replaces static timers with an intelligent, multi-strategy signal controller:
1. **Listens to real traffic**: Reads vehicle queues and arrival patterns calibrated from real-world Pune CCTV surveillance footage.
2. **Dynamically optimizes green lights**: Employs Deep Reinforcement Learning (PPO), Max-Pressure, and Dynamic Adaptive algorithms to continuously shift green light time to where demand is highest.
3. **Emergency Vehicle Priority (EVP)**: Provides an instant green corridor for ambulances, fire trucks, and police with safe clearance intervals.
4. **Independent Safety Supervisor**: Mathematically guarantees that conflicting green lights can never occur and that yellow/all-red clearance intervals are strictly maintained.
5. **Strategy Comparison Lab**: Allows city planners and evaluators to compare all 5 strategies under **strictly identical traffic demand** to measure exact waiting time saved.

---

## 🌟 Key Features

### 1. Strategy Comparison Lab (Central Benchmark)
Run head-to-head experiments across all 5 traffic control strategies under identical vehicle arrivals:
- **Identical Demand Replay**: Guarantees a 100% fair scientific comparison where every strategy experiences the exact same vehicles at the exact same seconds.
- **Measurable Time Saved**: Calculates real delay reduction per vehicle, total cumulative hours saved, and queue reduction percentages.
- **Fairness & Efficiency Index**: Computes Jain's Fairness Index to ensure side streets are not starved of green light.
- **Plain-English Rationale**: Explains why the winning strategy outperformed traditional timers without confusing jargon.
- **One-Click Export**: Save and download comprehensive benchmark reports in JSON and CSV formats.

### 2. Five Independent Traffic Control Strategies
| Strategy | Type | How It Works |
| :--- | :--- | :--- |
| **Fixed-Time** | Baseline | Traditional predetermined timer cycling on fixed green durations regardless of traffic. |
| **Vehicle Actuated** | Sensor-Driven | Detects approaching vehicles and extends green time only when queues are present. |
| **Dynamic Adaptive** | Multi-Factor | Continuously shifts green time towards roads with the longest queues and waiting times. |
| **Max-Pressure** | Network Theory | Balances upstream queues against downstream exit capacity to maximize intersection throughput. |
| **Deep AI (PPO)** | Reinforcement Learning | Deep neural network policy trained via Proximal Policy Optimization to minimize total network delay. |

### 3. Emergency Vehicle Priority (EVP)
- Instant dispatch for **Ambulance**, **Fire Engine**, and **Police** vehicles approaching from any direction.
- Safely transitions through yellow (3s) and all-red (2s) clearance stages before opening an exclusive green corridor.
- Prevents collisions while minimizing delay for critical emergency responders.

### 4. Zero-Violation Safety Supervisor
- **Hard Safety Invariants**: Conflicting green phases (e.g. North-South and East-West simultaneously) are physically impossible.
- **Uninterruptible Clearance**: Yellow (3s) and all-red (2s) intervals cannot be cut short by any AI algorithm.
- **Min/Max Green Clamping**: Enforces minimum (10s) and maximum (55s) green windows to avoid driver confusion and starvation.

### 5. Empirical Pune CCTV Surveillance Dataset
- Calibrated using real traffic surveillance from **Alankar Chowk, Pune, India** (Jan 2023).
- Uses Indian Road Congress (**IRC:106**) Passenger Car Unit (PCU) standards:
  - 🏍️ Two-Wheeler / Motorbike: `0.5 PCU`
  - 🚗 Passenger Car / Auto: `1.0 PCU`
  - 🚌 Bus: `2.5 PCU`
  - 🚚 Commercial Heavy Truck: `2.5 PCU`

---

## ⚡ Quick Start

### Prerequisites
- **Python 3.11+**
- **Node.js 18+** and **npm**

### 1. Start the Backend API Server
```powershell
# In repository root:
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
- API Health Status: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)
- Interactive API Documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 2. Start the Frontend Dashboard
```powershell
# In a new terminal:
cd frontend
npm run dev
```
- Web Application: [http://localhost:5173](http://localhost:5173)

---

## 🧪 3-Step Demo Walkthrough

1. **Compare All 5 Strategies**:
   - Open [http://localhost:5173](http://localhost:5173). You land on the **Strategy Comparison Lab**.
   - Select a duration (e.g. `15m`), choose a traffic condition (`Normal` or `Rush Hour`), and click **Run Strategy Comparison**.
   - View the overall winner badge, composite efficiency score (0–100), total waiting time saved, and plain-English performance rationale.

2. **Test Live Simulation & Control**:
   - Switch to the **Live Intersection** tab.
   - Choose a control strategy (e.g. **Deep AI (PPO)** or **Max-Pressure**).
   - Click **Start Simulation**. Watch the 2D junction display live signals, countdowns, and vehicle queues in real-time.

3. **Dispatch an Emergency Vehicle**:
   - In the Live Intersection tab, locate the **Emergency Vehicle Priority** panel on the right.
   - Select **Ambulance** and approach **North**, then click **Dispatch Priority Corridor**.
   - Observe the safety supervisor safely transition through yellow clearance before granting an uninterrupted green corridor.

---

## 🛠️ Automated Verification Suite

Run the full pytest suite (141 unit, integration, and safety tests):
```powershell
pytest backend/tests/test_system.py backend/tests/test_rl_environment.py backend/tests/test_sumo_integration.py backend/tests/test_emergency_lab.py -v
```
All 141 tests pass locally on standard CPU without requiring cloud GPUs or paid tooling.

---

## 💻 Tech Stack

- **Backend**: Python 3.11+, FastAPI, Uvicorn, Pydantic v2, Gymnasium, Stable-Baselines3 (PPO), PyTorch (CPU), NumPy, Pandas, Scikit-learn, TraCI / Sumolib, Pytest.
- **Frontend**: React 19, TypeScript, Vite, Tailwind CSS v4, Recharts, Lucide Icons.
- **Traffic Modeling**: IRC:106 Indian PCU Standards, SUMO (Simulation of Urban MObility) adapter with dual-engine fallback to `JunctionSimulator`.

---

## 🔬 Advanced Architecture & RL Details

<details>
<summary>Click to view PPO Reinforcement Learning Specifications</summary>

### Custom Gymnasium Environment (`TrafficSignalEnv`)
- **Observation Space (21 dimensions, Box[0.0, 1.0])**:
  - Normalized queue lengths per approach (4)
  - Normalized vehicle arrival rates (4)
  - Normalized average waiting delays (4)
  - Approach starvation timers (4)
  - Active signal phase one-hot encoding (2)
  - Elapsed phase duration (1)
  - Overall intersection density (1)
  - Episode progress ratio (1)
- **Action Space (Discrete 16)**:
  - Phase selection: Phase 1 (North-South) or Phase 2 (East-West)
  - Green duration selection: 8 intervals ($10\text{s}, 15\text{s}, 20\text{s}, 25\text{s}, 30\text{s}, 40\text{s}, 50\text{s}, 55\text{s}$)
- **Reward Function**:
  $$R = 0.35 \cdot \Delta Q + 0.30 \cdot T - 0.20 \cdot W - 0.10 \cdot S - 0.05 \cdot P_{\text{switch}} - 0.05 \cdot P_{\text{balance}}$$
  Where $\Delta Q$ is queue clearance, $T$ is throughput, $W$ is waiting delay penalty, $S$ is starvation overage, $P_{\text{switch}}$ penalizes rapid phase toggling, and $P_{\text{balance}}$ penalizes variance across approaches.

### Training Command
```powershell
python -m backend.rl.train --timesteps 100000 --episode-duration 900 --checkpoint-freq 10000
```
Trained model checkpoints are stored in `backend/models/rl/ppo_traffic_latest.zip`.
</details>

<details>
<summary>Click to view REST API Reference</summary>

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Service health status and simulation step counter |
| `GET` | `/api/controller/modes` | List available controller modes |
| `POST` | `/api/controller/mode` | Set active controller mode |
| `POST` | `/api/controller/step` | Advance discrete simulation by 1 second |
| `POST` | `/api/simulation/start` | Start replay simulation session |
| `POST` | `/api/simulation/pause` | Pause / resume replay |
| `POST` | `/api/simulation/reset` | Reset simulation state |
| `GET` | `/api/traffic/current` | Real-time approach queues, arrival rates, signal indications |
| `GET` | `/api/traffic/history` | High-frequency time-series history |
| `POST` | `/api/lab/run` | Execute multi-strategy benchmark under strictly identical demand |
| `GET` | `/api/lab/history` | Historical comparison lab sessions list |
| `GET` | `/api/lab/report/{id}` | Detailed strategy benchmark report with time-saved metrics |
| `GET` | `/api/lab/export/{id}` | Download comparison lab session as CSV or JSON |
| `POST` | `/api/emergency/request` | Dispatch emergency vehicle preemption (Ambulance, Fire Truck, Police) |
| `POST` | `/api/emergency/clear` | Clear active emergency priority |
| `GET` | `/api/emergency/status` | Current active emergency priority status |
| `GET` | `/api/emergency/events` | Audit log of past emergency vehicle preemption events |
| `GET` | `/api/safety/events` | Audit log of safety supervisor override events |
| `POST` | `/api/rl/train` | Trigger background PPO model training |
| `GET` | `/api/rl/training/status` | Real-time training progress and timesteps |
</details>

---

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
