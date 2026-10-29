# SIH1607 — AI-Based Adaptive Traffic Signal Optimization Agent

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/Frontend-React_19_Vite_TS-61DAFB?logo=react)](https://react.dev)
[![TailwindCSS](https://img.shields.io/badge/Styling-Tailwind_CSS_v4-38B2AC?logo=tailwind-css)](https://tailwindcss.com)
[![Tests](https://img.shields.io/badge/Tests-22%20Passed-success)]()

## 1. Problem Statement (SIH1607)
Urban intersections in Indian cities suffer from severe congestion, vehicle delay, and excessive emissions due to outdated pre-timed fixed traffic signals that cannot adapt to dynamic, heterogeneous traffic conditions.

This system implements an end-to-end intelligent traffic signal controller that operates directly on empirical traffic surveillance data from Pune, Maharashtra (Pune Traffic Police Commissioner CCTV dataset, Jan 2023). It dynamically allocates green durations based on real-time demand, enforces traffic safety clearance intervals, prevents vehicle starvation, and benchmarks against a conventional fixed-time baseline.

---

## 2. Key Features
- **Empirical Ingestion Engine**: Accurately extracts arrival counts from 25 fps cumulative tripwire counters across 115 CSV datasets from Alankar Chowk, Jehangir Chowk, and RTO Chowk.
- **Indian IRC:106 PCU Standards**: Converts heterogeneous mix of 2-wheelers, cars, buses, and commercial trucks into standard Passenger Car Units.
- **Demand-Responsive Controller**: Dynamically adjusts green splits (10s to 55s) using a 4-factor scoring model (queue length, arrival rate, waiting delay, and starvation timer).
- **Zero-Conflict Safety Invariants**: Strictly prevents conflicting green phases and enforces 3s Yellow + 2s All-Red clearance transitions.
- **Dual-Ring Queuing Simulator**: Discrete-time (1 Hz) simulator comparing baseline fixed-cycle vs adaptive controller under identical demand.
- **Side-by-Side Evaluation & Export**: Measures queue reduction, delay reduction, throughput improvement, and exports full JSON/CSV telemetry reports.
- **Modern Dark Operations Dashboard**: High-contrast operations room UI with interactive junction visualization, live replay controls, and real-time Recharts.

---

## 3. Technology Stack
- **Backend**: Python 3.11+, FastAPI, Uvicorn, Pydantic v2, Pandas, NumPy, Scikit-learn, Pytest.
- **Frontend**: React 19, TypeScript, Vite, Tailwind CSS v4, Recharts, Lucide Icons.
- **Data**: Pune Heterogeneous Traffic Count Dataset (January 11–18, 2023).

---

## 4. Quick Start & Local Run

### Prerequisites
- Python 3.11+
- Node.js v20+ and npm

### Step 1: Run Backend Server
```powershell
# In terminal 1 (project root)
cd D:\AI2
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation will be live at: [http://localhost:8000/docs](http://localhost:8000/docs)

### Step 2: Run Frontend Development Server
```powershell
# In terminal 2
cd D:\AI2\frontend
npm run dev
```
Dashboard will be live at: [http://localhost:5173](http://localhost:5173)

---

## 5. Running Automated Tests
```powershell
cd D:\AI2
python -m pytest backend/tests/test_system.py -v
```
All 22 unit and integration tests verify:
- Cumulative diff non-negativity and dataset adapter robustness
- Layout conflict prevention (conflicting approaches never green simultaneously)
- Clearance timing (yellow and all-red duration enforcement)
- Green time bounds clamping ($10\text{s} \le G \le 55\text{s}$)
- Starvation prevention priority override triggers
- Queue conservation ($Q \ge 0$) under heavy and zero demand

---

## 6. REST API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Health check & simulation status |
| `GET` | `/api/dataset/summary` | Summary of available junctions, cameras, files |
| `GET` | `/api/dataset/schema` | Dataset column definitions & PCU equivalencies |
| `GET` | `/api/traffic/current` | Latest junction state, per-approach queues, signal phase |
| `GET` | `/api/traffic/history` | Historical time-series of queue, delay, throughput |
| `GET` | `/api/controller/status` | Current controller mode, step, decision rationale |
| `POST` | `/api/controller/mode` | Switch between `FIXED_TIME` and `ADAPTIVE` |
| `POST` | `/api/controller/step` | Advance simulation by 1 second |
| `POST` | `/api/simulation/start` | Start/re-initialize simulation session |
| `POST` | `/api/simulation/pause` | Pause/resume simulation playback |
| `POST` | `/api/simulation/reset` | Reset simulation state |
| `POST` | `/api/evaluation/run` | Execute comparative evaluation (Fixed vs Adaptive) |
| `GET` | `/api/evaluation/summary` | Get latest evaluation metrics & improvement % |
| `GET` | `/api/evaluation/export` | Download evaluation report as JSON or CSV |
| `GET` | `/api/decision-log` | Log of controller phase decisions and rationale |

---

## 7. Hackathon Demonstration Script (For SIH Judges)
1. **Open Dashboard**: Navigate to [http://localhost:5173](http://localhost:5173).
2. **Overview Tab**: Click **Start** to begin empirical replay of Pune CCTV data. Observe the live 1-second step counter advancing, KPI cards updating in real time, and the decision rationale explaining why each phase was selected.
3. **Junction Tab**: View the 4-way intersection layout. Notice how conflicting approaches (North/South vs East/West) never turn green simultaneously. Watch the Yellow (3s) and All-Red (2s) safety intervals transition between phase switches.
4. **Monitoring Tab**: View the real-time Queue vs Delay chart and Phase timeline showing responsive phase switching under peak vehicle arrivals.
5. **Evaluation Tab**: Click **Run Evaluation (600s)**. In 5 seconds, the engine runs both the pre-timed baseline and adaptive controller across 600 seconds of identical empirical traffic flow, demonstrating measurable reductions in mean queue length and waiting delay.
6. **Export**: Click **Export JSON** or **Export CSV** to download reproducible benchmark results.

