// API service — connects to FastAPI backend
const BASE = 'http://localhost:8000'

async function apiFetch(path: string, options?: RequestInit) {
  const res = await fetch(`${BASE}${path}`, options)
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || res.statusText)
  }
  return res.json()
}

export const api = {
  health: () => apiFetch('/api/health'),
  datasetSummary: () => apiFetch('/api/dataset/summary'),
  datasetSchema: () => apiFetch('/api/dataset/schema'),
  trafficCurrent: () => apiFetch('/api/traffic/current'),
  trafficHistory: (limit = 300) => apiFetch(`/api/traffic/history?limit=${limit}`),
  trafficAggregated: (intervalSec = 600) => apiFetch(`/api/traffic/aggregated?interval_sec=${intervalSec}`),
  controllerStatus: () => apiFetch('/api/controller/status'),
  controllerModes: () => apiFetch('/api/controller/modes'),
  setMode: (mode: string) =>
    apiFetch('/api/controller/mode', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode }),
    }),
  controllerStep: () =>
    apiFetch('/api/controller/step', { method: 'POST' }),
  simulationStart: (params?: object) =>
    apiFetch('/api/simulation/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: params ? JSON.stringify(params) : undefined,
    }),
  simulationPause: () => apiFetch('/api/simulation/pause', { method: 'POST' }),
  simulationReset: () => apiFetch('/api/simulation/reset', { method: 'POST' }),
  simulationSeek: (targetStep: number) =>
    apiFetch('/api/simulation/seek', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ target_step: targetStep }),
    }),
  evaluationSummary: () => apiFetch('/api/evaluation/summary'),
  evaluationFullSummary: () => apiFetch('/api/evaluation/full-summary'),
  evaluationRun: (params?: object) =>
    apiFetch('/api/evaluation/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: params ? JSON.stringify(params) : undefined,
    }),
  evaluationExport: (format: 'json' | 'csv') =>
    fetch(`${BASE}/api/evaluation/export?format=${format}`),
  decisionLog: (limit = 100) => apiFetch(`/api/decision-log?limit=${limit}`),
  safetyEvents: (limit = 200) => apiFetch(`/api/safety/events?limit=${limit}`),
  replayHistory: (limit = 500) => apiFetch(`/api/replay/history?limit=${limit}`),
  mlTrain: () => apiFetch('/api/ml/train', { method: 'POST' }),
  mlMetrics: () => apiFetch('/api/ml/metrics'),
  mlPredict: (horizon = 30) => apiFetch(`/api/ml/predict?horizon_steps=${horizon}`),

  // ── Deep RL (PPO) endpoints ──────────────────────────────────────────────
  rlStatus: () => apiFetch('/api/rl/status'),
  rlTrain: (params?: {
    total_timesteps?: number;
    episode_duration_sec?: number;
    n_envs?: number;
    checkpoint_freq?: number;
    seed?: number;
    resume?: boolean;
    simulator?: string;
    sumo_gui?: boolean;
  }) =>
    apiFetch('/api/rl/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params ?? {}),
    }),
  rlTrainingStatus: () => apiFetch('/api/rl/training/status'),
  rlPredict: (observation: number[], deterministic = true) =>
    apiFetch('/api/rl/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ observation, deterministic }),
    }),
  rlMetrics: () => apiFetch('/api/rl/metrics'),
  rlEvaluate: (params?: { duration_sec?: number; seed?: number }) =>
    apiFetch('/api/rl/evaluate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params ?? {}),
    }),
  rlComparison: () => apiFetch('/api/rl/comparison'),

  // ── Emergency Vehicle Priority (EVP) ────────────────────────────────────
  emergencyRequest: (vehicle_type: string, approach_id: string) =>
    apiFetch('/api/emergency/request', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ vehicle_type, approach_id }),
    }),
  emergencyClear: () =>
    apiFetch('/api/emergency/clear', { method: 'POST' }),
  emergencyStatus: () => apiFetch('/api/emergency/status'),
  emergencyEvents: () => apiFetch('/api/emergency/events'),

  // ── Strategy Comparison Lab ──────────────────────────────────────────────
  labRun: (params?: {
    junction_id?: string;
    date?: string;
    duration_sec?: number;
    scenario?: string;
    demand_multiplier?: number;
    selected_strategies?: string[];
    seed?: number;
  }) =>
    apiFetch('/api/lab/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params ?? {}),
    }),
  labHistory: (limit = 20) => apiFetch(`/api/lab/history?limit=${limit}`),
  labReportDetail: (sessionId: string) => apiFetch(`/api/lab/report/${sessionId}`),
  labExportUrl: (sessionId: string, format: 'json' | 'csv') =>
    `${BASE}/api/lab/export/${sessionId}?format=${format}`,
}
