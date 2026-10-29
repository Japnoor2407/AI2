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
}
