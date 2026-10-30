import { useState, useEffect } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts'
import { Play, Download, CheckCircle2, AlertCircle, RefreshCw } from 'lucide-react'
import { api } from '../services/api'
import type { FullEvaluationSummary } from '../types'

export function BenchmarkPanel() {
  const [data, setData] = useState<FullEvaluationSummary | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [scenario, setScenario] = useState('normal')
  const [duration, setDuration] = useState(900)
  const [demandMultiplier, setDemandMultiplier] = useState(1.0)

  const fetchSummary = async () => {
    try {
      const res = await api.evaluationFullSummary()
      if (res && res.status === 'ok') {
        setData(res)
      }
    } catch {
      // not evaluated yet is expected initially
    }
  }

  useEffect(() => {
    fetchSummary()
  }, [])

  const handleRunEvaluation = async () => {
    setLoading(true)
    setError(null)
    try {
      await api.evaluationRun({
        junction_id: 'AlankarChowk',
        date: '11',
        duration_sec: duration,
        scenario,
        demand_multiplier: demandMultiplier,
      })
      await fetchSummary()
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Evaluation failed')
    } finally {
      setLoading(false)
    }
  }

  const handleExport = (format: 'json' | 'csv') => {
    window.open(`http://localhost:8000/api/evaluation/export?format=${format}`, '_blank')
  }

  const chartData = data?.fixed_time && data?.adaptive ? [
    {
      metric: 'Mean Queue (PCU)',
      'Fixed-Time': data.fixed_time.mean_queue_pcu,
      'Actuated': data.actuated?.mean_queue_pcu ?? 0,
      'Adaptive': data.adaptive.mean_queue_pcu,
      ...(data.deep_rl ? { 'Deep RL (PPO)': data.deep_rl.mean_queue_pcu } : {}),
    },
    {
      metric: 'Avg Wait (s)',
      'Fixed-Time': data.fixed_time.average_wait_sec,
      'Actuated': data.actuated?.average_wait_sec ?? 0,
      'Adaptive': data.adaptive.average_wait_sec,
      ...(data.deep_rl ? { 'Deep RL (PPO)': data.deep_rl.average_wait_sec } : {}),
    },
    {
      metric: 'P95 Wait (s)',
      'Fixed-Time': data.fixed_time.p95_wait_sec,
      'Actuated': data.actuated?.p95_wait_sec ?? 0,
      'Adaptive': data.adaptive.p95_wait_sec,
      ...(data.deep_rl ? { 'Deep RL (PPO)': data.deep_rl.p95_wait_sec } : {}),
    },
    {
      metric: 'Residual Queue (PCU)',
      'Fixed-Time': data.fixed_time.residual_queue_pcu,
      'Actuated': data.actuated?.residual_queue_pcu ?? 0,
      'Adaptive': data.adaptive.residual_queue_pcu,
      ...(data.deep_rl ? { 'Deep RL (PPO)': data.deep_rl.residual_queue_pcu } : {}),
    },
  ] : []

  const formatPct = (val?: number | null) => {
    if (val === undefined || val === null) return '—'
    const sign = val > 0 ? '+' : ''
    const color = val > 0 ? 'text-emerald-400' : val < 0 ? 'text-rose-400' : 'text-slate-400'
    return <span className={`font-mono font-bold ${color}`}>{sign}{val.toFixed(1)}%</span>
  }

  return (
    <div className="space-y-6">
      {/* Control Header */}
      <div className="card">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h2 className="text-base font-bold text-slate-100 flex items-center gap-2">
              <span>Scientific Controller Benchmarking</span>
              <span className="badge badge-cyan text-[10px]">3-Strategy Comparison</span>
            </h2>
            <p className="text-xs text-slate-400 mt-1">
              Side-by-side evaluation of Fixed-Time, Actuated (Gap/Max-out), and Adaptive controllers under identical empirical demand.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => handleExport('csv')}
              disabled={!data || data.status !== 'ok'}
              className="btn btn-secondary text-xs flex items-center gap-1.5"
            >
              <Download className="w-3.5 h-3.5" /> CSV
            </button>
            <button
              onClick={() => handleExport('json')}
              disabled={!data || data.status !== 'ok'}
              className="btn btn-secondary text-xs flex items-center gap-1.5"
            >
              <Download className="w-3.5 h-3.5" /> JSON
            </button>
            <button
              onClick={handleRunEvaluation}
              disabled={loading}
              className="btn btn-primary text-xs flex items-center gap-2"
            >
              {loading ? (
                <RefreshCw className="w-3.5 h-3.5 animate-spin" />
              ) : (
                <Play className="w-3.5 h-3.5 fill-current" />
              )}
              {loading ? 'Evaluating...' : 'Run Benchmark'}
            </button>
          </div>
        </div>

        {/* Experiment Configuration */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mt-4 pt-4 border-t border-slate-800 text-xs">
          <div>
            <label className="text-slate-400 font-semibold block mb-1">Traffic Scenario</label>
            <select
              value={scenario}
              onChange={(e) => {
                setScenario(e.target.value)
                if (e.target.value === 'peak') setDemandMultiplier(1.6)
                else if (e.target.value === 'surge') setDemandMultiplier(2.2)
                else if (e.target.value === 'low') setDemandMultiplier(0.5)
                else setDemandMultiplier(1.0)
              }}
              className="w-full bg-slate-900 border border-slate-700 rounded px-2.5 py-1.5 text-slate-200"
            >
              <option value="normal">Empirical CCTV (1.0x)</option>
              <option value="peak">Peak Hour Surge (1.6x)</option>
              <option value="surge">Extreme Congestion (2.2x)</option>
              <option value="low">Off-Peak Flow (0.5x)</option>
            </select>
          </div>

          <div>
            <label className="text-slate-400 font-semibold block mb-1">Evaluation Duration</label>
            <select
              value={duration}
              onChange={(e) => setDuration(Number(e.target.value))}
              className="w-full bg-slate-900 border border-slate-700 rounded px-2.5 py-1.5 text-slate-200"
            >
              <option value={300}>5 Minutes (300s)</option>
              <option value={600}>10 Minutes (600s)</option>
              <option value={900}>15 Minutes (900s)</option>
              <option value={1800}>30 Minutes (1800s)</option>
            </select>
          </div>

          <div>
            <label className="text-slate-400 font-semibold block mb-1">Demand Multiplier</label>
            <input
              type="number"
              step="0.1"
              min="0.2"
              max="3.0"
              value={demandMultiplier}
              onChange={(e) => setDemandMultiplier(parseFloat(e.target.value) || 1.0)}
              className="w-full bg-slate-900 border border-slate-700 rounded px-2.5 py-1.5 text-slate-200 font-mono"
            />
          </div>
        </div>

        {error && (
          <div className="mt-3 p-2 bg-rose-950/60 border border-rose-800 rounded text-rose-300 text-xs flex items-center gap-2">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}
      </div>

      {data && data.status === 'ok' ? (
        <>
          {/* Summary Banner */}
          <div className="p-3 bg-cyan-950/40 border border-cyan-800/60 rounded-lg text-xs text-cyan-200 flex items-start gap-2.5">
            <CheckCircle2 className="w-4 h-4 text-cyan-400 shrink-0 mt-0.5" />
            <div>
              <span className="font-semibold">{data.summary}</span>
              <div className="text-[11px] text-cyan-400/80 mt-0.5">
                Evaluated: {new Date(data.evaluated_at || '').toLocaleTimeString()} · Duration: {data.duration_sec}s · Junction: {data.junction}
              </div>
            </div>
          </div>

          {/* Metrics Comparison Table */}
          <div className="card overflow-x-auto">
            <div className="text-xs font-bold text-slate-300 uppercase tracking-wider mb-3">
              Performance Matrix (Exact Empirical Replay)
            </div>
            <table className="w-full text-xs text-left">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400">
                  <th className="py-2 px-3 font-semibold">Evaluation Metric</th>
                  <th className="py-2 px-3 font-semibold text-right">Fixed-Time Baseline</th>
                  <th className="py-2 px-3 font-semibold text-right">Actuated Baseline</th>
                  <th className="py-2 px-3 font-semibold text-right text-cyan-400">Adaptive (AI)</th>
                  {data.deep_rl && <th className="py-2 px-3 font-semibold text-right text-purple-400">Deep RL (PPO)</th>}
                  <th className="py-2 px-3 font-semibold text-right">vs Fixed-Time</th>
                  <th className="py-2 px-3 font-semibold text-right">vs Actuated</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono">
                <tr>
                  <td className="py-2.5 px-3 font-sans font-medium text-slate-200">Mean Queue Length</td>
                  <td className="py-2.5 px-3 text-right text-slate-300">{data.fixed_time?.mean_queue_pcu.toFixed(2)} PCU</td>
                  <td className="py-2.5 px-3 text-right text-slate-300">{data.actuated?.mean_queue_pcu.toFixed(2)} PCU</td>
                  <td className="py-2.5 px-3 text-right text-cyan-300 font-bold">{data.adaptive?.mean_queue_pcu.toFixed(2)} PCU</td>
                  {data.deep_rl && <td className="py-2.5 px-3 text-right text-purple-300 font-bold">{data.deep_rl.mean_queue_pcu.toFixed(2)} PCU</td>}
                  <td className="py-2.5 px-3 text-right">{formatPct(data.improvements_vs_fixed?.queue_reduction_percent)}</td>
                  <td className="py-2.5 px-3 text-right">{formatPct(data.improvements_vs_actuated?.queue_reduction_percent)}</td>
                </tr>
                <tr>
                  <td className="py-2.5 px-3 font-sans font-medium text-slate-200">Peak Queue Length</td>
                  <td className="py-2.5 px-3 text-right text-slate-300">{data.fixed_time?.max_queue_pcu.toFixed(2)} PCU</td>
                  <td className="py-2.5 px-3 text-right text-slate-300">{data.actuated?.max_queue_pcu.toFixed(2)} PCU</td>
                  <td className="py-2.5 px-3 text-right text-cyan-300 font-bold">{data.adaptive?.max_queue_pcu.toFixed(2)} PCU</td>
                  {data.deep_rl && <td className="py-2.5 px-3 text-right text-purple-300 font-bold">{data.deep_rl.max_queue_pcu.toFixed(2)} PCU</td>}
                  <td className="py-2.5 px-3 text-right text-slate-500">—</td>
                  <td className="py-2.5 px-3 text-right text-slate-500">—</td>
                </tr>
                <tr>
                  <td className="py-2.5 px-3 font-sans font-medium text-slate-200">Mean Wait Time</td>
                  <td className="py-2.5 px-3 text-right text-slate-300">{data.fixed_time?.average_wait_sec.toFixed(1)}s</td>
                  <td className="py-2.5 px-3 text-right text-slate-300">{data.actuated?.average_wait_sec.toFixed(1)}s</td>
                  <td className="py-2.5 px-3 text-right text-cyan-300 font-bold">{data.adaptive?.average_wait_sec.toFixed(1)}s</td>
                  {data.deep_rl && <td className="py-2.5 px-3 text-right text-purple-300 font-bold">{data.deep_rl.average_wait_sec.toFixed(1)}s</td>}
                  <td className="py-2.5 px-3 text-right">{formatPct(data.improvements_vs_fixed?.waiting_time_reduction_percent)}</td>
                  <td className="py-2.5 px-3 text-right">{formatPct(data.improvements_vs_actuated?.waiting_time_reduction_percent)}</td>
                </tr>
                <tr>
                  <td className="py-2.5 px-3 font-sans font-medium text-slate-200">Median Wait (P50)</td>
                  <td className="py-2.5 px-3 text-right text-slate-300">{data.fixed_time?.median_wait_sec.toFixed(1)}s</td>
                  <td className="py-2.5 px-3 text-right text-slate-300">{data.actuated?.median_wait_sec.toFixed(1)}s</td>
                  <td className="py-2.5 px-3 text-right text-cyan-300 font-bold">{data.adaptive?.median_wait_sec.toFixed(1)}s</td>
                  {data.deep_rl && <td className="py-2.5 px-3 text-right text-purple-300 font-bold">{data.deep_rl.median_wait_sec.toFixed(1)}s</td>}
                  <td className="py-2.5 px-3 text-right text-slate-500">—</td>
                  <td className="py-2.5 px-3 text-right text-slate-500">—</td>
                </tr>
                <tr>
                  <td className="py-2.5 px-3 font-sans font-medium text-slate-200">95th-Percentile Wait (P95)</td>
                  <td className="py-2.5 px-3 text-right text-slate-300">{data.fixed_time?.p95_wait_sec.toFixed(1)}s</td>
                  <td className="py-2.5 px-3 text-right text-slate-300">{data.actuated?.p95_wait_sec.toFixed(1)}s</td>
                  <td className="py-2.5 px-3 text-right text-cyan-300 font-bold">{data.adaptive?.p95_wait_sec.toFixed(1)}s</td>
                  {data.deep_rl && <td className="py-2.5 px-3 text-right text-purple-300 font-bold">{data.deep_rl.p95_wait_sec.toFixed(1)}s</td>}
                  <td className="py-2.5 px-3 text-right text-slate-500">—</td>
                  <td className="py-2.5 px-3 text-right text-slate-500">—</td>
                </tr>
                <tr>
                  <td className="py-2.5 px-3 font-sans font-medium text-slate-200">Total Throughput</td>
                  <td className="py-2.5 px-3 text-right text-slate-300">{data.fixed_time?.total_throughput_pcu.toFixed(1)} PCU</td>
                  <td className="py-2.5 px-3 text-right text-slate-300">{data.actuated?.total_throughput_pcu.toFixed(1)} PCU</td>
                  <td className="py-2.5 px-3 text-right text-cyan-300 font-bold">{data.adaptive?.total_throughput_pcu.toFixed(1)} PCU</td>
                  {data.deep_rl && <td className="py-2.5 px-3 text-right text-purple-300 font-bold">{data.deep_rl.total_throughput_pcu.toFixed(1)} PCU</td>}
                  <td className="py-2.5 px-3 text-right">{formatPct(data.improvements_vs_fixed?.throughput_improvement_percent)}</td>
                  <td className="py-2.5 px-3 text-right">{formatPct(data.improvements_vs_actuated?.throughput_improvement_percent)}</td>
                </tr>
                <tr>
                  <td className="py-2.5 px-3 font-sans font-medium text-slate-200">Residual Queue (End of Run)</td>
                  <td className="py-2.5 px-3 text-right text-slate-300">{data.fixed_time?.residual_queue_pcu.toFixed(2)} PCU</td>
                  <td className="py-2.5 px-3 text-right text-slate-300">{data.actuated?.residual_queue_pcu.toFixed(2)} PCU</td>
                  <td className="py-2.5 px-3 text-right text-cyan-300 font-bold">{data.adaptive?.residual_queue_pcu.toFixed(2)} PCU</td>
                  {data.deep_rl && <td className="py-2.5 px-3 text-right text-purple-300 font-bold">{data.deep_rl.residual_queue_pcu.toFixed(2)} PCU</td>}
                  <td className="py-2.5 px-3 text-right text-slate-500">—</td>
                  <td className="py-2.5 px-3 text-right text-slate-500">—</td>
                </tr>
                <tr>
                  <td className="py-2.5 px-3 font-sans font-medium text-slate-200">Starvation Occurrences</td>
                  <td className="py-2.5 px-3 text-right text-slate-300">{data.fixed_time?.starvation_occurrences ?? 0}</td>
                  <td className="py-2.5 px-3 text-right text-slate-300">{data.actuated?.starvation_occurrences ?? 0}</td>
                  <td className="py-2.5 px-3 text-right text-cyan-300 font-bold">{data.adaptive?.starvation_occurrences ?? 0}</td>
                  {data.deep_rl && <td className="py-2.5 px-3 text-right text-purple-300 font-bold">{data.deep_rl.starvation_occurrences ?? 0}</td>}
                  <td className="py-2.5 px-3 text-right text-slate-500">—</td>
                  <td className="py-2.5 px-3 text-right text-slate-500">—</td>
                </tr>
                <tr>
                  <td className="py-2.5 px-3 font-sans font-medium text-slate-200">Total Compute Time</td>
                  <td className="py-2.5 px-3 text-right text-slate-300">{data.fixed_time?.execution_time_ms.toFixed(1)}ms</td>
                  <td className="py-2.5 px-3 text-right text-slate-300">{data.actuated?.execution_time_ms.toFixed(1)}ms</td>
                  <td className="py-2.5 px-3 text-right text-cyan-300 font-bold">{data.adaptive?.execution_time_ms.toFixed(1)}ms</td>
                  {data.deep_rl && <td className="py-2.5 px-3 text-right text-purple-300 font-bold">{data.deep_rl.execution_time_ms.toFixed(1)}ms</td>}
                  <td className="py-2.5 px-3 text-right text-slate-500">—</td>
                  <td className="py-2.5 px-3 text-right text-slate-500">—</td>
                </tr>
              </tbody>
            </table>
          </div>

          {/* Bar Chart Comparison */}
          <div className="card">
            <div className="text-xs font-bold text-slate-300 uppercase tracking-wider mb-4">
              Side-by-Side Strategy Comparison
            </div>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 10, right: 20, left: -10, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="metric" stroke="#64748b" tick={{ fontSize: 11 }} />
                  <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', fontSize: '12px' }}
                    itemStyle={{ color: '#e2e8f0' }}
                  />
                  <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
                  <Bar dataKey="Fixed-Time" fill="#64748b" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="Actuated" fill="#eab308" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="Adaptive" fill="#06b6d4" radius={[4, 4, 0, 0]} />
                  {data?.deep_rl && <Bar dataKey="Deep RL (PPO)" fill="#a855f7" radius={[4, 4, 0, 0]} />}
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </>
      ) : (
        <div className="card text-center py-12">
          <div className="text-slate-500 text-sm mb-3">No benchmark experiment has been executed yet.</div>
          <button
            onClick={handleRunEvaluation}
            disabled={loading}
            className="btn btn-primary text-xs inline-flex items-center gap-2"
          >
            <Play className="w-3.5 h-3.5 fill-current" /> Run Baseline vs Actuated vs Adaptive Evaluation
          </button>
        </div>
      )}
    </div>
  )
}

