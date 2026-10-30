import { useState, useEffect } from 'react'
import { api } from '../services/api'
import type { ComparisonLabReport, LabHistoryItem } from '../types'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts'

export default function StrategyComparisonLab() {
  const [durationSec, setDurationSec] = useState<number>(900)
  const [scenario, setScenario] = useState<string>('normal')
  const [demandMultiplier, setDemandMultiplier] = useState<number>(1.0)
  const [strategies, setStrategies] = useState<string[]>([
    'fixed_time', 'actuated', 'adaptive', 'max_pressure', 'deep_rl'
  ])
  const [loading, setLoading] = useState<boolean>(false)
  const [report, setReport] = useState<ComparisonLabReport | null>(null)
  const [history, setHistory] = useState<LabHistoryItem[]>([])
  const [selectedHistoryId, setSelectedHistoryId] = useState<string>('')
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    loadHistory()
  }, [])

  const loadHistory = async () => {
    try {
      const res = await api.labHistory(10)
      if (res && res.history) {
        setHistory(res.history)
        if (res.history.length > 0 && !report) {
          loadReport(res.history[0].session_id)
        }
      }
    } catch (err) {
      console.warn('Failed to fetch comparison history:', err)
    }
  }

  const loadReport = async (sessionId: string) => {
    try {
      const res = await api.labReportDetail(sessionId)
      if (res && res.report) {
        setReport(res.report)
        setSelectedHistoryId(sessionId)
      }
    } catch (err) {
      console.warn('Failed to load report:', err)
    }
  }

  const handleRunComparison = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await api.labRun({
        junction_id: 'AlankarChowk',
        date: '11',
        duration_sec: durationSec,
        scenario,
        demand_multiplier: demandMultiplier,
        selected_strategies: strategies,
        seed: 42,
      })
      if (res && res.report) {
        setReport(res.report)
        setSelectedHistoryId(res.session_id)
        loadHistory()
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Benchmark run failed')
    } finally {
      setLoading(false)
    }
  }

  const toggleStrategy = (key: string) => {
    if (key === 'fixed_time') return // Baseline is required
    setStrategies(prev =>
      prev.includes(key) ? prev.filter(k => k !== key) : [...prev, key]
    )
  }

  // Prepare chart data
  const chartData = report ? Object.values(report.results).map(r => ({
    name: r.name,
    avgWait: r.average_wait_sec,
    timeSaved: r.avg_waiting_time_saved_per_vehicle_sec,
    queue: r.mean_queue_pcu,
    throughput: Math.round(r.total_throughput_pcu),
    pctSaved: r.waiting_time_saved_percent,
  })) : []

  const topStrat = report?.rankings?.[0]
  const topMetrics = topStrat && report ? report.results[topStrat.strategy_key] : null

  return (
    <div className="space-y-6">
      {/* Header & Controls Panel */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl">
        <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-4 border-b border-slate-800 pb-5">
          <div>
            <h2 className="text-xl font-bold text-white flex items-center gap-2">
              <span className="w-3 h-3 rounded-full bg-emerald-500 animate-pulse"></span>
              Strategy Comparison Lab
            </h2>
            <p className="text-sm text-slate-400 mt-1">
              Scientific benchmark of all 5 traffic control strategies under identical vehicle arrivals and conditions.
            </p>
          </div>
          {history.length > 0 && (
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-400 font-medium">Past Sessions:</span>
              <select
                value={selectedHistoryId}
                onChange={e => loadReport(e.target.value)}
                className="bg-slate-800 border border-slate-700 text-xs rounded-lg px-3 py-1.5 text-slate-200 focus:outline-none focus:border-emerald-500"
              >
                {history.map(h => (
                  <option key={h.session_id} value={h.session_id}>
                    {new Date(h.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })} — {h.best_strategy} ({h.duration_sec / 60}m)
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>

        {/* Configuration Controls */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mt-5">
          {/* Duration Selector */}
          <div>
            <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 block mb-2">
              Test Duration
            </label>
            <div className="grid grid-cols-3 gap-1.5">
              {[
                { label: '5m', sec: 300 },
                { label: '10m', sec: 600 },
                { label: '15m', sec: 900 },
                { label: '30m', sec: 1800 },
                { label: '45m', sec: 2700 },
                { label: '60m', sec: 3600 },
              ].map(d => (
                <button
                  key={d.sec}
                  type="button"
                  onClick={() => setDurationSec(d.sec)}
                  className={`py-1.5 text-xs font-medium rounded-lg border transition-all ${
                    durationSec === d.sec
                      ? 'bg-emerald-600/30 border-emerald-500 text-emerald-300'
                      : 'bg-slate-800/60 border-slate-700 text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {d.label}
                </button>
              ))}
            </div>
          </div>

          {/* Traffic Condition */}
          <div>
            <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 block mb-2">
              Traffic Demand
            </label>
            <select
              value={scenario}
              onChange={e => {
                setScenario(e.target.value)
                setDemandMultiplier(e.target.value === 'heavy' ? 1.5 : e.target.value === 'light' ? 0.7 : 1.0)
              }}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-emerald-500"
            >
              <option value="normal">Normal CCTV Traffic (1.0x)</option>
              <option value="heavy">Rush Hour Congestion (1.5x)</option>
              <option value="light">Off-Peak Flow (0.7x)</option>
            </select>
          </div>

          {/* Strategies Selection */}
          <div className="md:col-span-2">
            <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 block mb-2">
              Strategies to Benchmark
            </label>
            <div className="flex flex-wrap gap-2">
              {[
                { key: 'fixed_time', label: 'Fixed-Time (Reference)' },
                { key: 'actuated', label: 'Actuated' },
                { key: 'adaptive', label: 'Weighted Adaptive' },
                { key: 'max_pressure', label: 'Max-Pressure' },
                { key: 'deep_rl', label: 'Deep RL (PPO)' },
              ].map(s => {
                const checked = strategies.includes(s.key)
                const isFixed = s.key === 'fixed_time'
                return (
                  <button
                    key={s.key}
                    type="button"
                    onClick={() => toggleStrategy(s.key)}
                    disabled={isFixed}
                    className={`px-3 py-1.5 rounded-lg text-xs font-medium border flex items-center gap-1.5 transition-all ${
                      checked
                        ? 'bg-blue-600/25 border-blue-500 text-blue-300'
                        : 'bg-slate-800/40 border-slate-700/60 text-slate-500 hover:text-slate-400'
                    } ${isFixed ? 'opacity-80 cursor-not-allowed' : 'cursor-pointer'}`}
                  >
                    <span>{checked ? '✓' : '+'}</span>
                    {s.label}
                  </button>
                )
              })}
            </div>
          </div>
        </div>

        {/* Action Button */}
        <div className="mt-5 pt-4 border-t border-slate-800 flex justify-between items-center">
          <div className="text-xs text-slate-400">
            All algorithms will be evaluated against identical vehicle timestamps and arrival counts.
          </div>
          <button
            type="button"
            disabled={loading}
            onClick={handleRunComparison}
            className={`px-6 py-2.5 rounded-lg text-sm font-semibold text-white shadow-lg flex items-center gap-2 transition-all ${
              loading
                ? 'bg-slate-700 cursor-wait text-slate-400'
                : 'bg-emerald-600 hover:bg-emerald-500 active:scale-95 shadow-emerald-900/40'
            }`}
          >
            {loading ? (
              <>
                <div className="w-4 h-4 border-2 border-white/20 border-t-white rounded-full animate-spin"></div>
                Evaluating Strategies...
              </>
            ) : (
              <>
                <span>▶</span> Run Strategy Comparison
              </>
            )}
          </button>
        </div>

        {error && (
          <div className="mt-4 p-3 bg-rose-900/30 border border-rose-800 rounded-lg text-rose-300 text-xs">
            {error}
          </div>
        )}
      </div>

      {/* Results View */}
      {report && (
        <>
          {/* Executive Summary Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg relative overflow-hidden">
              <div className="text-xs uppercase font-semibold text-slate-400">Overall #1 Strategy</div>
              <div className="text-xl font-bold text-emerald-400 mt-1 truncate">
                {topStrat?.strategy_name}
              </div>
              <div className="text-xs text-slate-400 mt-2">
                Score: <span className="font-semibold text-white">{topStrat?.composite_score}</span>
              </div>
              <div className="absolute top-3 right-3 text-2xl">🏆</div>
            </div>

            <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg">
              <div className="text-xs uppercase font-semibold text-slate-400">Time Saved per Vehicle</div>
              <div className="text-2xl font-bold text-white mt-1">
                {topMetrics?.avg_waiting_time_saved_per_vehicle_sec}s
              </div>
              <div className="text-xs text-emerald-400 mt-2 font-medium">
                {topMetrics?.waiting_time_saved_percent}% less waiting
              </div>
            </div>

            <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg">
              <div className="text-xs uppercase font-semibold text-slate-400">Total Commute Time Saved</div>
              <div className="text-2xl font-bold text-white mt-1">
                {Math.round(topMetrics?.total_waiting_time_saved_sec || 0).toLocaleString()}s
              </div>
              <div className="text-xs text-slate-400 mt-2">
                Over {topMetrics?.total_arrived_vehicles.toLocaleString()} vehicles
              </div>
            </div>

            <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg">
              <div className="text-xs uppercase font-semibold text-slate-400">Congestion Reduction</div>
              <div className="text-2xl font-bold text-cyan-400 mt-1">
                {topMetrics?.queue_reduction_percent}%
              </div>
              <div className="text-xs text-slate-400 mt-2">
                Average queue: {topMetrics?.mean_queue_pcu} units
              </div>
            </div>
          </div>

          {/* Plain English Executive Summary Banner */}
          <div className="bg-gradient-to-r from-emerald-950/40 via-slate-900 to-slate-900 border border-emerald-800/40 rounded-xl p-5 shadow-md">
            <div className="flex items-start gap-3">
              <span className="text-xl">💡</span>
              <div>
                <h4 className="text-sm font-bold text-white">Summary of Findings</h4>
                <p className="text-sm text-slate-300 mt-1 leading-relaxed">
                  {report.plain_english_summary}
                </p>
              </div>
            </div>
          </div>

          {/* Detailed Side-by-Side Table */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-xl">
            <div className="px-6 py-4 border-b border-slate-800 flex justify-between items-center">
              <h3 className="font-bold text-white text-base">Strategy Performance Breakdown</h3>
              <div className="flex items-center gap-2">
                <a
                  href={api.labExportUrl(report.session_id, 'csv')}
                  download
                  className="px-3 py-1.5 text-xs font-medium rounded-lg bg-slate-800 border border-slate-700 text-slate-300 hover:text-white transition-colors"
                >
                  Download CSV
                </a>
                <a
                  href={api.labExportUrl(report.session_id, 'json')}
                  download
                  className="px-3 py-1.5 text-xs font-medium rounded-lg bg-slate-800 border border-slate-700 text-slate-300 hover:text-white transition-colors"
                >
                  Download JSON
                </a>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead className="bg-slate-800/60 text-xs text-slate-400 uppercase tracking-wider">
                  <tr>
                    <th className="py-3.5 px-4 font-semibold">Rank</th>
                    <th className="py-3.5 px-4 font-semibold">Strategy</th>
                    <th className="py-3.5 px-4 font-semibold">Avg Waiting Time</th>
                    <th className="py-3.5 px-4 font-semibold">Time Saved / Veh</th>
                    <th className="py-3.5 px-4 font-semibold">Total Time Saved</th>
                    <th className="py-3.5 px-4 font-semibold">Vehicles Waiting</th>
                    <th className="py-3.5 px-4 font-semibold">Vehicles Passed</th>
                    <th className="py-3.5 px-4 font-semibold">Starvations</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800">
                  {report.rankings.map(r => {
                    const m = report.results[r.strategy_key]
                    const isWinner = r.rank === 1
                    return (
                      <tr
                        key={r.strategy_key}
                        className={`transition-colors ${
                          isWinner ? 'bg-emerald-950/20 font-medium' : 'hover:bg-slate-800/40'
                        }`}
                      >
                        <td className="py-3 px-4">
                          <span
                            className={`inline-flex items-center justify-center w-6 h-6 rounded-full text-xs font-bold ${
                              r.rank === 1
                                ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
                                : r.rank === 2
                                ? 'bg-slate-700 text-slate-200'
                                : 'bg-slate-800 text-slate-400'
                            }`}
                          >
                            {r.rank}
                          </span>
                        </td>
                        <td className="py-3 px-4">
                          <div className="font-semibold text-white">{m.name}</div>
                          <div className="text-xs text-slate-400">{m.category}</div>
                        </td>
                        <td className="py-3 px-4 text-white">
                          {m.average_wait_sec}s
                        </td>
                        <td className="py-3 px-4">
                          <span
                            className={`font-semibold ${
                              m.avg_waiting_time_saved_per_vehicle_sec > 0
                                ? 'text-emerald-400'
                                : m.avg_waiting_time_saved_per_vehicle_sec === 0
                                ? 'text-slate-400'
                                : 'text-rose-400'
                            }`}
                          >
                            {m.avg_waiting_time_saved_per_vehicle_sec > 0 ? '+' : ''}
                            {m.avg_waiting_time_saved_per_vehicle_sec}s
                          </span>
                        </td>
                        <td className="py-3 px-4 text-slate-200">
                          {Math.round(m.total_waiting_time_saved_sec).toLocaleString()}s
                        </td>
                        <td className="py-3 px-4 text-slate-200">
                          {m.mean_queue_pcu} units
                        </td>
                        <td className="py-3 px-4 text-slate-200">
                          {Math.round(m.total_throughput_pcu).toLocaleString()} units
                        </td>
                        <td className="py-3 px-4">
                          <span
                            className={`px-2 py-0.5 rounded text-xs ${
                              m.starvation_occurrences === 0
                                ? 'bg-emerald-900/30 text-emerald-400'
                                : 'bg-rose-900/30 text-rose-400 font-semibold'
                            }`}
                          >
                            {m.starvation_occurrences}
                          </span>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>

            {/* Why This Strategy Ranked #1 Callout */}
            <div className="p-4 bg-slate-800/40 border-t border-slate-800">
              <div className="text-xs text-slate-400 font-medium uppercase tracking-wider mb-2">
                Ranking Justification:
              </div>
              <div className="space-y-1.5">
                {report.rankings.map(r => (
                  <div key={r.strategy_key} className="text-xs text-slate-300 flex items-start gap-2">
                    <span className="font-bold text-slate-400">#{r.rank} {r.strategy_name}:</span>
                    <span>{r.rationale}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Interactive Comparison Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Average Wait Time Chart */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg">
              <h3 className="text-sm font-bold text-white mb-4">
                Average Vehicle Delay by Strategy (Lower is Better)
              </h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData} margin={{ top: 10, right: 20, left: -10, bottom: 25 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis
                      dataKey="name"
                      stroke="#94a3b8"
                      fontSize={11}
                      interval={0}
                      angle={-20}
                      textAnchor="end"
                    />
                    <YAxis stroke="#94a3b8" fontSize={11} unit="s" />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }}
                      formatter={(val: any) => [`${val} seconds`, 'Avg Wait']}
                    />
                    <Bar dataKey="avgWait" fill="#38bdf8" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Time Saved per Vehicle Chart */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg">
              <h3 className="text-sm font-bold text-white mb-4">
                Time Saved vs Fixed-Time Baseline (Higher is Better)
              </h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData} margin={{ top: 10, right: 20, left: -10, bottom: 25 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis
                      dataKey="name"
                      stroke="#94a3b8"
                      fontSize={11}
                      interval={0}
                      angle={-20}
                      textAnchor="end"
                    />
                    <YAxis stroke="#94a3b8" fontSize={11} unit="s" />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }}
                      formatter={(val: any) => [`${val}s saved`, 'Time Saved / Veh']}
                    />
                    <Bar dataKey="timeSaved" fill="#10b981" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  )
}

