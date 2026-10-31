import { useState, useEffect } from 'react'
import { api } from '../services/api'
import type { ComparisonLabReport, LabHistoryItem } from '../types'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts'
import { Trophy, Clock, TrendingUp, ShieldAlert, Award, Download, RefreshCw, CheckCircle2 } from 'lucide-react'

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
      const res = await api.labHistory(15)
      if (res && res.history && res.history.length > 0) {
        setHistory(res.history)
        if (!report) {
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
      setError(err instanceof Error ? err.message : 'Benchmark run failed. Ensure backend server is reachable.')
    } finally {
      setLoading(false)
    }
  }

  const toggleStrategy = (key: string) => {
    if (key === 'fixed_time') return // Baseline is required as reference
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
    throughput: r.vehicles_processed || Math.round(r.total_throughput_pcu),
    pctSaved: r.waiting_time_saved_percent,
    score: report.rankings.find(rk => rk.strategy_key === r.key)?.composite_score || 50,
  })) : []

  const topStrat = report?.rankings?.[0]
  const topMetrics = topStrat && report ? report.results[topStrat.strategy_key] : null
  const fixedMetrics = report?.results?.['fixed_time']

  return (
    <div className="space-y-6">
      {/* ── Top Strategy Comparison Setup ── */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
        <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-4 border-b border-slate-800 pb-5">
          <div>
            <div className="flex items-center gap-2">
              <span className="p-1.5 rounded-lg bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                <Award size={18} />
              </span>
              <h2 className="text-xl font-bold text-white">Strategy Comparison Lab</h2>
              <span className="px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-emerald-950/80 border border-emerald-600/40 text-emerald-300">
                Central Benchmark
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-1 max-w-2xl leading-relaxed">
              Compares all five traffic signal controllers under <strong>strictly identical vehicle arrival timestamps</strong> and Pune CCTV road demand. Evaluates delay reduction, queues, throughput, and time saved.
            </p>
          </div>

          {/* Past session picker */}
          {history.length > 0 && (
            <div className="flex items-center gap-2 bg-slate-950/70 border border-slate-800 px-3 py-1.5 rounded-xl">
              <span className="text-xs text-slate-400 whitespace-nowrap">Session:</span>
              <select
                value={selectedHistoryId}
                onChange={e => loadReport(e.target.value)}
                className="bg-transparent text-xs text-slate-200 focus:outline-none cursor-pointer"
              >
                {history.map(h => (
                  <option key={h.session_id} value={h.session_id} className="bg-slate-900">
                    {new Date(h.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })} — {h.duration_sec / 60}min · {h.best_strategy}
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>

        {/* Configuration grid */}
        <div className="grid grid-cols-1 md:grid-cols-12 gap-5 mt-5">
          {/* Duration Selector */}
          <div className="md:col-span-4">
            <div className="flex items-center justify-between mb-2">
              <label className="text-xs font-semibold uppercase tracking-wider text-slate-300">
                Simulation Duration
              </label>
              <span className="text-[11px] text-slate-500 font-mono">{durationSec / 60} minutes</span>
            </div>
            <div className="grid grid-cols-6 gap-1.5">
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
                  className={`py-2 text-xs font-bold rounded-lg border transition-all ${
                    durationSec === d.sec
                      ? 'bg-emerald-600 text-white border-emerald-500 shadow-md shadow-emerald-950/40'
                      : 'bg-slate-800/60 border-slate-700/80 text-slate-400 hover:text-slate-200 hover:bg-slate-800'
                  }`}
                >
                  {d.label}
                </button>
              ))}
            </div>
          </div>

          {/* Traffic condition selector */}
          <div className="md:col-span-3">
            <label className="text-xs font-semibold uppercase tracking-wider text-slate-300 block mb-2">
              Traffic Demand
            </label>
            <select
              value={scenario}
              onChange={e => {
                setScenario(e.target.value)
                setDemandMultiplier(e.target.value === 'heavy' ? 1.5 : e.target.value === 'light' ? 0.7 : 1.0)
              }}
              className="w-full bg-slate-800/80 border border-slate-700 rounded-lg px-3 py-2 text-xs font-medium text-slate-200 focus:outline-none focus:border-emerald-500"
            >
              <option value="normal">Normal CCTV Traffic (1.0x)</option>
              <option value="heavy">Rush Hour Congestion (1.5x)</option>
              <option value="light">Off-Peak Free Flow (0.7x)</option>
            </select>
          </div>

          {/* Strategies Checkboxes */}
          <div className="md:col-span-5">
            <label className="text-xs font-semibold uppercase tracking-wider text-slate-300 block mb-2">
              Selected Strategies
            </label>
            <div className="flex flex-wrap gap-1.5">
              {[
                { key: 'fixed_time', label: 'Fixed-Time' },
                { key: 'actuated', label: 'Actuated' },
                { key: 'adaptive', label: 'Adaptive' },
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
                    className={`px-2.5 py-1.5 rounded-lg text-xs font-medium border flex items-center gap-1.5 transition-all ${
                      checked
                        ? 'bg-cyan-950/60 border-cyan-500/70 text-cyan-300'
                        : 'bg-slate-800/30 border-slate-700/50 text-slate-500 hover:text-slate-400'
                    } ${isFixed ? 'opacity-80 cursor-default' : 'cursor-pointer'}`}
                  >
                    <span>{checked ? '✓' : '+'}</span>
                    <span>{s.label}</span>
                  </button>
                )
              })}
            </div>
          </div>
        </div>

        {/* Action Button Bar */}
        <div className="mt-5 pt-4 border-t border-slate-800/80 flex flex-wrap items-center justify-between gap-3">
          <div className="text-xs text-slate-400 flex items-center gap-1.5">
            <CheckCircle2 size={14} className="text-emerald-400" />
            <span>Fair benchmarking guarantee: Every strategy runs on the exact same arrival sequence and seed.</span>
          </div>

          <button
            type="button"
            disabled={loading}
            onClick={handleRunComparison}
            className={`px-6 py-2.5 rounded-xl text-xs font-bold text-white shadow-lg flex items-center gap-2 transition-all ${
              loading
                ? 'bg-slate-700 cursor-wait text-slate-400'
                : 'bg-emerald-600 hover:bg-emerald-500 active:scale-95 shadow-emerald-950/40 cursor-pointer'
            }`}
          >
            {loading ? (
              <>
                <RefreshCw size={14} className="animate-spin" />
                <span>Simulating All 5 Strategies...</span>
              </>
            ) : (
              <>
                <span>▶</span>
                <span>Run Strategy Comparison</span>
              </>
            )}
          </button>
        </div>

        {error && (
          <div className="mt-4 p-3 bg-rose-950/40 border border-rose-800/60 rounded-xl text-rose-300 text-xs flex items-center gap-2">
            <ShieldAlert size={14} />
            <span>{error}</span>
          </div>
        )}
      </div>

      {/* ── Results View ── */}
      {report ? (
        <>
          {/* Executive Impact Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* Winner Card */}
            <div className="bg-gradient-to-br from-amber-950/30 via-slate-900 to-slate-900 border border-amber-600/40 rounded-2xl p-5 shadow-lg relative overflow-hidden">
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-bold uppercase tracking-wider text-amber-400">
                  Overall Winner
                </span>
                <Trophy size={18} className="text-amber-400" />
              </div>
              <div className="text-xl font-bold text-white mt-1.5 truncate">
                {topStrat?.strategy_name}
              </div>
              <div className="text-xs text-slate-400 mt-2 flex items-center gap-2">
                <span>Score:</span>
                <span className="font-bold text-amber-300 font-mono text-sm">
                  {topStrat?.composite_score}/100
                </span>
              </div>
              <div className="text-[11px] text-emerald-400 mt-1 font-medium">
                {topMetrics?.waiting_time_saved_percent}% delay reduction
              </div>
            </div>

            {/* Time Saved per Vehicle */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg">
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
                  Time Saved / Vehicle
                </span>
                <Clock size={16} className="text-emerald-400" />
              </div>
              <div className="text-2xl font-bold text-white mt-1.5 font-mono">
                {topMetrics?.avg_waiting_time_saved_per_vehicle_sec && topMetrics.avg_waiting_time_saved_per_vehicle_sec > 0 ? '+' : ''}
                {topMetrics?.avg_waiting_time_saved_per_vehicle_sec}s
              </div>
              <div className="text-xs text-slate-400 mt-2">
                Baseline wait: <span className="font-mono text-slate-300">{fixedMetrics?.average_wait_sec}s</span>
              </div>
              <div className="text-[11px] text-emerald-400 mt-1 font-medium">
                Winner avg wait: <span className="font-mono text-white font-bold">{topMetrics?.average_wait_sec}s</span>
              </div>
            </div>

            {/* Cumulative Simulated Time Saved */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg">
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
                  Simulated Time Saved
                </span>
                <TrendingUp size={16} className="text-cyan-400" />
              </div>
              <div className="text-2xl font-bold text-cyan-400 mt-1.5 font-mono">
                {Math.round(topMetrics?.total_waiting_time_saved_sec || 0).toLocaleString()}s
              </div>
              <div className="text-xs text-slate-400 mt-2">
                Across <span className="text-white font-bold">{topMetrics?.total_arrived_vehicles.toLocaleString()}</span> vehicles arrived
              </div>
              <div className="text-[11px] text-slate-500 mt-1">
                Cleared through green: <span className="font-mono text-slate-400">{(topMetrics?.vehicles_processed ?? 0).toLocaleString()}</span>
              </div>
            </div>

            {/* Projected City Impact */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg">
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
                  Projected Daily Savings
                </span>
                <span className="px-1.5 py-0.5 rounded text-[9px] uppercase font-bold bg-violet-950/80 text-violet-300 border border-violet-700/50">
                  Estimated
                </span>
              </div>
              <div className="text-2xl font-bold text-violet-300 mt-1.5 font-mono">
                {(() => {
                  const hrs = topMetrics?.projected_daily_time_saved_hours || 0
                  if (hrs >= 1.0) return `~${hrs.toFixed(1)} hrs`
                  if (hrs > 0) return `~${(hrs * 60).toFixed(1)} min`
                  return '—'
                })()}
              </div>
              <div className="text-xs text-slate-400 mt-2">
                Extrapolated from {report.duration_sec / 60}min test → 12h day
              </div>
              <div className="text-[11px] text-slate-500 mt-1">
                Simulation estimate, not measured
              </div>
            </div>
          </div>

          {/* Plain-English Summary Banner */}
          <div className="bg-gradient-to-r from-emerald-950/40 via-slate-900 to-slate-900 border border-emerald-800/40 rounded-2xl p-5 shadow-md">
            <div className="flex items-start gap-3">
              <span className="text-2xl shrink-0 mt-0.5">💡</span>
              <div className="space-y-1">
                <h4 className="text-sm font-bold text-white flex items-center gap-2">
                  <span>Executive Performance Summary</span>
                  <span className="text-xs font-normal text-slate-400">({report.junction_id}, {report.duration_sec / 60} minutes)</span>
                </h4>
                <p className="text-xs text-slate-300 leading-relaxed">
                  {report.plain_english_summary}
                </p>
              </div>
            </div>
          </div>

          {/* Detailed Side-by-Side Table */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-xl">
            <div className="px-6 py-4 border-b border-slate-800 flex flex-wrap items-center justify-between gap-3">
              <div>
                <h3 className="font-bold text-white text-sm">Full Strategy Scorecard</h3>
                <p className="text-xs text-slate-400 mt-0.5">Side-by-side comparison under identical conditions</p>
              </div>
              <div className="flex items-center gap-2">
                <a
                  href={api.labExportUrl(report.session_id, 'csv')}
                  download
                  className="px-3 py-1.5 text-xs font-semibold rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white border border-slate-700 flex items-center gap-1.5 transition-colors"
                >
                  <Download size={13} />
                  <span>Download CSV</span>
                </a>
                <a
                  href={api.labExportUrl(report.session_id, 'json')}
                  download
                  className="px-3 py-1.5 text-xs font-semibold rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white border border-slate-700 flex items-center gap-1.5 transition-colors"
                >
                  <Download size={13} />
                  <span>Download JSON</span>
                </a>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-800/60 text-[11px] text-slate-400 uppercase tracking-wider border-b border-slate-800">
                  <tr>
                    <th className="py-3 px-4 font-semibold">Rank</th>
                    <th className="py-3 px-4 font-semibold">Strategy</th>
                    <th className="py-3 px-4 font-semibold">Efficiency Score</th>
                    <th className="py-3 px-4 font-semibold">Avg Waiting Time</th>
                    <th className="py-3 px-4 font-semibold">Time Saved / Veh</th>
                    <th className="py-3 px-4 font-semibold">Simulated Time Saved</th>
                    <th className="py-3 px-4 font-semibold">Queue Reduction</th>
                    <th className="py-3 px-4 font-semibold">Vehicles Processed</th>
                    <th className="py-3 px-4 font-semibold">Starvations</th>
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
                        <td className="py-3.5 px-4">
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
                        <td className="py-3.5 px-4">
                          <div className="font-bold text-white text-sm flex items-center gap-2">
                            {m.name}
                            {r.strategy_key === 'deep_rl' && m.avg_waiting_time_saved_per_vehicle_sec < -20 && (
                              <span className="px-1.5 py-0.5 rounded text-[9px] font-bold bg-amber-950/80 border border-amber-600/50 text-amber-300 whitespace-nowrap">
                                ⚠ Undertrained
                              </span>
                            )}
                          </div>
                          <div className="text-[11px] text-slate-400">{m.category}</div>
                        </td>
                        <td className="py-3.5 px-4">
                          <div className="flex items-center gap-2">
                            <span className="font-mono font-bold text-white text-xs">{r.composite_score}</span>
                            <div className="w-16 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                              <div
                                className={`h-full rounded-full ${
                                  r.composite_score >= 70
                                    ? 'bg-emerald-400'
                                    : r.composite_score >= 50
                                    ? 'bg-cyan-400'
                                    : 'bg-amber-400'
                                }`}
                                style={{ width: `${Math.min(100, Math.max(0, r.composite_score))}%` }}
                              />
                            </div>
                          </div>
                        </td>
                        <td className="py-3.5 px-4 font-mono text-white text-xs">
                          {m.average_wait_sec}s
                        </td>
                        <td className="py-3.5 px-4 font-mono">
                          <span
                            className={`font-bold ${
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
                        <td className="py-3.5 px-4 font-mono text-slate-300">
                          {Math.round(m.total_waiting_time_saved_sec).toLocaleString()}s
                        </td>
                        <td className="py-3.5 px-4 font-mono">
                          <span
                            className={
                              m.queue_reduction_percent > 0
                                ? 'text-emerald-400 font-semibold'
                                : m.queue_reduction_percent === 0
                                ? 'text-slate-400'
                                : 'text-rose-400'
                            }
                          >
                            {m.queue_reduction_percent > 0 ? '+' : ''}
                            {m.queue_reduction_percent}%
                          </span>
                        </td>
                        <td className="py-3.5 px-4 font-mono text-slate-300">
                          {(m.vehicles_processed || Math.round(m.total_throughput_pcu)).toLocaleString()}
                        </td>
                        <td className="py-3.5 px-4">
                          <span
                            className={`px-2 py-0.5 rounded text-[11px] ${
                              m.starvation_occurrences === 0
                                ? 'bg-emerald-950/60 text-emerald-400 border border-emerald-800/40'
                                : 'bg-rose-950/60 text-rose-400 font-semibold border border-rose-800/40'
                            }`}
                          >
                            {m.starvation_occurrences === 0 ? '0 (None)' : m.starvation_occurrences}
                          </span>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>

            {/* Plain-English Why Each Strategy Performed as it Did */}
            <div className="p-5 bg-slate-950/40 border-t border-slate-800">
              <div className="text-xs text-slate-400 font-bold uppercase tracking-wider mb-3">
                Why Did Each Strategy Perform This Way?
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {report.rankings.map(r => {
                  const m = report.results[r.strategy_key]
                  return (
                    <div
                      key={r.strategy_key}
                      className="p-3 bg-slate-900/60 border border-slate-800 rounded-xl space-y-1"
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-bold text-white text-xs">
                          #{r.rank} {r.strategy_name}
                        </span>
                        <span className="text-[10px] text-slate-500 font-mono">Score {r.composite_score}</span>
                      </div>
                      <p className="text-[11px] text-slate-300 leading-relaxed">
                        {m.plain_english_analysis || r.rationale}
                      </p>
                      {r.strategy_key === 'deep_rl' && m.avg_waiting_time_saved_per_vehicle_sec < -20 && (
                        <div className="mt-2 p-2 bg-amber-950/30 border border-amber-700/40 rounded-lg text-[11px] text-amber-300 leading-relaxed">
                          ⚠ <strong>PPO model appears undertrained</strong> — results reflect insufficient training steps. 
                          Retrain via the <em>AI &amp; Diagnostics → AI Controller (PPO)</em> tab for better performance.
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
            </div>
          </div>

          {/* Interactive Visual Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Chart 1: Average Waiting Time */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-4">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <div>
                  <h4 className="text-xs font-bold text-white uppercase tracking-wider">
                    Average Vehicle Delay
                  </h4>
                  <p className="text-[11px] text-slate-400">Lower bar = Less delay = Better performance</p>
                </div>
                <span className="text-xs font-mono text-cyan-400">Seconds (s)</span>
              </div>
              <div className="h-60">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData} margin={{ top: 10, right: 15, left: -15, bottom: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                    <XAxis
                      dataKey="name"
                      stroke="#64748b"
                      fontSize={10}
                      interval={0}
                      angle={-15}
                      textAnchor="end"
                    />
                    <YAxis stroke="#64748b" fontSize={10} unit="s" />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '10px', fontSize: 11 }}
                      formatter={(val: any) => [`${val}s average wait`, 'Delay']}
                    />
                    <Bar dataKey="avgWait" fill="#38bdf8" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Chart 2: Cumulative Time Saved */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-4">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <div>
                  <h4 className="text-xs font-bold text-white uppercase tracking-wider">
                    Time Saved vs Fixed-Time Baseline
                  </h4>
                  <p className="text-[11px] text-slate-400">Higher bar = More commuter time saved per vehicle</p>
                </div>
                <span className="text-xs font-mono text-emerald-400">Seconds (s)</span>
              </div>
              <div className="h-60">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData} margin={{ top: 10, right: 15, left: -15, bottom: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                    <XAxis
                      dataKey="name"
                      stroke="#64748b"
                      fontSize={10}
                      interval={0}
                      angle={-15}
                      textAnchor="end"
                    />
                    <YAxis stroke="#64748b" fontSize={10} unit="s" />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '10px', fontSize: 11 }}
                      formatter={(val: any) => [`${val}s saved per vehicle`, 'Time Saved']}
                    />
                    <Bar dataKey="timeSaved" fill="#10b981" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </>
      ) : (
        /* Empty / Ready State */
        <div className="bg-slate-900/60 border border-dashed border-slate-800 rounded-2xl p-12 text-center space-y-4">
          <div className="w-16 h-16 rounded-full bg-emerald-950/80 border border-emerald-600/30 text-emerald-400 flex items-center justify-center mx-auto text-2xl">
            🏆
          </div>
          <div className="max-w-md mx-auto space-y-2">
            <h3 className="text-base font-bold text-white">Ready to Compare All 5 Strategies</h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              Click <strong>Run Strategy Comparison</strong> above to evaluate Fixed-Time, Actuated, Weighted Adaptive, Max-Pressure, and Deep RL on identical Pune CCTV traffic demand.
            </p>
          </div>
          <button
            type="button"
            onClick={handleRunComparison}
            disabled={loading}
            className="px-6 py-2.5 rounded-xl text-xs font-bold bg-emerald-600 hover:bg-emerald-500 text-white shadow-lg cursor-pointer"
          >
            Start 15-Minute Benchmark
          </button>
        </div>
      )}
    </div>
  )
}
