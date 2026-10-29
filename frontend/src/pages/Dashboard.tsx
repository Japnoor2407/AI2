import React, { useEffect, useState, useCallback, useRef } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  BarChart, Bar, ResponsiveContainer,
} from 'recharts'
import {
  Play, Pause, RotateCcw, Activity, AlertTriangle,
  Download, Database, Zap, Clock, TrendingUp, Video,
  BarChart3, GitFork, FileText, CheckCircle2, ShieldCheck, Brain,
} from 'lucide-react'

import { api } from '../services/api'
import type {
  TrafficCurrent, ControllerStatus, EvaluationSummary,
  HistoryPoint, DecisionLogEntry, AggregatedInterval,
} from '../types'
import { JunctionViz } from '../components/JunctionViz'
import { SignalBadge } from '../components/SignalBadge'
import { TrafficVideoFeed } from '../components/TrafficVideoFeed'
import { AggregatedTrafficCharts } from '../components/AggregatedTrafficCharts'
import { BenchmarkPanel } from '../components/BenchmarkPanel'
import { SafetyAuditPanel } from '../components/SafetyAuditPanel'
import { MLForecastPanel } from '../components/MLForecastPanel'

// ─── Metric Card ────────────────────────────────────────────────────────────

function MetricCard({
  label, value, unit, sub, icon, accent = 'cyan',
}: {
  label: string; value: string | number; unit?: string; sub?: string;
  icon?: React.ReactNode; accent?: 'cyan' | 'green' | 'amber' | 'red'
}) {
  const colors: Record<string, string> = {
    cyan: 'text-cyan-400', green: 'text-emerald-400',
    amber: 'text-amber-400', red: 'text-rose-400',
  }
  return (
    <div className="card flex flex-col justify-between">
      <div className="flex items-center justify-between text-slate-400 text-xs uppercase tracking-wider font-semibold">
        <span>{label}</span>
        {icon}
      </div>
      <div className="my-2">
        <span className={`text-2xl font-bold font-mono tracking-tight ${colors[accent]}`}>
          {value}
        </span>
        {unit && <span className="text-sm font-normal text-slate-400 ml-1">{unit}</span>}
      </div>
      {sub && <div className="text-[11px] text-slate-500">{sub}</div>}
    </div>
  )
}

// ─── Main Dashboard ──────────────────────────────────────────────────────────

export default function Dashboard() {
  const [running, setRunning] = useState(false)
  const [paused, setPaused] = useState(false)
  const [speed, setSpeed] = useState(1000)
  const [mode, setMode] = useState<'ADAPTIVE' | 'ACTUATED' | 'FIXED_TIME'>('ADAPTIVE')
  const [status, setStatus] = useState<ControllerStatus | null>(null)
  const [current, setCurrent] = useState<TrafficCurrent | null>(null)
  const [history, setHistory] = useState<HistoryPoint[]>([])
  const [evalSummary, setEvalSummary] = useState<EvaluationSummary | null>(null)
  const [decisionLog, setDecisionLog] = useState<DecisionLogEntry[]>([])
  const [evalRunning, setEvalRunning] = useState(false)
  const [activeTab, setActiveTab] = useState<'overview' | 'video' | 'junction' | 'monitoring' | 'benchmark' | 'safety' | 'ml' | 'evaluation' | 'log' | 'dataset'>('overview')
  const [monitoringUnit, setMonitoringUnit] = useState<'10min' | '1sec'>('10min')
  const [overviewGraphUnit, setOverviewGraphUnit] = useState<'10min' | '1sec'>('10min')
  const [aggregated10Min, setAggregated10Min] = useState<AggregatedInterval[]>([])
  const [apiError, setApiError] = useState<string | null>(null)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const fetchAll = useCallback(async () => {
    try {
      const [st, cur, hist, dl, agg] = await Promise.all([
        api.controllerStatus(),
        api.trafficCurrent(),
        api.trafficHistory(300),
        api.decisionLog(50),
        api.trafficAggregated(600).catch(() => null),
      ])
      setStatus(st)
      if (cur?.step !== undefined) setCurrent(cur)
      setHistory((hist?.history || []).slice(-150))
      setDecisionLog(dl?.log || [])
      if (agg?.intervals) setAggregated10Min(agg.intervals)
      setApiError(null)
    } catch (e: unknown) {
      setApiError(e instanceof Error ? e.message : 'API error')
    }
  }, [])

  useEffect(() => { fetchAll() }, [fetchAll])

  useEffect(() => {
    if (running && !paused) {
      intervalRef.current = setInterval(async () => {
        try {
          const res = await api.controllerStep()
          if (res?.status === 'finished') { setRunning(false); return }
          await fetchAll()
        } catch (e) { console.error(e) }
      }, speed)
    } else {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
    return () => { if (intervalRef.current) clearInterval(intervalRef.current) }
  }, [running, paused, speed, fetchAll])

  const handleStart = async () => {
    await api.simulationStart({ junction_id: 'AlankarChowk', date: '11', duration_sec: 1800, mode })
    await api.setMode(mode)
    setRunning(true); setPaused(false)
    await fetchAll()
  }

  const handlePause = async () => {
    await api.simulationPause()
    setPaused(p => !p)
  }

  const handleReset = async () => {
    setRunning(false); setPaused(false)
    await api.simulationReset()
    setCurrent(null); setHistory([])
    await fetchAll()
  }

  const handleModeSwitch = async (m: 'ADAPTIVE' | 'ACTUATED' | 'FIXED_TIME') => {
    setMode(m)
    await api.setMode(m)
  }

  const runEvaluation = async () => {
    setEvalRunning(true)
    try {
      await api.evaluationRun({ junction_id: 'AlankarChowk', date: '11', duration_sec: 600 })
      const s = await api.evaluationSummary()
      setEvalSummary(s)
    } catch (e: unknown) {
      alert('Evaluation error: ' + (e instanceof Error ? e.message : e))
    } finally { setEvalRunning(false) }
  }

  const exportEval = async (fmt: 'json' | 'csv') => {
    const res = await api.evaluationExport(fmt)
    const blob = await res.blob()
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url; a.download = `evaluation_results.${fmt}`; a.click()
    URL.revokeObjectURL(url)
  }

  const ctrl = current?.controller

  const tabs = [
    { id: 'overview',   label: 'Overview', icon: <Activity size={15} /> },
    { id: 'video',      label: 'CCTV Video Feed', icon: <Video size={15} /> },
    { id: 'junction',   label: 'Junction View', icon: <GitFork size={15} /> },
    { id: 'monitoring', label: '10-Min Graphs', icon: <BarChart3 size={15} /> },
    { id: 'benchmark',  label: '3-Way Benchmark', icon: <TrendingUp size={15} /> },
    { id: 'safety',     label: 'Safety Audit', icon: <ShieldCheck size={15} /> },
    { id: 'ml',         label: 'ML Demand Forecaster', icon: <Brain size={15} /> },
    { id: 'evaluation', label: 'Evaluation', icon: <CheckCircle2 size={15} /> },
    { id: 'log',        label: 'Decision Log', icon: <FileText size={15} /> },
    { id: 'dataset',    label: 'Dataset', icon: <Database size={15} /> },
  ] as const

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* Top Operations Header */}
      <header className="bg-slate-900 border-b border-slate-800 px-6 py-3.5 sticky top-0 z-30 shadow-md">
        <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-cyan-600/20 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
              <Zap size={20} />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-base font-bold tracking-tight text-white">
                  Adaptive Traffic Signal Optimization Agent
                </span>
                <span className="text-[11px] font-semibold bg-cyan-950 text-cyan-300 border border-cyan-700/60 px-2 py-0.5 rounded">
                  SIH1607
                </span>
              </div>
              <div className="text-xs text-slate-400">
                Pune Heterogeneous CCTV Dataset · Alankar Chowk
              </div>
            </div>
          </div>

          {apiError && (
            <div className="flex items-center gap-2 text-rose-400 text-xs border border-rose-800/60 bg-rose-950/40 px-3 py-1.5 rounded">
              <AlertTriangle size={14} /> {apiError}
            </div>
          )}

          <div className="flex items-center gap-4">
            <div className="text-right">
              <div className="text-xs text-slate-400">Simulation Clock</div>
              <div className="text-sm font-mono font-bold text-white">
                {current?.timestamp || '09:00:34'}
              </div>
            </div>
            <div className="h-8 w-px bg-slate-800" />
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-400">
                Step <span className="font-mono text-white font-bold">{status?.step ?? 0}</span> / {status?.total_steps ?? 0}
              </span>
              {status?.signal_state && <SignalBadge state={status.signal_state} />}
            </div>
          </div>
        </div>
      </header>

      {/* Navigation Tabs Bar */}
      <nav className="bg-slate-900/90 border-b border-slate-800/80 px-6 backdrop-blur-sm sticky top-[61px] z-20">
        <div className="max-w-7xl mx-auto flex items-center gap-2 overflow-x-auto py-2">
          {tabs.map(t => (
            <button
              key={t.id}
              onClick={() => setActiveTab(t.id)}
              className={`flex items-center gap-2 px-3.5 py-1.5 text-xs font-semibold rounded-lg transition-all whitespace-nowrap ${
                activeTab === t.id
                  ? 'bg-cyan-600 text-white shadow-sm shadow-cyan-950 border border-cyan-400/40'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
              }`}
            >
              {t.icon}
              {t.label}
            </button>
          ))}
        </div>
      </nav>

      {/* Main Content Area */}
      <main className="max-w-7xl mx-auto w-full p-6 space-y-6 flex-1">

        {/* ─── OVERVIEW TAB ─── */}
        {activeTab === 'overview' && (
          <div className="space-y-6">
            {/* Control Bar */}
            <div className="card space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-4">
                {/* Controller Mode Switch */}
                <div className="flex items-center gap-1.5 bg-slate-900 p-1 rounded-lg border border-slate-800">
                  <button
                    onClick={() => handleModeSwitch('ADAPTIVE')}
                    className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-all ${
                      mode === 'ADAPTIVE'
                        ? 'bg-cyan-600 text-white shadow-sm shadow-cyan-950'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    ⚡ Adaptive (AI)
                  </button>
                  <button
                    onClick={() => handleModeSwitch('ACTUATED')}
                    className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-all ${
                      mode === 'ACTUATED'
                        ? 'bg-emerald-600 text-white shadow-sm shadow-emerald-950'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    🚦 Actuated
                  </button>
                  <button
                    onClick={() => handleModeSwitch('FIXED_TIME')}
                    className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-all ${
                      mode === 'FIXED_TIME'
                        ? 'bg-amber-600 text-white shadow-sm shadow-amber-950'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    ⏱ Fixed-Time
                  </button>
                </div>

                {/* Replay Controls */}
                <div className="flex items-center gap-2">
                  {!running ? (
                    <button
                      onClick={handleStart}
                      className="flex items-center gap-1.5 px-4 py-1.5 rounded-md bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold shadow-sm shadow-emerald-950 transition-all"
                    >
                      <Play size={14} /> Start Replay
                    </button>
                  ) : (
                    <button
                      onClick={handlePause}
                      className="flex items-center gap-1.5 px-4 py-1.5 rounded-md bg-amber-600 hover:bg-amber-500 text-white text-xs font-semibold shadow-sm shadow-amber-950 transition-all"
                    >
                      <Pause size={14} /> {paused ? 'Resume' : 'Pause'}
                    </button>
                  )}
                  <button
                    onClick={handleReset}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium border border-slate-700 transition-all"
                  >
                    <RotateCcw size={13} /> Reset
                  </button>
                </div>

                {/* Playback Speed Slider */}
                <div className="flex items-center gap-2 text-xs text-slate-400">
                  <span>Speed:</span>
                  <div className="flex items-center gap-1 bg-slate-900 p-1 rounded-md border border-slate-800 font-mono">
                    {[2000, 1000, 500, 200].map(s => (
                      <button
                        key={s}
                        onClick={() => setSpeed(s)}
                        className={`px-2 py-0.5 rounded text-[11px] font-semibold transition-all ${
                          speed === s
                            ? 'bg-cyan-600 text-white'
                            : 'text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        {s === 2000 ? '0.5×' : s === 1000 ? '1×' : s === 500 ? '2×' : '5×'}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Live Status indicator */}
                <div className="flex items-center gap-2 text-xs font-mono font-semibold">
                  <span className={`w-2.5 h-2.5 rounded-full ${running && !paused ? 'bg-emerald-400 animate-ping' : paused ? 'bg-amber-400' : 'bg-slate-600'}`} />
                  <span className={running && !paused ? 'text-emerald-400' : paused ? 'text-amber-400' : 'text-slate-500'}>
                    {running && !paused ? 'LIVE REPLAY' : paused ? 'PAUSED' : 'STANDBY'}
                  </span>
                </div>
              </div>

              {/* Decision Rationale Banner */}
              {ctrl?.decision_rationale && (
                <div className="text-xs text-cyan-300 bg-cyan-950/30 border border-cyan-800/50 rounded-lg p-2.5 font-mono flex items-center gap-2">
                  <span className="text-sm">🤖</span>
                  <span className="flex-1">{ctrl.decision_rationale}</span>
                  <span className="text-[11px] text-slate-400 font-sans">
                    Phase {ctrl.active_phase_id} · {ctrl.signal_state}
                  </span>
                </div>
              )}
            </div>

            {/* KPI Cards Grid */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              <MetricCard
                label="Estimated Queue"
                value={current?.total_queue_pcu?.toFixed(1) ?? '—'}
                unit="PCU"
                icon={<Activity size={16} className="text-amber-400" />}
                accent="amber"
                sub="Hydrodynamic queuing model"
              />
              <MetricCard
                label="Average Delay"
                value={current?.average_delay_sec?.toFixed(1) ?? '—'}
                unit="s/veh"
                icon={<Clock size={16} className="text-rose-400" />}
                accent="red"
                sub="Little's Law delay proxy"
              />
              <MetricCard
                label="Throughput"
                value={current?.total_throughput_pcu?.toFixed(0) ?? '—'}
                unit="PCU"
                icon={<TrendingUp size={16} className="text-emerald-400" />}
                accent="green"
                sub="Cumulative discharged vehicles"
              />
              <MetricCard
                label="Active Strategy"
                value={mode === 'ADAPTIVE' ? 'Adaptive' : mode === 'ACTUATED' ? 'Actuated' : 'Fixed-Time'}
                icon={<Zap size={16} className="text-cyan-400" />}
                accent="cyan"
                sub={`Phase ${ctrl?.active_phase_id ?? 1} · ${ctrl?.signal_state ?? 'GREEN'}`}
              />
            </div>

            {/* Real-time Video Stream & Live Queue Graph */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">
              <TrafficVideoFeed data={current} />
              
              <div className="space-y-6">
                {/* 4-Way Approach Status Summary */}
                <div className="card space-y-3">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                    <span className="text-sm font-bold text-slate-200">Approach Traffic Demand</span>
                    <span className="text-xs text-slate-400 font-mono">IRC:106 PCU Standard</span>
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    {current?.approaches && Object.entries(current.approaches).map(([id, a]) => (
                      <div key={id} className="bg-slate-900/70 border border-slate-800 rounded-lg p-2.5 space-y-1">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-bold text-white truncate max-w-[120px]">{a.name}</span>
                          <SignalBadge state={a.signal} />
                        </div>
                        <div className="flex items-center justify-between text-xs font-mono text-slate-300">
                          <span>Q: <strong className="text-amber-400">{a.queue_pcu.toFixed(1)}</strong> PCU</span>
                          <span>{a.arrival_rate_pcu_per_min.toFixed(1)}/min</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* PCU Traffic Progression Graph (10-Min Samples / 1-Sec Samples) */}
                <div className="card space-y-3">
                  <div className="flex items-center justify-between">
                    <div>
                      <span className="text-sm font-bold text-slate-200">
                        {overviewGraphUnit === '10min' ? 'PCU Traffic Demand & Volume' : 'Live Queue Progression (PCU)'}
                      </span>
                      <div className="text-[11px] text-slate-400">
                        {overviewGraphUnit === '10min' ? '10-Minute Unit Aggregated PCU Samples' : '1-Second Real-Time Telemetry'}
                      </div>
                    </div>

                    <div className="flex items-center gap-1 bg-slate-900 p-1 rounded-md border border-slate-800 text-[11px] font-mono">
                      <button
                        onClick={() => setOverviewGraphUnit('10min')}
                        className={`px-2 py-0.5 rounded transition-all ${
                          overviewGraphUnit === '10min' ? 'bg-cyan-600 text-white font-bold' : 'text-slate-400 hover:text-white'
                        }`}
                      >
                        10-Min PCU
                      </button>
                      <button
                        onClick={() => setOverviewGraphUnit('1sec')}
                        className={`px-2 py-0.5 rounded transition-all ${
                          overviewGraphUnit === '1sec' ? 'bg-cyan-600 text-white font-bold' : 'text-slate-400 hover:text-white'
                        }`}
                      >
                        1-Sec Live
                      </button>
                    </div>
                  </div>

                  <div className="h-52 w-full">
                    {overviewGraphUnit === '10min' ? (
                      aggregated10Min.length > 0 ? (
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={aggregated10Min} margin={{ top: 10, right: 10, left: -15, bottom: 15 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false} />
                            <XAxis dataKey="label" tick={{ fontSize: 9, fill: '#94a3b8' }} angle={-20} textAnchor="end" />
                            <YAxis tick={{ fontSize: 9, fill: '#94a3b8' }} />
                            <Tooltip
                              contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: 6, fontSize: 11 }}
                              formatter={(val: any, name: any) => [`${val} PCU`, name === 'pcu_arrivals' ? 'PCU Arrivals' : name]}
                            />
                            <Bar dataKey="pcu_arrivals" fill="#06b6d4" radius={[3, 3, 0, 0]} name="PCU (10-min)" />
                          </BarChart>
                        </ResponsiveContainer>
                      ) : (
                        <div className="h-full flex items-center justify-center text-xs text-slate-500">
                          Loading 10-minute PCU sample intervals...
                        </div>
                      )
                    ) : (
                      history.length > 0 ? (
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={history} margin={{ top: 10, right: 10, left: -15, bottom: 15 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false} />
                            <XAxis dataKey="time" tick={{ fontSize: 9, fill: '#64748b' }} tickFormatter={t => t.slice(0, 5)} />
                            <YAxis tick={{ fontSize: 9, fill: '#64748b' }} />
                            <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: 6, fontSize: 11 }} />
                            <Line type="monotone" dataKey="queue_pcu" stroke="#22d3ee" strokeWidth={2} dot={false} name="Queue (PCU)" />
                          </LineChart>
                        </ResponsiveContainer>
                      ) : (
                        <div className="h-full flex items-center justify-center text-xs text-slate-500">
                          Press Start Replay to observe real-time queue evolution.
                        </div>
                      )
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ─── CCTV VIDEO FEED TAB ─── */}
        {activeTab === 'video' && (
          <div className="space-y-6">
            <TrafficVideoFeed data={current} />
            <div className="card text-xs text-slate-400 space-y-2">
              <div className="font-bold text-slate-200 text-sm flex items-center gap-2">
                <CheckCircle2 size={16} className="text-emerald-400" />
                ICCC Real-Time Video Surveillance Integration
              </div>
              <p>
                The video stream above replays the traffic arrivals documented in the <strong>Pune Police CCTV Dataset (Jan 2023)</strong>.
                Vehicles spawn in accordance with the real empirical arrival rates, while the live signal state drives stop-line queuing
                and intersection clearance. Object detection bounding boxes illustrate the YOLOv8 classification pipeline (Car, Motorbike, Bus, Truck).
              </p>
            </div>
          </div>
        )}

        {/* ─── JUNCTION VIEW TAB ─── */}
        {activeTab === 'junction' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div>
              <div className="text-sm font-semibold text-slate-300 mb-3">
                4-Way Intersection Schematic · Alankar Chowk
              </div>
              <JunctionViz data={current} />
            </div>
            <div className="space-y-3">
              <div className="text-sm font-semibold text-slate-300">Approach Telemetry</div>
              {current?.approaches
                ? Object.entries(current.approaches).map(([id, a]) => (
                    <div key={id} className="card space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-semibold text-white">{a.name}</span>
                        <SignalBadge state={a.signal} />
                      </div>
                      <div className="grid grid-cols-3 gap-2 text-xs text-slate-400 font-mono">
                        <div>Queue: <strong className="text-white">{a.queue_pcu.toFixed(1)}</strong> PCU</div>
                        <div>Rate: <strong className="text-white">{a.arrival_rate_pcu_per_min.toFixed(1)}</strong>/min</div>
                        <div>Wait: <strong className="text-white">{a.avg_wait_sec.toFixed(1)}</strong>s</div>
                      </div>
                      {a.is_starving && (
                        <div className="text-amber-400 text-xs font-bold bg-amber-950/40 border border-amber-800 p-1 rounded">
                          ⚠ STARVATION WARNING: {a.time_since_served_sec.toFixed(0)}s without service
                        </div>
                      )}
                    </div>
                  ))
                : <div className="text-slate-500 text-xs">Start simulation to see approach data.</div>
              }
            </div>
          </div>
        )}

        {/* ─── 10-MIN GRAPHS / MONITORING TAB ─── */}
        {activeTab === 'monitoring' && (
          <div className="space-y-6">
            {/* Unit Switcher */}
            <div className="card flex flex-wrap items-center justify-between gap-4">
              <div>
                <div className="text-sm font-bold text-white">Traffic Analysis Granularity</div>
                <div className="text-xs text-slate-400">Switch between 10-minute aggregated blocks and instantaneous 1-second telemetry</div>
              </div>
              <div className="flex items-center gap-1.5 bg-slate-900 p-1 rounded-lg border border-slate-800">
                <button
                  onClick={() => setMonitoringUnit('10min')}
                  className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-all ${
                    monitoringUnit === '10min'
                      ? 'bg-cyan-600 text-white shadow-sm'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  📊 10-Minute Units
                </button>
                <button
                  onClick={() => setMonitoringUnit('1sec')}
                  className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-all ${
                    monitoringUnit === '1sec'
                      ? 'bg-cyan-600 text-white shadow-sm'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  ⚡ 1-Sec Telemetry
                </button>
              </div>
            </div>

            {/* Render 10-Min Graphs Component */}
            {monitoringUnit === '10min' ? (
              <AggregatedTrafficCharts intervalSec={600} />
            ) : (
              <div className="space-y-6">
                <div className="card space-y-3">
                  <div className="text-sm font-bold text-slate-200">Instantaneous Queue vs Delay</div>
                  <div className="h-64 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={history}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false} />
                        <XAxis dataKey="time" tick={{ fontSize: 10, fill: '#64748b' }} tickFormatter={t => t.slice(0, 5)} />
                        <YAxis tick={{ fontSize: 10, fill: '#64748b' }} />
                        <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: 6, fontSize: 11 }} />
                        <Legend wrapperStyle={{ fontSize: 11 }} />
                        <Line type="monotone" dataKey="queue_pcu" stroke="#f59e0b" strokeWidth={2} dot={false} name="Queue (PCU)" />
                        <Line type="monotone" dataKey="delay_sec" stroke="#ef4444" strokeWidth={2} dot={false} name="Delay (s)" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="card space-y-3">
                  <div className="text-sm font-bold text-slate-200">Phase Timeline (Last 60 Steps)</div>
                  <div className="h-56 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={history.slice(-60)}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false} />
                        <XAxis dataKey="time" tick={{ fontSize: 10, fill: '#64748b' }} tickFormatter={t => t.slice(3, 8)} />
                        <YAxis tick={{ fontSize: 10, fill: '#64748b' }} domain={[0, 3]} />
                        <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: 6, fontSize: 11 }} />
                        <Bar dataKey="phase_id" fill="#06b6d4" name="Active Phase ID" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ─── 3-WAY BENCHMARK TAB ─── */}
        {activeTab === 'benchmark' && (
          <BenchmarkPanel />
        )}

        {/* ─── SAFETY AUDIT TAB ─── */}
        {activeTab === 'safety' && (
          <SafetyAuditPanel />
        )}

        {/* ─── ML DEMAND FORECASTER TAB ─── */}
        {activeTab === 'ml' && (
          <MLForecastPanel />
        )}

        {/* ─── EVALUATION TAB ─── */}
        {activeTab === 'evaluation' && (
          <div className="space-y-6">
            <div className="card flex flex-wrap items-center justify-between gap-4">
              <div>
                <div className="text-sm font-bold text-white">Comparative Controller Benchmark</div>
                <div className="text-xs text-slate-400">
                  Evaluates both Fixed-Time and Demand-Adaptive controllers across 600s of identical Pune CCTV traffic demand
                </div>
              </div>
              <button
                onClick={runEvaluation}
                disabled={evalRunning}
                className="flex items-center gap-2 px-5 py-2 rounded-lg bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-white text-xs font-semibold shadow-md shadow-cyan-950 transition-all"
              >
                <Activity size={14} className={evalRunning ? 'animate-spin' : ''} />
                {evalRunning ? 'Evaluating...' : 'Run Benchmark (600s)'}
              </button>
            </div>

            {evalSummary?.status === 'ok' && evalSummary.fixed_time && evalSummary.adaptive && (
              <>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {[
                    { label: 'Mean Queue Reduction', val: evalSummary.improvements?.queue_reduction_percent, good: true },
                    { label: 'Average Delay Reduction', val: evalSummary.improvements?.waiting_time_reduction_percent, good: true },
                    { label: 'Throughput Improvement', val: evalSummary.improvements?.throughput_improvement_percent, good: true },
                  ].map(({ label, val, good }) => (
                    <div key={label} className="card text-center p-4">
                      <div className="text-xs text-slate-400 mb-1 font-semibold">{label}</div>
                      <div className={`text-3xl font-extrabold font-mono ${
                        val === undefined ? 'text-slate-500' :
                        (good ? val > 0 : val < 0) ? 'text-emerald-400' : 'text-rose-400'
                      }`}>
                        {val === undefined ? '—' : `${val > 0 ? '+' : ''}${val.toFixed(1)}%`}
                      </div>
                    </div>
                  ))}
                </div>

                {/* Side-by-Side Comparison Table */}
                <div className="card overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="text-slate-400 uppercase text-[11px] border-b border-slate-800 bg-slate-900/60">
                        <th className="py-2.5 px-3 text-left">Performance Metric</th>
                        <th className="py-2.5 px-3 text-right">Fixed-Time Baseline</th>
                        <th className="py-2.5 px-3 text-right font-bold text-cyan-400">Demand-Adaptive</th>
                        <th className="py-2.5 px-3 text-right">Improvement %</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60 font-mono">
                      {[
                        { k: 'Mean Queue Length (PCU)', f: evalSummary.fixed_time.mean_queue_pcu, a: evalSummary.adaptive.mean_queue_pcu, imp: evalSummary.improvements?.queue_reduction_percent },
                        { k: 'Max Queue Peak (PCU)', f: evalSummary.fixed_time.max_queue_pcu, a: evalSummary.adaptive.max_queue_pcu, imp: ((evalSummary.fixed_time.max_queue_pcu - evalSummary.adaptive.max_queue_pcu) / (evalSummary.fixed_time.max_queue_pcu || 1)) * 100 },
                        { k: 'Average Delay (sec/veh)', f: evalSummary.fixed_time.average_wait_sec, a: evalSummary.adaptive.average_wait_sec, imp: evalSummary.improvements?.waiting_time_reduction_percent },
                        { k: 'Discharged Throughput (PCU)', f: evalSummary.fixed_time.total_throughput_pcu, a: evalSummary.adaptive.total_throughput_pcu, imp: evalSummary.improvements?.throughput_improvement_percent },
                        { k: 'Starvation Occurrences', f: evalSummary.fixed_time.starvation_occurrences, a: evalSummary.adaptive.starvation_occurrences, imp: null },
                      ].map(({ k, f, a, imp }) => (
                        <tr key={k} className="hover:bg-slate-900/40">
                          <td className="py-2.5 px-3 text-slate-300 font-sans">{k}</td>
                          <td className="py-2.5 px-3 text-right text-slate-400">{typeof f === 'number' ? f.toFixed(1) : f}</td>
                          <td className="py-2.5 px-3 text-right font-bold text-cyan-400">{typeof a === 'number' ? a.toFixed(1) : a}</td>
                          <td className="py-2.5 px-3 text-right font-bold text-emerald-400">
                            {imp != null ? `${imp > 0 ? '+' : ''}${imp.toFixed(1)}%` : '—'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div className="flex gap-3">
                  <button onClick={() => exportEval('json')} className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs text-white border border-slate-700">
                    <Download size={14} /> Export JSON Benchmark
                  </button>
                  <button onClick={() => exportEval('csv')} className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs text-white border border-slate-700">
                    <Download size={14} /> Export Time-Series CSV
                  </button>
                </div>
              </>
            )}
          </div>
        )}

        {/* ─── DECISION LOG TAB ─── */}
        {activeTab === 'log' && (
          <div className="card space-y-3">
            <div className="flex items-center justify-between border-b border-slate-800 pb-2">
              <span className="text-sm font-bold text-slate-200">Adaptive Decision Rationale Audit Log</span>
              <span className="text-xs text-slate-400 font-mono">Last {decisionLog.length} steps</span>
            </div>
            <div className="overflow-x-auto max-h-[60vh]">
              <table className="w-full text-xs">
                <thead className="sticky top-0 bg-slate-900">
                  <tr className="text-slate-400 uppercase text-[10px] border-b border-slate-800">
                    <th className="py-2 px-2 text-left">Step</th>
                    <th className="py-2 px-2 text-left">Timestamp</th>
                    <th className="py-2 px-2 text-left">Signal</th>
                    <th className="py-2 px-2 text-left">Active Phase</th>
                    <th className="py-2 px-2 text-left">Decision Rationale</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono text-slate-300">
                  {[...decisionLog].reverse().map((d, i) => (
                    <tr key={i} className="hover:bg-slate-900/40">
                      <td className="py-1.5 px-2 text-slate-500">{d.step}</td>
                      <td className="py-1.5 px-2 text-slate-400">{d.time}</td>
                      <td className="py-1.5 px-2"><SignalBadge state={d.signal} /></td>
                      <td className="py-1.5 px-2 text-white">{d.phase_name}</td>
                      <td className="py-1.5 px-2 text-slate-400 font-sans text-[11px]">{d.rationale}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ─── DATASET TAB ─── */}
        {activeTab === 'dataset' && (
          <div className="space-y-4">
            <DatasetInfo />
          </div>
        )}

      </main>
    </div>
  )
}

function DatasetInfo() {
  const [summary, setSummary] = useState<Record<string, unknown> | null>(null)
  const [schema, setSchema] = useState<Record<string, unknown> | null>(null)

  useEffect(() => {
    Promise.all([api.datasetSummary(), api.datasetSchema()]).then(([s, sc]) => {
      setSummary(s); setSchema(sc)
    })
  }, [])

  return (
    <>
      <div className="card space-y-3">
        <div className="flex items-center gap-2">
          <Database size={18} className="text-cyan-400" />
          <span className="font-bold text-white text-base">Heterogeneous Traffic Count Dataset, Pune (Jan 2023)</span>
        </div>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 text-xs text-slate-400 font-mono">
          <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800">
            Junctions<br />
            <span className="text-white font-bold text-base">{summary?.junction_count != null ? String(summary.junction_count) : '—'}</span>
          </div>
          <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800">
            Total Files<br />
            <span className="text-white font-bold text-base">{summary?.total_files != null ? String(summary.total_files) : '—'}</span>
          </div>
          <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800">
            Camera FPS<br />
            <span className="text-white font-bold text-base">25.0 fps</span>
          </div>
          <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800">
            Coverage Period<br />
            <span className="text-white font-bold text-base">Jan 11–18 2023</span>
          </div>
        </div>
      </div>

      <div className="card space-y-3">
        <div className="text-sm font-bold text-slate-200">Schema & Indian IRC:106 PCU Equivalencies</div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-slate-400 text-[11px] uppercase border-b border-slate-800 bg-slate-900/60">
                <th className="py-2.5 px-3 text-left">Column</th>
                <th className="py-2.5 px-3 text-left">Data Type</th>
                <th className="py-2.5 px-3 text-left">PCU Weight</th>
                <th className="py-2.5 px-3 text-left">Description</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {((schema?.columns || []) as Record<string, unknown>[]).map((col, i) => (
                <tr key={i} className="hover:bg-slate-900/40">
                  <td className="py-2 px-3 text-cyan-400 font-bold">{String(col.name)}</td>
                  <td className="py-2 px-3 text-slate-400">{String(col.type)}</td>
                  <td className="py-2 px-3 text-amber-400 font-bold">{col.pcu_weight !== undefined ? String(col.pcu_weight) : '—'}</td>
                  <td className="py-2 px-3 text-slate-300 font-sans">{String(col.description)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </>
  )
}
