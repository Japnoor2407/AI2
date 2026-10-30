import React, { useEffect, useState, useCallback, useRef } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer,
} from 'recharts'
import {
  Play, Pause, RotateCcw, Activity, AlertTriangle,
  Database, Zap, Clock, TrendingUp, Video,
  MapPin, ChevronRight, Info, Shield, Brain, BarChart3,
  CheckCircle, FileText, Cpu,
} from 'lucide-react'

import { api } from '../services/api'
import type {
  TrafficCurrent, ControllerStatus,
  HistoryPoint, DecisionLogEntry,
} from '../types'
import { JunctionViz } from '../components/JunctionViz'
import { SignalBadge } from '../components/SignalBadge'
import { TrafficVideoFeed } from '../components/TrafficVideoFeed'
import { AggregatedTrafficCharts } from '../components/AggregatedTrafficCharts'
import { SafetyAuditPanel } from '../components/SafetyAuditPanel'
import { MLForecastPanel } from '../components/MLForecastPanel'
import { RLTrainingPanel } from '../components/RLTrainingPanel'
import StrategyComparisonLab from '../components/StrategyComparisonLab'
import EmergencyPriorityControl from '../components/EmergencyPriorityControl'

// ─── Helpers ────────────────────────────────────────────────────────────────

function trafficStatus(avgDelay: number | null): {
  label: string; color: string; dot: string
} {
  if (avgDelay === null) return { label: 'Unknown', color: 'text-slate-400', dot: 'bg-slate-500' }
  if (avgDelay < 10) return { label: 'Normal', color: 'text-emerald-400', dot: 'bg-emerald-400' }
  if (avgDelay < 25) return { label: 'Busy', color: 'text-amber-400', dot: 'bg-amber-400' }
  return { label: 'Congested', color: 'text-rose-400', dot: 'bg-rose-400' }
}

function simplifyRationale(raw: string | null | undefined): string {
  if (!raw) return 'Monitoring all roads…'
  const r = raw.toLowerCase()
  if (r.includes('starv')) return 'A road has been waiting too long — giving it priority now.'
  if (r.includes('high demand') || r.includes('demand score')) return 'Giving more green time to the busiest road right now.'
  if (r.includes('fixed') || r.includes('pre-timed')) return 'Following a fixed schedule — signals change on a timer.'
  if (r.includes('predict') || r.includes('forecast')) return 'Anticipating heavier traffic ahead — adjusting signals early.'
  if (r.includes('yellow') || r.includes('clearance')) return 'Clearing the intersection safely before switching signals.'
  if (r.includes('all_red') || r.includes('all-red')) return 'Brief safety pause — all roads are stopped momentarily.'
  if (r.includes('ppo') || r.includes('deep_rl') || r.includes('reinforcement')) return 'AI (PPO) chose the best signal based on learned traffic patterns.'
  if (r.includes('phase')) return 'Switching to the next group of roads.'
  return 'Monitoring live traffic and adjusting signals as needed.'
}

const MODE_OPTIONS = [
  {
    id: 'ADAPTIVE' as const,
    label: 'Let AI Decide',
    desc: 'The system automatically adjusts signal timing based on live traffic',
    accent: 'bg-cyan-600 text-white',
  },
  {
    id: 'ACTUATED' as const,
    label: 'On-Demand',
    desc: 'Signals extend green only when vehicles are detected waiting',
    accent: 'bg-emerald-600 text-white',
  },
  {
    id: 'FIXED_TIME' as const,
    label: 'Timed Signals',
    desc: 'All signals follow a fixed schedule regardless of traffic',
    accent: 'bg-amber-600 text-white',
  },
  {
    id: 'DEEP_RL' as const,
    label: 'AI (PPO)',
    desc: 'Deep reinforcement learning agent trained to minimise waiting time and queue length',
    accent: 'bg-purple-600 text-white',
  },
] as const

// ─── Sub-components ──────────────────────────────────────────────────────────

function SummaryCard({
  label, value, sub, accent = 'cyan', icon,
}: {
  label: string; value: React.ReactNode; sub: string
  accent?: 'cyan' | 'amber' | 'emerald' | 'rose'; icon: React.ReactNode
}) {
  const ring: Record<string, string> = {
    cyan: 'border-cyan-800/60',
    amber: 'border-amber-800/50',
    emerald: 'border-emerald-800/50',
    rose: 'border-rose-800/50',
  }
  const ic: Record<string, string> = {
    cyan: 'text-cyan-400',
    amber: 'text-amber-400',
    emerald: 'text-emerald-400',
    rose: 'text-rose-400',
  }
  return (
    <div className={`card border ${ring[accent]} flex flex-col gap-2`}>
      <div className={`flex items-center gap-2 text-xs font-semibold uppercase tracking-wider ${ic[accent]}`}>
        {icon}
        <span>{label}</span>
      </div>
      <div className="text-2xl font-bold text-white tracking-tight font-mono">{value}</div>
      <div className="text-[12px] text-slate-400 leading-snug">{sub}</div>
    </div>
  )
}

function StatusDot({ running, paused }: { running: boolean; paused: boolean }) {
  if (running && !paused) return (
    <span className="flex items-center gap-1.5 text-xs font-semibold text-emerald-400">
      <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping" />
      Live
    </span>
  )
  if (paused) return (
    <span className="flex items-center gap-1.5 text-xs font-semibold text-amber-400">
      <span className="w-2 h-2 rounded-full bg-amber-400" />
      Paused
    </span>
  )
  return (
    <span className="flex items-center gap-1.5 text-xs font-semibold text-slate-400">
      <span className="w-2 h-2 rounded-full bg-slate-600" />
      Ready
    </span>
  )
}

// ─── Advanced sub-tab ────────────────────────────────────────────────────────

type AdvancedSub = 'safety' | 'ml' | 'rl' | 'lab' | 'log' | 'dataset'

function AdvancedTabBar({
  sub, setSub,
}: { sub: AdvancedSub; setSub: (s: AdvancedSub) => void }) {
  const items: { id: AdvancedSub; label: string; icon: React.ReactNode }[] = [
    { id: 'lab',     label: 'Strategy Lab',    icon: <BarChart3 size={13} /> },
    { id: 'rl',      label: 'AI Controller (PPO)', icon: <Cpu size={13} /> },
    { id: 'ml',      label: 'AI Forecasting', icon: <Brain size={13} /> },
    { id: 'safety',  label: 'Safety Monitor', icon: <Shield size={13} /> },
    { id: 'log',     label: 'Decision Log',   icon: <FileText size={13} /> },
    { id: 'dataset', label: 'Dataset Info',   icon: <Database size={13} /> },
  ]
  return (
    <div className="flex items-center gap-1 flex-wrap">
      {items.map(t => (
        <button
          key={t.id}
          onClick={() => setSub(t.id)}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-semibold transition-all ${
            sub === t.id
              ? 'bg-slate-700 text-white'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
          }`}
        >
          {t.icon}{t.label}
        </button>
      ))}
    </div>
  )
}

// ─── Main Dashboard ──────────────────────────────────────────────────────────

type Tab = 'overview' | 'map' | 'video' | 'compare' | 'advanced'

export default function Dashboard() {
  const [running, setRunning]   = useState(false)
  const [paused, setPaused]     = useState(false)
  const [speed, setSpeed]       = useState(1000)
  const [mode, setMode]         = useState<'ADAPTIVE' | 'PREDICTION_ASSISTED' | 'ACTUATED' | 'FIXED_TIME' | 'DEEP_RL'>('ADAPTIVE')
  const [status, setStatus]     = useState<ControllerStatus | null>(null)
  const [current, setCurrent]   = useState<TrafficCurrent | null>(null)
  const [history, setHistory]   = useState<HistoryPoint[]>([])
  const [decisionLog, setDecisionLog] = useState<DecisionLogEntry[]>([])
  const [apiError, setApiError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<Tab>('overview')
  const [advSub, setAdvSub]     = useState<AdvancedSub>('lab')
  const [showModeTooltip, setShowModeTooltip] = useState<string | null>(null)
  const [sliderPos, setSliderPos] = useState<number | null>(null)
  const seekDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const fetchAll = useCallback(async () => {
    try {
      const [st, cur, hist, dl] = await Promise.all([
        api.controllerStatus(),
        api.trafficCurrent(),
        api.trafficHistory(300),
        api.decisionLog(50),
      ])
      setStatus(st)
      if (cur?.step !== undefined) setCurrent(cur)
      setHistory((hist?.history || []).slice(-150))
      setDecisionLog(dl?.log || [])
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

  const handleModeSwitch = async (m: 'ADAPTIVE' | 'PREDICTION_ASSISTED' | 'ACTUATED' | 'FIXED_TIME' | 'DEEP_RL') => {
    setMode(m)
    await api.setMode(m)
  }

  const handleSeek = async (targetStep: number) => {
    try {
      await api.simulationSeek(targetStep)
      await fetchAll()
    } catch (e: unknown) { console.error(e) }
  }

  const onSliderChange = (newVal: number) => {
    setSliderPos(newVal)
    if (seekDebounceRef.current) clearTimeout(seekDebounceRef.current)
    seekDebounceRef.current = setTimeout(async () => {
      await handleSeek(newVal)
      setSliderPos(null)
    }, 150)
  }

  const onSliderCommit = async (newVal?: number) => {
    if (seekDebounceRef.current) clearTimeout(seekDebounceRef.current)
    const target = newVal !== undefined ? newVal : (sliderPos ?? curStep)
    setSliderPos(null)
    await handleSeek(target)
  }

  const ctrl = current?.controller

  // Derived summary values
  const approaches = current?.approaches ? Object.values(current.approaches) : []
  const busyRoads  = approaches.filter(a => a.queue_pcu > 2).length
  const avgDelay   = current?.average_delay_sec ?? null
  const throughput = current?.total_throughput_pcu ?? null
  const tStatus    = trafficStatus(avgDelay)
  const totalStep  = status?.total_steps ?? 1920
  const curStep    = status?.step ?? 0
  const timeLabel  = current?.timestamp ?? (status?.date ? `${status.date}` : '09:00:00')

  const displayStep = sliderPos !== null ? sliderPos : curStep
  const calcTime = (step: number) => {
    const totalSec = 9 * 3600 + step
    const h = Math.floor(totalSec / 3600) % 24
    const m = Math.floor((totalSec % 3600) / 60)
    const s = totalSec % 60
    return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
  }
  const displayTime = sliderPos !== null ? calcTime(sliderPos) : timeLabel

  const tabs: { id: Tab; label: string; icon: React.ReactNode }[] = [
    { id: 'overview', label: 'Overview',          icon: <Activity size={14} /> },
    { id: 'map',      label: 'Live Traffic',       icon: <MapPin size={14} /> },
    { id: 'video',    label: 'Live Camera',        icon: <Video size={14} /> },
    { id: 'compare',  label: 'Compare Strategies', icon: <BarChart3 size={14} /> },
    { id: 'advanced', label: 'AI & Reports',       icon: <ChevronRight size={14} /> },
  ]

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">

      {/* ── Header ────────────────────────────────────────────────────────── */}
      <header className="bg-slate-900 border-b border-slate-800 px-5 py-3 sticky top-0 z-30 shadow-sm">
        <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-3">

          {/* Brand */}
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-cyan-600/20 border border-cyan-600/30 flex items-center justify-center text-cyan-400 shrink-0">
              <Zap size={18} />
            </div>
            <div>
              <div className="text-sm font-bold text-white leading-tight">Traffic Overview — Alankar Chowk, Pune</div>
              <div className="text-[11px] text-slate-400 leading-tight">See live traffic conditions and how the system is managing congestion.</div>
            </div>
          </div>

          {/* Error */}
          {apiError && (
            <div className="flex items-center gap-2 text-rose-400 text-xs border border-rose-800/50 bg-rose-950/30 px-3 py-1.5 rounded-lg">
              <AlertTriangle size={13} /> {apiError}
            </div>
          )}

          {/* Right: status + clock */}
          <div className="flex items-center gap-4">
            <StatusDot running={running} paused={paused} />
            <div className="h-5 w-px bg-slate-800" />
            <div className="text-right">
              <div className="text-[10px] text-slate-500 uppercase tracking-wider">Time</div>
              <div className="text-sm font-mono font-bold text-white">{timeLabel}</div>
            </div>
            <div className="text-right hidden sm:block">
              <div className="text-[10px] text-slate-500 uppercase tracking-wider">Progress</div>
              <div className="text-xs font-mono text-slate-300">
                {curStep} <span className="text-slate-600">/</span> {totalStep}
              </div>
            </div>
            {status?.signal_state && <SignalBadge state={status.signal_state} />}
          </div>
        </div>
      </header>

      {/* ── Nav Tabs ──────────────────────────────────────────────────────── */}
      <nav className="bg-slate-900/90 border-b border-slate-800/80 px-5 backdrop-blur-sm sticky top-[57px] z-20">
        <div className="max-w-7xl mx-auto flex items-center gap-1 overflow-x-auto py-2">
          {tabs.map(t => (
            <button
              key={t.id}
              onClick={() => setActiveTab(t.id)}
              className={`flex items-center gap-1.5 px-3.5 py-1.5 text-xs font-semibold rounded-lg transition-all whitespace-nowrap ${
                activeTab === t.id
                  ? 'bg-cyan-600 text-white border border-cyan-500/40'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
              }`}
            >
              {t.icon}
              {t.label}
            </button>
          ))}
        </div>
      </nav>

      {/* ── Main Content ──────────────────────────────────────────────────── */}
      <main className="max-w-7xl mx-auto w-full p-5 space-y-5 flex-1">

        {/* ══ OVERVIEW TAB ══════════════════════════════════════════════════ */}
        {activeTab === 'overview' && (
          <div className="space-y-5">

            {/* ── Controls ────────────────────────────────────────────────── */}
            <div className="card space-y-4">

              {/* Controller mode selector */}
              <div>
                <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Management Strategy</div>
                <div className="flex flex-wrap gap-2">
                  {MODE_OPTIONS.map(opt => (
                    <div key={opt.id} className="relative">
                      <button
                        onClick={() => handleModeSwitch(opt.id)}
                        className={`px-4 py-2 rounded-lg text-sm font-semibold transition-all border ${
                          mode === opt.id
                            ? `${opt.accent} border-transparent shadow-sm`
                            : 'text-slate-400 bg-slate-900 border-slate-700 hover:text-white hover:border-slate-600'
                        }`}
                      >
                        {opt.label}
                      </button>
                      <button
                        onMouseEnter={() => setShowModeTooltip(opt.id)}
                        onMouseLeave={() => setShowModeTooltip(null)}
                        className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-slate-700 text-slate-400 hover:text-white flex items-center justify-center"
                      >
                        <Info size={10} />
                      </button>
                      {showModeTooltip === opt.id && (
                        <div className="absolute top-full left-0 mt-1.5 w-56 bg-slate-800 border border-slate-700 rounded-lg p-2.5 text-[11px] text-slate-300 z-50 shadow-lg">
                          {opt.desc}
                        </div>
                      )}
                    </div>
                  ))}
                  {/* Predictive mode — in mode selector but styled subtly */}
                  <button
                    onClick={() => handleModeSwitch('PREDICTION_ASSISTED')}
                    className={`px-4 py-2 rounded-lg text-sm font-semibold transition-all border ${
                      mode === 'PREDICTION_ASSISTED'
                        ? 'bg-violet-600 text-white border-transparent shadow-sm'
                        : 'text-slate-400 bg-slate-900 border-slate-700 hover:text-white hover:border-slate-600'
                    }`}
                    title="Uses AI demand prediction to pre-adjust signals before congestion builds"
                  >
                    AI + Forecast
                  </button>
                </div>
              </div>

              {/* Simulation controls */}
              <div className="flex flex-wrap items-center justify-between gap-3 pt-3 border-t border-slate-800">
                <div className="flex items-center gap-2">
                  {!running ? (
                    <button
                      onClick={handleStart}
                      className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-sm font-semibold shadow-sm transition-all"
                    >
                      <Play size={15} /> Start
                    </button>
                  ) : (
                    <button
                      onClick={handlePause}
                      className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-amber-600 hover:bg-amber-500 text-white text-sm font-semibold shadow-sm transition-all"
                    >
                      <Pause size={15} /> {paused ? 'Resume' : 'Pause'}
                    </button>
                  )}
                  <button
                    onClick={handleReset}
                    className="flex items-center gap-1.5 px-3 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-sm font-medium border border-slate-700 transition-all"
                  >
                    <RotateCcw size={14} /> Reset
                  </button>
                </div>

                {/* Speed */}
                <div className="flex items-center gap-2">
                  <span className="text-xs text-slate-500">Speed:</span>
                  <div className="flex items-center gap-1 bg-slate-900 p-1 rounded-md border border-slate-800">
                    {[2000, 1000, 500, 200].map(s => (
                      <button
                        key={s}
                        onClick={() => setSpeed(s)}
                        className={`px-2 py-0.5 rounded text-[11px] font-semibold transition-all ${
                          speed === s ? 'bg-cyan-600 text-white' : 'text-slate-400 hover:text-white'
                        }`}
                      >
                        {s === 2000 ? '½×' : s === 1000 ? '1×' : s === 500 ? '2×' : '5×'}
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              {/* Timeline scrubber */}
              <div className="flex items-center gap-3 pt-2 border-t border-slate-800">
                <span className="text-[11px] text-slate-500 whitespace-nowrap">Timeline</span>
                <span className="text-[11px] text-slate-600 font-mono whitespace-nowrap">09:00</span>
                <button
                  onClick={() => onSliderCommit(Math.max(0, curStep - 10))}
                  disabled={curStep <= 0}
                  className="px-2 py-1 bg-slate-800 hover:bg-slate-700 disabled:opacity-30 rounded text-slate-300 text-[11px] font-mono transition-colors"
                  title="Rewind 10 seconds"
                >
                  −10s
                </button>
                <input
                  type="range"
                  min={0}
                  max={Math.max(1, totalStep - 1)}
                  value={displayStep}
                  onChange={e => onSliderChange(Number(e.target.value))}
                  onMouseUp={() => onSliderCommit()}
                  onTouchEnd={() => onSliderCommit()}
                  className="flex-1 accent-cyan-500 cursor-pointer h-1.5"
                />
                <button
                  onClick={() => onSliderCommit(Math.min(totalStep - 1, curStep + 10))}
                  disabled={curStep >= totalStep - 1}
                  className="px-2 py-1 bg-slate-800 hover:bg-slate-700 disabled:opacity-30 rounded text-slate-300 text-[11px] font-mono transition-colors"
                  title="Forward 10 seconds"
                >
                  +10s
                </button>
                <span className="text-[11px] text-slate-600 font-mono whitespace-nowrap">09:30</span>
                <span className="text-cyan-400 font-mono font-bold text-[11px] whitespace-nowrap min-w-[55px] text-right">
                  {displayTime}
                </span>
              </div>

              {/* Plain-English decision banner */}
              <div className="pt-2 border-t border-slate-800">
                <div className="flex items-start gap-2 text-sm text-slate-300 bg-slate-900/60 border border-slate-800 rounded-lg px-3 py-2.5">
                  <span className="shrink-0 text-base mt-0.5">💡</span>
                  <span className="flex-1">{simplifyRationale(ctrl?.decision_rationale)}</span>
                  {ctrl?.phase_name && (
                    <span className="shrink-0 text-xs text-slate-500">{ctrl.phase_name}</span>
                  )}
                </div>
              </div>
            </div>

            {/* ── 4 Summary Cards ─────────────────────────────────────────── */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              <SummaryCard
                label="Busy Roads"
                value={<>{busyRoads} <span className="text-base font-normal text-slate-400">/ {approaches.length}</span></>}
                sub="Roads with high vehicle queues right now"
                accent="amber"
                icon={<Activity size={14} />}
              />
              <SummaryCard
                label="Average Wait"
                value={avgDelay !== null ? `${avgDelay.toFixed(0)}s` : '—'}
                sub="Average time a vehicle waits at this junction"
                accent="rose"
                icon={<Clock size={14} />}
              />
              <SummaryCard
                label="Vehicles Passing"
                value={throughput !== null ? Math.round(throughput).toLocaleString() : '—'}
                sub="Total vehicles passed through since start"
                accent="emerald"
                icon={<TrendingUp size={14} />}
              />
              <SummaryCard
                label="Traffic Status"
                value={
                  <span className={tStatus.color}>
                    <span className={`inline-block w-2.5 h-2.5 rounded-full mr-2 ${tStatus.dot}`} />
                    {tStatus.label}
                  </span>
                }
                sub={
                  avgDelay !== null
                    ? avgDelay < 10
                      ? 'Traffic is flowing well. No significant delays.'
                      : avgDelay < 25
                      ? 'Moderate congestion. System is adapting.'
                      : 'Heavy congestion detected. Priority adjustments active.'
                    : 'Start the simulation to see live status.'
                }
                accent={
                  avgDelay === null ? 'cyan'
                    : avgDelay < 10 ? 'emerald'
                    : avgDelay < 25 ? 'amber'
                    : 'rose'
                }
                icon={<CheckCircle size={14} />}
              />
            </div>

            {/* ── Road Status + Queue Chart ────────────────────────────────── */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 items-start">

              {/* Per-road approach summary — plain English */}
              <div className="card space-y-3">
                <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                  <span className="text-sm font-bold text-slate-200">Road-by-Road Status</span>
                  <span className="text-[11px] text-slate-500">Updated every second</span>
                </div>
                {approaches.length > 0 ? (
                  <div className="space-y-2">
                    {approaches.map(a => (
                      <div
                        key={a.direction}
                        className="flex items-center justify-between bg-slate-900/60 border border-slate-800 rounded-lg px-3 py-2.5"
                      >
                        <div className="flex-1 min-w-0">
                          <div className="text-sm font-semibold text-white truncate">{a.name}</div>
                          <div className="text-[11px] text-slate-400 mt-0.5">
                            {a.queue_pcu > 0
                              ? `${Math.round(a.queue_pcu * 1.2)} vehicles waiting · ${a.avg_wait_sec.toFixed(0)}s avg wait`
                              : 'Road is clear'}
                            {a.is_starving && (
                              <span className="ml-2 text-amber-400 font-semibold">⚠ Waiting too long</span>
                            )}
                          </div>
                        </div>
                        <SignalBadge state={a.signal} />
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-sm text-slate-500 text-center py-8">
                    Press <strong>Start</strong> to see live road conditions.
                  </div>
                )}
              </div>

              {/* Queue chart */}
              <div className="card space-y-3">
                <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                  <div>
                    <div className="text-sm font-bold text-slate-200">Vehicles Waiting Over Time</div>
                    <div className="text-[11px] text-slate-500">How many vehicles are queued at the junction</div>
                  </div>
                </div>
                <div className="h-52 w-full">
                  {history.length > 0 ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={history} margin={{ top: 8, right: 8, left: -18, bottom: 8 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                        <XAxis
                          dataKey="time"
                          tick={{ fontSize: 9, fill: '#64748b' }}
                          tickFormatter={t => t.slice(0, 5)}
                          interval="preserveStartEnd"
                        />
                        <YAxis tick={{ fontSize: 9, fill: '#64748b' }} />
                        <Tooltip
                          contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: 6, fontSize: 11 }}
                          formatter={(val: any) => [`${Number(val ?? 0).toFixed(1)} vehicles`, 'Queue']}
                          labelFormatter={l => `Time: ${l}`}
                        />
                        <Line
                          type="monotone"
                          dataKey="queue_pcu"
                          stroke="#22d3ee"
                          strokeWidth={2}
                          dot={false}
                          name="Vehicles waiting"
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="h-full flex flex-col items-center justify-center text-slate-500 text-sm gap-2">
                      <Activity size={24} className="text-slate-700" />
                      <span>Start simulation to view the live queue chart</span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ══ TRAFFIC MAP TAB ═══════════════════════════════════════════════ */}
        {activeTab === 'map' && (
          <div className="space-y-5">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 items-start">
              <div>
                <div className="text-sm font-semibold text-slate-300 mb-3 flex items-center gap-2">
                  <MapPin size={15} className="text-cyan-400" />
                  Alankar Chowk — Live Intersection View
                </div>
                <JunctionViz data={current} />
              </div>
              <div className="space-y-3">
                <div className="text-sm font-semibold text-slate-300 flex items-center gap-2">
                  <Activity size={15} className="text-cyan-400" />
                  Road Details
                </div>
                {current?.approaches
                  ? Object.entries(current.approaches).map(([id, a]) => (
                      <div key={id} className="card space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-sm font-semibold text-white">{a.name}</span>
                          <SignalBadge state={a.signal} />
                        </div>
                        <div className="grid grid-cols-3 gap-3 text-xs text-slate-400">
                          <div>
                            <div className="text-[10px] uppercase tracking-wider text-slate-600 mb-0.5">Waiting</div>
                            <span className="text-white font-semibold">{Math.round(a.queue_pcu * 1.2)} vehicles</span>
                          </div>
                          <div>
                            <div className="text-[10px] uppercase tracking-wider text-slate-600 mb-0.5">Arrival rate</div>
                            <span className="text-white font-semibold">{a.arrival_rate_pcu_per_min.toFixed(0)}/min</span>
                          </div>
                          <div>
                            <div className="text-[10px] uppercase tracking-wider text-slate-600 mb-0.5">Avg wait</div>
                            <span className="text-white font-semibold">{a.avg_wait_sec.toFixed(0)}s</span>
                          </div>
                        </div>
                        {a.is_starving && (
                          <div className="text-amber-400 text-xs font-semibold bg-amber-950/30 border border-amber-800/60 px-2 py-1 rounded-md">
                            ⚠ This road has been waiting too long — the system is requesting priority.
                          </div>
                        )}
                      </div>
                    ))
                  : <div className="text-slate-500 text-sm py-6 text-center">Start simulation to see approach data.</div>
                }
              </div>
            </div>

            {/* 10-Min historical charts below the map */}
            <div className="card space-y-3">
              <div className="text-sm font-bold text-slate-200">Historical Traffic Demand (10-Minute Blocks)</div>
              <AggregatedTrafficCharts intervalSec={600} />
            </div>
            {/* Emergency Vehicle Priority Panel */}
            <EmergencyPriorityControl />
          </div>
        )}

        {/* ══ LIVE CAMERA TAB ═══════════════════════════════════════════════ */}
        {activeTab === 'video' && (
          <div className="space-y-5">
            <TrafficVideoFeed data={current} />
            <div className="card text-sm text-slate-400 space-y-2">
              <div className="font-semibold text-slate-200 flex items-center gap-2">
                <Video size={16} className="text-cyan-400" /> About this feed
              </div>
              <p>
                This view replays real vehicle arrivals from the <strong>Pune Police CCTV Dataset (Jan 2023)</strong> for Alankar Chowk. Vehicles appear according to actual recorded counts. Object detection boxes show how the system identifies cars, motorbikes, buses, and trucks. The signal in the corner reflects live simulation state.
              </p>
            </div>
          </div>
        )}

        {/* ══ COMPARE STRATEGIES TAB ════════════════════════════════════════ */}
        {activeTab === 'compare' && (
          <div className="space-y-5">
            <StrategyComparisonLab />
          </div>
        )}

        {/* ══ ADVANCED TAB ══════════════════════════════════════════════════ */}
        {activeTab === 'advanced' && (
          <div className="space-y-5">
            <div className="card flex flex-wrap items-center justify-between gap-3">
              <div>
                <div className="text-sm font-bold text-white">AI Controller, Strategy Lab & Technical Reports</div>
                <div className="text-xs text-slate-400 mt-0.5">
                  Strategy comparison, PPO AI training, safety monitoring, forecasting, signal logs, and dataset reference.
                </div>
              </div>
              <AdvancedTabBar sub={advSub} setSub={setAdvSub} />
            </div>

            {advSub === 'lab'     && <StrategyComparisonLab />}
            {advSub === 'safety'  && <SafetyAuditPanel />}
            {advSub === 'ml'      && <MLForecastPanel />}
            {advSub === 'rl'      && <RLTrainingPanel />}

            {advSub === 'log' && (
              <div className="card space-y-3">
                <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                  <span className="text-sm font-bold text-slate-200">Signal Decision Log</span>
                  <span className="text-xs text-slate-400">Last {decisionLog.length} decisions</span>
                </div>
                <div className="overflow-x-auto max-h-[60vh]">
                  <table className="w-full text-xs">
                    <thead className="sticky top-0 bg-slate-900">
                      <tr className="text-slate-400 uppercase text-[10px] border-b border-slate-800">
                        <th className="py-2 px-2 text-left">Step</th>
                        <th className="py-2 px-2 text-left">Time</th>
                        <th className="py-2 px-2 text-left">Signal</th>
                        <th className="py-2 px-2 text-left">Road Group</th>
                        <th className="py-2 px-2 text-left">Reason</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60 font-mono text-slate-300">
                      {[...decisionLog].reverse().map((d, i) => (
                        <tr key={i} className="hover:bg-slate-900/40">
                          <td className="py-1.5 px-2 text-slate-500">{d.step}</td>
                          <td className="py-1.5 px-2 text-slate-400">{d.time}</td>
                          <td className="py-1.5 px-2"><SignalBadge state={d.signal} /></td>
                          <td className="py-1.5 px-2 text-white font-sans">{d.phase_name}</td>
                          <td className="py-1.5 px-2 text-slate-400 font-sans text-[11px]">
                            {simplifyRationale(d.rationale)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {decisionLog.length === 0 && (
                    <div className="text-center text-slate-500 text-sm py-10">
                      Start the simulation to record signal decisions.
                    </div>
                  )}
                </div>
              </div>
            )}

            {advSub === 'dataset' && <DatasetInfo />}
          </div>
        )}

      </main>
    </div>
  )
}

// ─── Dataset Info (unchanged logic, cleaner wording) ─────────────────────────

function DatasetInfo() {
  const [summary, setSummary] = useState<Record<string, unknown> | null>(null)
  const [schema,  setSchema]  = useState<Record<string, unknown> | null>(null)

  useEffect(() => {
    Promise.all([api.datasetSummary(), api.datasetSchema()]).then(([s, sc]) => {
      setSummary(s); setSchema(sc)
    })
  }, [])

  return (
    <>
      <div className="card space-y-4">
        <div className="flex items-center gap-2">
          <Database size={18} className="text-cyan-400" />
          <span className="font-bold text-white">Pune Traffic Count Dataset (Jan 2023)</span>
        </div>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 text-xs text-slate-400">
          {[
            { label: 'Junctions', val: summary?.junction_count },
            { label: 'Total Files', val: summary?.total_files },
            { label: 'Camera FPS', val: '25.0' },
            { label: 'Coverage', val: 'Jan 11–18 2023' },
          ].map(({ label, val }) => (
            <div key={label} className="bg-slate-900/60 p-3 rounded-lg border border-slate-800">
              <div className="text-[10px] uppercase tracking-wider mb-1">{label}</div>
              <span className="text-white font-bold text-base">{val != null ? String(val) : '—'}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="card space-y-3">
        <div className="text-sm font-bold text-slate-200">Column Reference</div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-slate-400 text-[11px] uppercase border-b border-slate-800 bg-slate-900/60">
                <th className="py-2.5 px-3 text-left">Column</th>
                <th className="py-2.5 px-3 text-left">Type</th>
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
