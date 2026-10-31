import React, { useEffect, useState, useCallback, useRef } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer,
} from 'recharts'
import {
  Play, Pause, RotateCcw, Activity, AlertTriangle,
  Database, Zap, Clock, TrendingUp, Video,
  MapPin, Info, Shield, Brain, BarChart3,
  CheckCircle, FileText, Cpu, BookOpen,
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
import StrategyGuide from '../components/StrategyGuide'

// ─── Helpers ────────────────────────────────────────────────────────────────

function trafficStatus(avgDelay: number | null): {
  label: string; color: string; dot: string
} {
  if (avgDelay === null) return { label: 'Ready', color: 'text-slate-400', dot: 'bg-slate-500' }
  if (avgDelay < 12) return { label: 'Free Flow', color: 'text-emerald-400', dot: 'bg-emerald-400' }
  if (avgDelay < 25) return { label: 'Moderate Traffic', color: 'text-amber-400', dot: 'bg-amber-400' }
  return { label: 'Heavy Delay', color: 'text-rose-400', dot: 'bg-rose-400' }
}

function simplifyRationale(raw: string | null | undefined): string {
  if (!raw) return 'Monitoring intersection flow and evaluating traffic demands…'
  const r = raw.toLowerCase()
  if (r.includes('emergency') || r.includes('preemption')) return '🚨 Emergency vehicle priority active — clearing the corridor immediately.'
  if (r.includes('starv')) return 'A waiting direction exceeded threshold — shifting green immediately to maintain fairness.'
  if (r.includes('high demand') || r.includes('demand score')) return 'Allocating extended green to the direction with the highest vehicle backlog.'
  if (r.includes('fixed') || r.includes('pre-timed')) return 'Operating on pre-timed signal splits based on historical averages.'
  if (r.includes('predict') || r.includes('forecast')) return 'Anticipating vehicle arrival spike from upstream — preemptively adjusting split.'
  if (r.includes('yellow') || r.includes('clearance')) return 'Clearance phase active — safely clearing the junction before next green.'
  if (r.includes('all_red') || r.includes('all-red')) return 'All-red clearance active — all approaches halted momentarily for safety.'
  if (r.includes('pressure') || r.includes('max_pressure')) return 'Max-Pressure algorithm optimizing flow by maximizing discharge capacity.'
  if (r.includes('ppo') || r.includes('deep_rl') || r.includes('reinforcement')) return 'Deep RL PPO policy selected optimal phase to minimize total intersection delay.'
  if (r.includes('phase')) return 'Switching green phase to next approach group.'
  return 'Monitoring live traffic and dynamically adjusting signal timings.'
}

const MODE_OPTIONS = [
  {
    id: 'DEEP_RL' as const,
    label: 'Deep AI (PPO)',
    desc: 'Self-learning neural network policy trained with reinforcement learning to minimize delays and queues',
    accent: 'bg-purple-600 text-white',
  },
  {
    id: 'MAX_PRESSURE' as const,
    label: 'Max-Pressure',
    desc: 'Mathematical network algorithm balancing incoming and outgoing vehicle queues to maximize discharge',
    accent: 'bg-blue-600 text-white',
  },
  {
    id: 'ADAPTIVE' as const,
    label: 'Dynamic Adaptive',
    desc: 'Continuously shifts green time towards the roads with the longest queues and waiting times',
    accent: 'bg-cyan-600 text-white',
  },
  {
    id: 'ACTUATED' as const,
    label: 'Vehicle Actuated',
    desc: 'Extends green light only while approaching vehicles are actively detected waiting',
    accent: 'bg-emerald-600 text-white',
  },
  {
    id: 'FIXED_TIME' as const,
    label: 'Fixed-Time',
    desc: 'Traditional predetermined signal timer cycling on fixed green durations regardless of traffic',
    accent: 'bg-amber-600 text-white',
  },
] as const

// ─── Sub-components ──────────────────────────────────────────────────────────

function SummaryCard({
  label, value, sub, accent = 'cyan', icon,
}: {
  label: string; value: React.ReactNode; sub: string
  accent?: 'cyan' | 'amber' | 'emerald' | 'rose' | 'purple'; icon: React.ReactNode
}) {
  const ring: Record<string, string> = {
    cyan: 'border-cyan-800/60',
    amber: 'border-amber-800/50',
    emerald: 'border-emerald-800/50',
    rose: 'border-rose-800/50',
    purple: 'border-purple-800/50',
  }
  const ic: Record<string, string> = {
    cyan: 'text-cyan-400',
    amber: 'text-amber-400',
    emerald: 'text-emerald-400',
    rose: 'text-rose-400',
    purple: 'text-purple-400',
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
      Simulating Live
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
      Ready to Run
    </span>
  )
}

// ─── Advanced sub-tab ────────────────────────────────────────────────────────

type AdvancedSub = 'rl' | 'safety' | 'ml' | 'log' | 'dataset'

function AdvancedTabBar({
  sub, setSub,
}: { sub: AdvancedSub; setSub: (s: AdvancedSub) => void }) {
  const items: { id: AdvancedSub; label: string; icon: React.ReactNode }[] = [
    { id: 'rl',      label: 'AI Controller (PPO)', icon: <Cpu size={13} /> },
    { id: 'safety',  label: 'Safety Supervisor',   icon: <Shield size={13} /> },
    { id: 'ml',      label: 'AI Demand Forecast',  icon: <Brain size={13} /> },
    { id: 'log',     label: 'Signal Decision Log', icon: <FileText size={13} /> },
    { id: 'dataset', label: 'Pune Dataset Reference', icon: <Database size={13} /> },
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

type Tab = 'compare' | 'guide' | 'live' | 'video' | 'advanced'

export default function Dashboard() {
  const [running, setRunning]   = useState(false)
  const [paused, setPaused]     = useState(false)
  const [speed, setSpeed]       = useState(1000)
  const [mode, setMode]         = useState<'ADAPTIVE' | 'PREDICTION_ASSISTED' | 'ACTUATED' | 'FIXED_TIME' | 'MAX_PRESSURE' | 'DEEP_RL'>('DEEP_RL')
  const [status, setStatus]     = useState<ControllerStatus | null>(null)
  const [current, setCurrent]   = useState<TrafficCurrent | null>(null)
  const [history, setHistory]   = useState<HistoryPoint[]>([])
  const [decisionLog, setDecisionLog] = useState<DecisionLogEntry[]>([])
  const [apiError, setApiError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<Tab>('compare')
  const [advSub, setAdvSub]     = useState<AdvancedSub>('rl')
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
      setApiError(e instanceof Error ? e.message : 'Backend connection error')
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

  const handleModeSwitch = async (m: 'ADAPTIVE' | 'PREDICTION_ASSISTED' | 'ACTUATED' | 'FIXED_TIME' | 'MAX_PRESSURE' | 'DEEP_RL') => {
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

  const tabs: { id: Tab; label: string; icon: React.ReactNode; badge?: string }[] = [
    { id: 'compare',  label: 'Strategy Comparison Lab', icon: <BarChart3 size={15} />, badge: 'Primary' },
    { id: 'guide',    label: 'Strategy Guide',          icon: <BookOpen size={15} /> },
    { id: 'live',     label: 'Live Intersection',       icon: <Activity size={15} /> },
    { id: 'video',    label: 'CCTV Camera',             icon: <Video size={15} /> },
    { id: 'advanced', label: 'AI & Diagnostics',        icon: <Cpu size={15} /> },
  ]

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">

      {/* ── Header ────────────────────────────────────────────────────────── */}
      <header className="bg-slate-900 border-b border-slate-800 px-5 py-3 sticky top-0 z-30 shadow-md">
        <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-3">

          {/* Brand */}
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-cyan-600/20 border border-cyan-500/40 flex items-center justify-center text-cyan-400 shrink-0 shadow-inner">
              <Zap size={20} />
            </div>
            <div>
              <div className="text-sm font-bold text-white flex items-center gap-2">
                <span>AI2 Traffic Signal Optimization</span>
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-cyan-950 border border-cyan-600/40 text-cyan-300 font-mono">
                  Alankar Chowk, Pune
                </span>
              </div>
              <div className="text-[11px] text-slate-400 leading-tight">
                Deep Reinforcement Learning & Multi-Strategy Traffic Signal Management
              </div>
            </div>
          </div>

          {/* Error Banner */}
          {apiError && (
            <div className="flex items-center gap-2 text-rose-400 text-xs border border-rose-800/50 bg-rose-950/40 px-3 py-1.5 rounded-lg">
              <AlertTriangle size={13} /> {apiError}
            </div>
          )}

          {/* Right Status */}
          <div className="flex items-center gap-4">
            <StatusDot running={running} paused={paused} />
            <div className="h-5 w-px bg-slate-800 hidden sm:block" />
            <div className="text-right hidden sm:block">
              <div className="text-[10px] text-slate-500 uppercase tracking-wider">Sim Time</div>
              <div className="text-sm font-mono font-bold text-white">{timeLabel}</div>
            </div>
            {status?.signal_state && <SignalBadge state={status.signal_state} />}
          </div>
        </div>
      </header>

      {/* ── Main Navigation Tabs ──────────────────────────────────────────── */}
      <nav className="bg-slate-900/95 border-b border-slate-800/80 px-5 backdrop-blur-md sticky top-[61px] z-20">
        <div className="max-w-7xl mx-auto flex items-center gap-2 overflow-x-auto py-2.5">
          {tabs.map(t => (
            <button
              key={t.id}
              onClick={() => setActiveTab(t.id)}
              className={`flex items-center gap-2 px-4 py-2 text-xs font-semibold rounded-xl transition-all whitespace-nowrap ${
                activeTab === t.id
                  ? 'bg-cyan-600 text-white shadow-lg shadow-cyan-950/50 border border-cyan-400/40'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60 border border-transparent'
              }`}
            >
              {t.icon}
              <span>{t.label}</span>
              {t.badge && (
                <span className={`text-[10px] px-1.5 py-0.2 rounded-full font-bold uppercase tracking-wider ${
                  activeTab === t.id ? 'bg-white/20 text-white' : 'bg-emerald-950 border border-emerald-700/60 text-emerald-400'
                }`}>
                  {t.badge}
                </span>
              )}
            </button>
          ))}
        </div>
      </nav>

      {/* ── Main Content Area ────────────────────────────────────────────── */}
      <main className="max-w-7xl mx-auto w-full p-5 space-y-6 flex-1">

        {/* ══ 1. STRATEGY COMPARISON LAB (HERO FEATURE) ═════════════════════ */}
        {activeTab === 'compare' && (
          <div className="space-y-6">
            <StrategyComparisonLab />
          </div>
        )}

        {/* ══ 2. STRATEGY GUIDE (PLAIN ENGLISH REFERENCE) ═════════════════════ */}
        {activeTab === 'guide' && (
          <div className="space-y-6">
            <StrategyGuide />
          </div>
        )}

        {/* ══ 3. LIVE INTERSECTION (UNIFIED CONTROL & MAP) ═══════════════════ */}
        {activeTab === 'live' && (
          <div className="space-y-5">

            {/* ── Controller Strategy Selector & Playback Bar ── */}
            <div className="card space-y-4">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
                    <Zap size={13} className="text-cyan-400" />
                    <span>Active Control Strategy</span>
                  </div>
                  <span className="text-[11px] text-slate-500">Select any algorithm to take over signal control</span>
                </div>
                <div className="flex flex-wrap gap-2">
                  {MODE_OPTIONS.map(opt => (
                    <div key={opt.id} className="relative">
                      <button
                        onClick={() => handleModeSwitch(opt.id)}
                        className={`px-3.5 py-2 rounded-xl text-xs font-semibold transition-all border ${
                          mode === opt.id
                            ? `${opt.accent} border-transparent shadow-md`
                            : 'text-slate-400 bg-slate-900 border-slate-700 hover:text-white hover:border-slate-600'
                        }`}
                      >
                        {opt.label}
                      </button>
                      <button
                        onMouseEnter={() => setShowModeTooltip(opt.id)}
                        onMouseLeave={() => setShowModeTooltip(null)}
                        className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-slate-700 text-slate-400 hover:text-white flex items-center justify-center text-[10px]"
                        title="Click or hover for explanation"
                      >
                        <Info size={9} />
                      </button>
                      {showModeTooltip === opt.id && (
                        <div className="absolute top-full left-0 mt-1.5 w-60 bg-slate-800 border border-slate-700 rounded-lg p-2.5 text-[11px] text-slate-300 z-50 shadow-xl">
                          {opt.desc}
                        </div>
                      )}
                    </div>
                  ))}
                  <button
                    onClick={() => handleModeSwitch('PREDICTION_ASSISTED')}
                    className={`px-3.5 py-2 rounded-xl text-xs font-semibold transition-all border ${
                      mode === 'PREDICTION_ASSISTED'
                        ? 'bg-violet-600 text-white border-transparent shadow-md'
                        : 'text-slate-400 bg-slate-900 border-slate-700 hover:text-white hover:border-slate-600'
                    }`}
                    title="Anticipates congestion by pre-adjusting splits before queues build"
                  >
                    AI + Forecast
                  </button>
                </div>
              </div>

              {/* Playback Controls & Speed Toggle */}
              <div className="flex flex-wrap items-center justify-between gap-3 pt-3 border-t border-slate-800">
                <div className="flex items-center gap-2">
                  {!running ? (
                    <button
                      onClick={handleStart}
                      className="flex items-center gap-1.5 px-4 py-2 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-bold shadow-md transition-all"
                    >
                      <Play size={14} /> Start Simulation
                    </button>
                  ) : (
                    <button
                      onClick={handlePause}
                      className="flex items-center gap-1.5 px-4 py-2 rounded-xl bg-amber-600 hover:bg-amber-500 text-white text-xs font-bold shadow-md transition-all"
                    >
                      <Pause size={14} /> {paused ? 'Resume' : 'Pause'}
                    </button>
                  )}
                  <button
                    onClick={handleReset}
                    className="flex items-center gap-1.5 px-3 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium border border-slate-700 transition-all"
                  >
                    <RotateCcw size={13} /> Reset
                  </button>
                </div>

                {/* Speed selector */}
                <div className="flex items-center gap-2">
                  <span className="text-xs text-slate-500">Replay Speed:</span>
                  <div className="flex items-center gap-1 bg-slate-900 p-1 rounded-lg border border-slate-800">
                    {[2000, 1000, 500, 200].map(s => (
                      <button
                        key={s}
                        onClick={() => setSpeed(s)}
                        className={`px-2.5 py-1 rounded-md text-[11px] font-semibold transition-all ${
                          speed === s ? 'bg-cyan-600 text-white shadow-sm' : 'text-slate-400 hover:text-white'
                        }`}
                      >
                        {s === 2000 ? '½×' : s === 1000 ? '1×' : s === 500 ? '2×' : '5×'}
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              {/* Timeline Scrubber */}
              <div className="flex items-center gap-3 pt-2 border-t border-slate-800">
                <span className="text-[11px] text-slate-400 font-semibold whitespace-nowrap">Timeline:</span>
                <span className="text-[11px] text-slate-500 font-mono">09:00</span>
                <button
                  onClick={() => onSliderCommit(Math.max(0, curStep - 10))}
                  disabled={curStep <= 0}
                  className="px-2 py-0.5 bg-slate-800 hover:bg-slate-700 disabled:opacity-30 rounded text-slate-300 text-[11px] font-mono transition-colors"
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
                  className="px-2 py-0.5 bg-slate-800 hover:bg-slate-700 disabled:opacity-30 rounded text-slate-300 text-[11px] font-mono transition-colors"
                  title="Forward 10 seconds"
                >
                  +10s
                </button>
                <span className="text-[11px] text-slate-500 font-mono">09:30</span>
                <span className="text-cyan-400 font-mono font-bold text-xs whitespace-nowrap min-w-[55px] text-right">
                  {displayTime}
                </span>
              </div>

              {/* Plain-English Live Rationale */}
              <div className="pt-2 border-t border-slate-800">
                <div className="flex items-start gap-2.5 text-xs text-slate-300 bg-slate-900/80 border border-slate-800 rounded-xl px-3.5 py-2.5">
                  <span className="shrink-0 text-base mt-0.5">💡</span>
                  <div className="flex-1">
                    <span className="font-semibold text-white mr-1.5">Signal Decision:</span>
                    <span>{simplifyRationale(ctrl?.decision_rationale)}</span>
                  </div>
                  {ctrl?.phase_name && (
                    <span className="shrink-0 text-[11px] px-2 py-0.5 rounded bg-slate-800 border border-slate-700 font-mono text-cyan-300">
                      {ctrl.phase_name}
                    </span>
                  )}
                </div>
              </div>
            </div>

            {/* ── 4 Key Performance Cards ── */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              <SummaryCard
                label="Average Delay"
                value={avgDelay !== null ? `${avgDelay.toFixed(1)}s` : '—'}
                sub="Mean waiting time per vehicle at signal"
                accent={avgDelay === null ? 'cyan' : avgDelay < 12 ? 'emerald' : avgDelay < 25 ? 'amber' : 'rose'}
                icon={<Clock size={14} />}
              />
              <SummaryCard
                label="Vehicles Waiting"
                value={<>{busyRoads} <span className="text-sm font-normal text-slate-400">busy roads</span></>}
                sub={`${current?.total_queue_pcu?.toFixed(1) ?? '0.0'} PCU queued across approaches`}
                accent="amber"
                icon={<Activity size={14} />}
              />
              <SummaryCard
                label="Vehicles Cleared"
                value={throughput !== null ? Math.round(throughput).toLocaleString() : '—'}
                sub="Total cumulative vehicles discharged"
                accent="emerald"
                icon={<TrendingUp size={14} />}
              />
              <SummaryCard
                label="Flow Condition"
                value={
                  <span className={tStatus.color}>
                    <span className={`inline-block w-2.5 h-2.5 rounded-full mr-2 ${tStatus.dot}`} />
                    {tStatus.label}
                  </span>
                }
                sub={
                  avgDelay !== null
                    ? avgDelay < 12
                      ? 'Signals clearing vehicles smoothly without bottleneck.'
                      : avgDelay < 25
                      ? 'Moderate queue build-up. System adjusting phase splits.'
                      : 'High delay observed. Prioritizing clearance phases.'
                    : 'Click Start to begin real-time vehicle simulation.'
                }
                accent="cyan"
                icon={<CheckCircle size={14} />}
              />
            </div>

            {/* ── Main 2-Column Section: 2D Junction + Approaches & Emergency ── */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">

              {/* Left Column: 2D Live Junction Graphic */}
              <div className="lg:col-span-7 space-y-4">
                <div className="card space-y-3">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                    <div className="flex items-center gap-2">
                      <MapPin size={16} className="text-cyan-400" />
                      <span className="text-sm font-bold text-white">Live Intersection Graphic</span>
                    </div>
                    <span className="text-[11px] text-slate-500">Alankar Chowk 4-Way Layout</span>
                  </div>
                  <JunctionViz data={current} />
                </div>

                {/* Queue over time chart */}
                <div className="card space-y-3">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                    <div>
                      <div className="text-sm font-bold text-slate-200">Active Queue Progression</div>
                      <div className="text-[11px] text-slate-500">Live vehicle backlog over simulation steps</div>
                    </div>
                  </div>
                  <div className="h-48 w-full">
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
                            contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: 8, fontSize: 11 }}
                            formatter={(val: any) => [`${Number(val ?? 0).toFixed(1)} PCU`, 'Queue']}
                            labelFormatter={l => `Time: ${l}`}
                          />
                          <Line
                            type="monotone"
                            dataKey="queue_pcu"
                            stroke="#06b6d4"
                            strokeWidth={2}
                            dot={false}
                            name="Queued Vehicles"
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    ) : (
                      <div className="h-full flex flex-col items-center justify-center text-slate-500 text-xs gap-1.5">
                        <Activity size={20} className="text-slate-700" />
                        <span>Start simulation to view live queue graph</span>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Right Column: Road Status + Emergency Vehicle Priority */}
              <div className="lg:col-span-5 space-y-4">

                {/* Road-by-Road Status */}
                <div className="card space-y-3">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                    <span className="text-sm font-bold text-slate-200">Approach Details</span>
                    <span className="text-[11px] text-slate-500">Per-Road Breakdown</span>
                  </div>
                  {approaches.length > 0 ? (
                    <div className="space-y-2">
                      {approaches.map(a => (
                        <div
                          key={a.direction}
                          className="flex items-center justify-between bg-slate-900/80 border border-slate-800 rounded-xl p-3"
                        >
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2">
                              <span className="text-xs font-bold text-white">{a.name}</span>
                              {a.is_starving && (
                                <span className="text-[10px] text-amber-400 bg-amber-950/80 border border-amber-800/80 px-1.5 py-0.2 rounded font-semibold">
                                  Priority Needed
                                </span>
                              )}
                            </div>
                            <div className="text-[11px] text-slate-400 mt-1 flex items-center gap-3">
                              <span>Queue: <strong className="text-slate-200">{a.queue_pcu.toFixed(1)} PCU</strong></span>
                              <span>Wait: <strong className="text-slate-200">{a.avg_wait_sec.toFixed(0)}s</strong></span>
                              <span>Inflow: <strong className="text-slate-200">{a.arrival_rate_pcu_per_min.toFixed(0)}/m</strong></span>
                            </div>
                          </div>
                          <SignalBadge state={a.signal} />
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-xs text-slate-500 text-center py-6">
                      Click <strong>Start Simulation</strong> to stream approach metrics.
                    </div>
                  )}
                </div>

                {/* 1-Click Emergency Vehicle Priority Dispatcher */}
                <EmergencyPriorityControl onPriorityChange={fetchAll} />

                {/* Historical 10-Minute Traffic Demand */}
                <div className="card space-y-3">
                  <div className="text-sm font-bold text-slate-200">Historical Pune CCTV Profile (10-Min Blocks)</div>
                  <AggregatedTrafficCharts intervalSec={600} />
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ══ 3. CCTV CAMERA TAB ════════════════════════════════════════════ */}
        {activeTab === 'video' && (
          <div className="space-y-5">
            <TrafficVideoFeed data={current} />
            <div className="card text-sm text-slate-400 space-y-2">
              <div className="font-semibold text-slate-200 flex items-center gap-2">
                <Video size={16} className="text-cyan-400" /> About this feed
              </div>
              <p className="text-xs leading-relaxed">
                This feed replays vehicle detections from the <strong>Pune Police CCTV Dataset (Jan 2023)</strong> at Alankar Chowk. Vehicles are classified using computer vision bounding boxes (cars, motorbikes, buses, trucks) and converted into Indian Passenger Car Units (IRC:106 PCU) to drive genuine adaptive signal timings.
              </p>
            </div>
          </div>
        )}

        {/* ══ 4. AI & DIAGNOSTICS TAB ═══════════════════════════════════════ */}
        {activeTab === 'advanced' && (
          <div className="space-y-5">
            <div className="card flex flex-wrap items-center justify-between gap-3">
              <div>
                <div className="text-sm font-bold text-white">AI Controller, Safety Invariants & Diagnostics</div>
                <div className="text-xs text-slate-400 mt-0.5">
                  Manage PPO neural network training, safety supervisor rules, predictive models, and dataset schema.
                </div>
              </div>
              <AdvancedTabBar sub={advSub} setSub={setAdvSub} />
            </div>

            {advSub === 'rl'      && <RLTrainingPanel />}
            {advSub === 'safety'  && <SafetyAuditPanel />}
            {advSub === 'ml'      && <MLForecastPanel />}

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
                        <th className="py-2 px-2 text-left">Phase Group</th>
                        <th className="py-2 px-2 text-left">Plain English Reason</th>
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

// ─── Dataset Info Reference ──────────────────────────────────────────────────

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
        <div className="text-sm font-bold text-slate-200">IRC:106 PCU Standard & Data Dictionary</div>
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
