import { SignalBadge } from './SignalBadge'
import type { TrafficCurrent } from '../types'

interface JunctionVizProps {
  data: TrafficCurrent | null
}

const APPROACH_CONFIG = [
  {
    id: 'North_Approach',
    name: 'North · Station Road',
    dirSymbol: '↓',
    positionClasses: 'top-2 left-1/2 -translate-x-1/2',
    directionLabel: 'Heading South',
  },
  {
    id: 'South_Approach',
    name: 'South · Camp Corridor',
    dirSymbol: '↑',
    positionClasses: 'bottom-2 left-1/2 -translate-x-1/2',
    directionLabel: 'Heading North',
  },
  {
    id: 'East_Approach',
    name: 'East · Sassoon Arterial',
    dirSymbol: '←',
    positionClasses: 'right-2 top-1/2 -translate-y-1/2',
    directionLabel: 'Heading West',
  },
  {
    id: 'West_Approach',
    name: 'West · Collectorate Rd',
    dirSymbol: '→',
    positionClasses: 'left-2 top-1/2 -translate-y-1/2',
    directionLabel: 'Heading East',
  },
]

export function JunctionViz({ data }: JunctionVizProps) {
  const ctrl = data?.controller
  const activePhase = ctrl?.active_phase_id ?? 1
  const signalState = ctrl?.signal_state || 'RED'

  // Determine current active road name in plain English
  let activeRoadName = 'All Roads Stopped'
  if (signalState === 'GREEN' || signalState === 'YELLOW') {
    if (activePhase === 1) activeRoadName = 'North (Station Road)'
    else if (activePhase === 2) activeRoadName = 'East (Sassoon Arterial)'
    else if (activePhase === 3) activeRoadName = 'South (Camp Corridor)'
    else if (activePhase === 4) activeRoadName = 'West (Collectorate Rd)'
  } else if (signalState === 'ALL_RED') {
    activeRoadName = 'Safety Clearance (All Red)'
  }

  return (
    <div className="relative w-full aspect-[4/3] max-h-[520px] rounded-2xl bg-slate-950 border border-slate-800/80 overflow-hidden shadow-xl select-none">
      {/* Background Intersection Roads */}
      <div className="absolute inset-0 flex items-center justify-center">
        {/* Vertical Road (North-South) */}
        <div className="absolute w-28 sm:w-36 h-full bg-slate-900 border-x border-slate-800">
          {/* Dashed Center Lane Marking */}
          <div className="absolute top-0 bottom-0 left-1/2 -translate-x-1/2 w-0.5 border-l-2 border-dashed border-slate-700/60" />
          {/* Crosswalk markings North */}
          <div className="absolute top-28 left-0 right-0 h-4 flex justify-between px-2 opacity-30">
            {[...Array(6)].map((_, i) => (
              <div key={i} className="w-2.5 h-full bg-white rounded-xs" />
            ))}
          </div>
          {/* Crosswalk markings South */}
          <div className="absolute bottom-28 left-0 right-0 h-4 flex justify-between px-2 opacity-30">
            {[...Array(6)].map((_, i) => (
              <div key={i} className="w-2.5 h-full bg-white rounded-xs" />
            ))}
          </div>
        </div>

        {/* Horizontal Road (East-West) */}
        <div className="absolute h-28 sm:h-36 w-full bg-slate-900 border-y border-slate-800">
          {/* Dashed Center Lane Marking */}
          <div className="absolute left-0 right-0 top-1/2 -translate-y-1/2 h-0.5 border-t-2 border-dashed border-slate-700/60" />
          {/* Crosswalk markings West */}
          <div className="absolute left-28 top-0 bottom-0 w-4 flex flex-col justify-between py-2 opacity-30">
            {[...Array(6)].map((_, i) => (
              <div key={i} className="h-2.5 w-full bg-white rounded-xs" />
            ))}
          </div>
          {/* Crosswalk markings East */}
          <div className="absolute right-28 top-0 bottom-0 w-4 flex flex-col justify-between py-2 opacity-30">
            {[...Array(6)].map((_, i) => (
              <div key={i} className="h-2.5 w-full bg-white rounded-xs" />
            ))}
          </div>
        </div>

        {/* Center Intersection Hub Box */}
        <div
          className={`z-10 rounded-2xl px-3 py-2 sm:px-5 sm:py-3 text-center border-2 shadow-2xl transition-all duration-300 ${
            signalState === 'GREEN'
              ? 'bg-emerald-950/90 border-emerald-500/80 shadow-emerald-950/80 text-emerald-300'
              : signalState === 'YELLOW'
              ? 'bg-amber-950/90 border-amber-500/80 shadow-amber-950/80 text-amber-300'
              : 'bg-rose-950/90 border-rose-500/80 shadow-rose-950/80 text-rose-300'
          }`}
        >
          <div className="text-[10px] sm:text-xs font-semibold uppercase tracking-wider opacity-80">
            Active Signal
          </div>
          <div className="text-sm sm:text-base font-bold truncate max-w-[150px] sm:max-w-[200px]">
            {activeRoadName}
          </div>
          <div className="flex items-center justify-center gap-1.5 mt-1">
            <span
              className={`w-2 h-2 rounded-full ${
                signalState === 'GREEN'
                  ? 'bg-emerald-400 animate-pulse'
                  : signalState === 'YELLOW'
                  ? 'bg-amber-400 animate-pulse'
                  : 'bg-rose-400'
              }`}
            />
            <span className="text-[11px] font-mono font-bold">
              {signalState === 'GREEN'
                ? `${ctrl?.time_remaining_in_state ?? 0}s green remaining`
                : signalState === 'YELLOW'
                ? `${ctrl?.time_remaining_in_state ?? 0}s yellow clearance`
                : 'All signals red'}
            </span>
          </div>
        </div>
      </div>

      {/* 4 Road Approach Badges */}
      {APPROACH_CONFIG.map((cfg) => {
        const app = data?.approaches?.[cfg.id]
        const sig = app?.signal || 'RED'
        const isGreen = sig === 'GREEN'
        const isYellow = sig === 'YELLOW'
        const vehCount = app ? Math.max(0, Math.round(app.queue_pcu)) : 0
        const waitTime = app ? Math.round(app.avg_wait_sec) : 0

        return (
          <div
            key={cfg.id}
            className={`absolute ${cfg.positionClasses} z-20 w-44 sm:w-52 p-2 sm:p-2.5 rounded-xl border backdrop-blur-md transition-all duration-200 shadow-lg ${
              isGreen
                ? 'bg-emerald-950/80 border-emerald-500/60 shadow-emerald-950/50'
                : isYellow
                ? 'bg-amber-950/80 border-amber-500/60 shadow-amber-950/50'
                : 'bg-slate-900/90 border-slate-800 shadow-slate-950/50'
            }`}
          >
            <div className="flex items-center justify-between gap-1 mb-1">
              <span className="text-xs font-bold text-slate-100 flex items-center gap-1 truncate">
                <span className="text-cyan-400 font-bold">{cfg.dirSymbol}</span>
                {cfg.name}
              </span>
              <SignalBadge state={sig} />
            </div>

            <div className="grid grid-cols-2 gap-1 text-[11px] text-slate-300">
              <div className="bg-slate-950/60 rounded px-1.5 py-0.5 border border-slate-800/60">
                <span className="text-slate-500 text-[10px] block">Waiting</span>
                <span className="font-semibold text-white">
                  {vehCount === 0 ? 'Clear (0)' : `${vehCount} vehicle${vehCount > 1 ? 's' : ''}`}
                </span>
              </div>
              <div className="bg-slate-950/60 rounded px-1.5 py-0.5 border border-slate-800/60">
                <span className="text-slate-500 text-[10px] block">Wait Time</span>
                <span className="font-semibold text-white">
                  {waitTime === 0 ? '0s' : `${waitTime}s`}
                </span>
              </div>
            </div>

            {app?.is_starving && (
              <div className="mt-1 px-1.5 py-0.5 bg-amber-500/20 border border-amber-500/40 rounded text-amber-300 text-[10px] font-bold flex items-center gap-1">
                <span>⚠ Waiting too long — priority requested</span>
              </div>
            )}
          </div>
        )
      })}

      {/* Subtle Legend Bar at Bottom */}
      <div className="absolute bottom-2 left-3 right-3 sm:left-4 sm:right-4 flex items-center justify-between text-[10px] text-slate-500 bg-slate-950/80 px-2.5 py-1 rounded-lg border border-slate-800/60 z-20">
        <span className="font-medium text-slate-400">
          Location: Alankar Chowk, Pune
        </span>
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-emerald-500" /> Green = Proceed
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-amber-500" /> Yellow = Prepare to stop
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-rose-500" /> Red = Stop
          </span>
        </div>
      </div>
    </div>
  )
}
