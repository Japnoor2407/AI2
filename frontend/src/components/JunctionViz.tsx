import type { TrafficCurrent } from '../types'
import { SignalBadge } from './SignalBadge'

interface JunctionVizProps {
  data: TrafficCurrent | null
}

const APPROACH_POSITIONS: Record<string, { label: string; x: number; y: number; dir: string }> = {
  North_Approach: { label: 'North', x: 50, y:  2, dir: '↓' },
  South_Approach: { label: 'South', x: 50, y: 82, dir: '↑' },
  East_Approach:  { label: 'East',  x: 82, y: 42, dir: '←' },
  West_Approach:  { label: 'West',  x:  2, y: 42, dir: '→' },
}

function ApproachBox({
  pos, app,
}: {
  pos: { label: string; x: number; y: number; dir: string }
  app?: TrafficCurrent['approaches'][string]
}) {
  const signal = app?.signal || 'RED'
  const borderColor =
    signal === 'GREEN' ? '#22c55e' :
    signal === 'YELLOW' ? '#eab308' :
    signal === 'ALL_RED' ? '#f97316' :
    '#ef4444'
  const bgColor =
    signal === 'GREEN' ? '#052e16' :
    signal === 'YELLOW' ? '#1c1400' :
    '#1a0000'

  return (
    <div
      className="absolute rounded-lg p-2 text-xs"
      style={{
        left: `${pos.x}%`,
        top: `${pos.y}%`,
        transform: 'translate(-50%, -50%)',
        border: `2px solid ${borderColor}`,
        background: bgColor,
        minWidth: 100,
      }}
    >
      <div className="font-bold text-slate-200">{pos.dir} {pos.label}</div>
      {app ? (
        <>
          <div className="text-slate-400">Queue: <span className="text-white font-mono">{app.queue_pcu.toFixed(1)} PCU</span></div>
          <div className="text-slate-400">Rate: <span className="text-white font-mono">{app.arrival_rate_pcu_per_min.toFixed(1)}/min</span></div>
          <div className="mt-1"><SignalBadge state={signal} /></div>
          {app.is_starving && (
            <div className="mt-1 text-orange-400 font-bold text-[10px]">⚠ STARVATION</div>
          )}
        </>
      ) : (
        <div className="text-slate-500 text-[10px]">No data</div>
      )}
    </div>
  )
}

export function JunctionViz({ data }: JunctionVizProps) {
  const phase = data?.controller?.active_phase_id
  const state = data?.controller?.signal_state || 'RED'
  const phaseName = data?.controller?.phase_name || '—'

  return (
    <div className="relative w-full" style={{ paddingTop: '75%' }}>
      {/* Outer container */}
      <div className="absolute inset-0 rounded-xl" style={{ background: '#080f1e', border: '1px solid #1e293b' }}>
        {/* Road lines */}
        <div className="absolute" style={{ left: '42%', top: '15%', width: '16%', height: '70%', background: '#1e293b', borderRadius: 4 }} />
        <div className="absolute" style={{ left: '15%', top: '42%', width: '70%', height: '16%', background: '#1e293b', borderRadius: 4 }} />
        {/* Intersection box */}
        <div
          className="absolute rounded-lg flex items-center justify-center text-[10px] text-center font-bold"
          style={{
            left: '40%', top: '40%', width: '20%', height: '20%',
            background: '#0d1428',
            border: `3px solid ${state === 'GREEN' ? '#22c55e' : state === 'YELLOW' ? '#eab308' : '#ef4444'}`,
            color: state === 'GREEN' ? '#22c55e' : state === 'YELLOW' ? '#eab308' : '#ef4444',
          }}
        >
          <div>
            <div className="text-lg">P{phase}</div>
            <div className="opacity-60" style={{ fontSize: 9 }}>{state}</div>
          </div>
        </div>
        {/* Approach boxes */}
        {Object.entries(APPROACH_POSITIONS).map(([id, pos]) => (
          <ApproachBox key={id} pos={pos} app={data?.approaches?.[id]} />
        ))}
        {/* Phase label */}
        <div className="absolute bottom-2 left-0 right-0 text-center text-[10px] text-slate-500">
          {phaseName}
        </div>
        {/* Data source label */}
        <div className="absolute top-2 right-2 text-[9px] text-slate-600 font-mono">
          REPLAY: Pune CCTV Dataset (Jan 2023)
        </div>
      </div>
    </div>
  )
}

