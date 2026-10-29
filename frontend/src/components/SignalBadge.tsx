interface SignalBadgeProps {
  state: string
}

export function SignalBadge({ state }: SignalBadgeProps) {
  const configs: Record<string, { label: string; cls: string }> = {
    GREEN:   { label: 'GREEN',    cls: 'bg-green-900/40 text-green-400 border border-green-600' },
    YELLOW:  { label: 'YELLOW',   cls: 'bg-yellow-900/40 text-yellow-400 border border-yellow-600' },
    RED:     { label: 'RED',      cls: 'bg-red-900/40 text-red-400 border border-red-700' },
    ALL_RED: { label: 'ALL-RED',  cls: 'bg-orange-900/40 text-orange-400 border border-orange-600' },
  }
  const cfg = configs[state] || { label: state, cls: 'bg-slate-700 text-slate-300 border border-slate-600' }
  return (
    <span className={`inline-block px-2 py-0.5 rounded text-xs font-bold ${cfg.cls}`}>
      {cfg.label}
    </span>
  )
}

