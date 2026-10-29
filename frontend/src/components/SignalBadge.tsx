interface SignalBadgeProps {
  state: string
}

export function SignalBadge({ state }: SignalBadgeProps) {
  const configs: Record<string, { label: string; dot: string; cls: string }> = {
    GREEN: {
      label: 'GO',
      dot: 'bg-emerald-400',
      cls: 'bg-emerald-950/80 text-emerald-300 border-emerald-600/70',
    },
    YELLOW: {
      label: 'CLEAR',
      dot: 'bg-amber-400',
      cls: 'bg-amber-950/80 text-amber-300 border-amber-600/70',
    },
    RED: {
      label: 'STOP',
      dot: 'bg-rose-400',
      cls: 'bg-rose-950/80 text-rose-300 border-rose-600/70',
    },
    ALL_RED: {
      label: 'ALL RED',
      dot: 'bg-rose-400',
      cls: 'bg-rose-950/80 text-rose-300 border-rose-600/70',
    },
  }

  const cfg = configs[state] || {
    label: state,
    dot: 'bg-slate-400',
    cls: 'bg-slate-800 text-slate-300 border-slate-700',
  }

  return (
    <span
      className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded-md text-[10px] font-bold border ${cfg.cls} tracking-wider`}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${cfg.dot}`} />
      <span>{cfg.label}</span>
    </span>
  )
}
