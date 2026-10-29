import { useState, useEffect } from 'react'
import { ShieldCheck, ShieldAlert, AlertTriangle, RefreshCw, CheckCircle, Info } from 'lucide-react'
import { api } from '../services/api'
import type { SafetyResponse } from '../types'

export function SafetyAuditPanel() {
  const [data, setData] = useState<SafetyResponse | null>(null)
  const [loading, setLoading] = useState(false)

  const fetchSafetyData = async () => {
    setLoading(true)
    try {
      const res = await api.safetyEvents(200)
      setData(res)
    } catch (e) {
      console.error(e)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchSafetyData()
    const timer = setInterval(fetchSafetyData, 4000)
    return () => clearInterval(timer)
  }, [])

  return (
    <div className="space-y-6">
      {/* Header and Stats */}
      <div className="card">
        <div className="flex flex-wrap items-center justify-between gap-4 mb-4">
          <div>
            <h2 className="text-base font-bold text-slate-100 flex items-center gap-2">
              <ShieldCheck className="w-5 h-5 text-emerald-400" />
              <span>Signal Safety Supervisor & Audit Boundary</span>
              <span className="badge badge-green text-[10px]">ISO 26262 Invariant Layer</span>
            </h2>
            <p className="text-xs text-slate-400 mt-1">
              Independent safety supervisory layer operating between optimization decisions and signal actuators to enforce clearance intervals, bounds, and conflict zero-tolerance.
            </p>
          </div>

          <button
            onClick={fetchSafetyData}
            disabled={loading}
            className="btn btn-secondary text-xs flex items-center gap-1.5"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
            Refresh Log
          </button>
        </div>

        {/* Safety Invariants Checklist */}
        <div className="grid grid-cols-1 sm:grid-cols-4 gap-3 pt-4 border-t border-slate-800 text-xs">
          <div className="bg-slate-900/80 border border-slate-800 rounded p-2.5 flex items-start gap-2">
            <CheckCircle className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
            <div>
              <div className="font-semibold text-slate-200">Zero Conflict Invariant</div>
              <div className="text-[10px] text-slate-400">Strict mutual exclusion between orthogonal approaches (NS vs EW)</div>
            </div>
          </div>

          <div className="bg-slate-900/80 border border-slate-800 rounded p-2.5 flex items-start gap-2">
            <CheckCircle className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
            <div>
              <div className="font-semibold text-slate-200">Clearance Protection</div>
              <div className="text-[10px] text-slate-400">Yellow (3s) and All-Red (2s) cannot be truncated or skipped</div>
            </div>
          </div>

          <div className="bg-slate-900/80 border border-slate-800 rounded p-2.5 flex items-start gap-2">
            <CheckCircle className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
            <div>
              <div className="font-semibold text-slate-200">Min/Max Green Enforced</div>
              <div className="text-[10px] text-slate-400">Green duration guaranteed within [10s, 55s] limits</div>
            </div>
          </div>

          <div className="bg-slate-900/80 border border-slate-800 rounded p-2.5 flex items-start gap-2">
            <ShieldAlert className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
            <div>
              <div className="font-semibold text-slate-200">Total Interventions</div>
              <div className="text-lg font-bold font-mono text-amber-400">
                {data?.total_violations_prevented ?? 0}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Safety Audit Events Table */}
      <div className="card">
        <div className="flex items-center justify-between mb-3">
          <div className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-amber-400" />
            <span>Safety Audit Event Log ({data?.events?.length ?? 0} events)</span>
          </div>
        </div>

        {data && data.events && data.events.length > 0 ? (
          <div className="overflow-x-auto max-h-96">
            <table className="w-full text-xs text-left">
              <thead className="sticky top-0 bg-slate-900 border-b border-slate-800 text-slate-400">
                <tr>
                  <th className="py-2 px-3">Step</th>
                  <th className="py-2 px-3">Time</th>
                  <th className="py-2 px-3">Junction</th>
                  <th className="py-2 px-3">Requested Phase</th>
                  <th className="py-2 px-3">Approved Phase</th>
                  <th className="py-2 px-3">Constraints Active</th>
                  <th className="py-2 px-3">Supervisor Intervention Reason</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono">
                {data.events.map((evt, idx) => (
                  <tr key={idx} className="hover:bg-slate-850">
                    <td className="py-2 px-3 text-slate-400">#{evt.step}</td>
                    <td className="py-2 px-3 text-cyan-400">{evt.timestamp_str}</td>
                    <td className="py-2 px-3 text-slate-300">{evt.intersection_id}</td>
                    <td className="py-2 px-3 text-rose-400">Phase {evt.requested_phase_id}</td>
                    <td className="py-2 px-3 text-emerald-400 font-bold">Phase {evt.approved_phase_id}</td>
                    <td className="py-2 px-3 font-sans text-[11px]">
                      <div className="flex flex-wrap gap-1">
                        {evt.active_constraints.map((c, i) => (
                          <span key={i} className="px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 text-[10px]">
                            {c}
                          </span>
                        ))}
                      </div>
                    </td>
                    <td className="py-2 px-3 font-sans text-amber-300 text-xs">
                      {evt.override_reason}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-center py-10 text-slate-500 text-xs flex flex-col items-center gap-2">
            <Info className="w-5 h-5 text-slate-600" />
            <span>Zero safety violations logged during this active session. The controller operated within compliant constraints.</span>
          </div>
        )}
      </div>
    </div>
  )
}

