import { useState, useEffect } from 'react'
import { api } from '../services/api'
import type { EmergencyStatus } from '../types'

interface Props {
  onPriorityChange?: () => void
}

export default function EmergencyPriorityControl({ onPriorityChange }: Props) {
  const [status, setStatus] = useState<EmergencyStatus | null>(null)
  const [vehicleType, setVehicleType] = useState<'AMBULANCE' | 'FIRE_TRUCK' | 'POLICE'>('AMBULANCE')
  const [approachId, setApproachId] = useState<string>('N')
  const [loading, setLoading] = useState<boolean>(false)

  const fetchStatus = async () => {
    try {
      const res = await api.emergencyStatus()
      if (res) {
        setStatus(res)
      }
    } catch (err) {
      console.warn('Failed to fetch emergency status:', err)
    }
  }

  useEffect(() => {
    fetchStatus()
    const timer = setInterval(fetchStatus, 1500)
    return () => clearInterval(timer)
  }, [])

  const handleRequestPriority = async () => {
    setLoading(true)
    try {
      await api.emergencyRequest(vehicleType, approachId)
      await fetchStatus()
      if (onPriorityChange) onPriorityChange()
    } catch (err) {
      console.error('Failed to trigger emergency priority:', err)
    } finally {
      setLoading(false)
    }
  }

  const handleClearPriority = async () => {
    setLoading(true)
    try {
      await api.emergencyClear()
      await fetchStatus()
      if (onPriorityChange) onPriorityChange()
    } catch (err) {
      console.error('Failed to clear emergency priority:', err)
    } finally {
      setLoading(false)
    }
  }

  const isActive = status?.is_active || false
  const activeEvent = status?.active_event

  return (
    <div className={`rounded-xl border p-4 transition-all ${
      isActive
        ? 'bg-rose-950/40 border-rose-600/80 shadow-lg shadow-rose-900/30 animate-pulse'
        : 'bg-slate-900 border-slate-800'
    }`}>
      <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-3">
        <div className="flex items-center gap-2">
          <span className={`text-base ${isActive ? 'animate-bounce' : ''}`}>
            {isActive ? '🚨' : '🚑'}
          </span>
          <div>
            <h4 className="text-sm font-bold text-white flex items-center gap-2">
              Emergency Vehicle Preemption
              {isActive && (
                <span className="px-2 py-0.5 rounded text-[10px] font-extrabold uppercase bg-rose-600 text-white tracking-wider">
                  ACTIVE PRIORITY
                </span>
              )}
            </h4>
            <p className="text-xs text-slate-400">
              {isActive
                ? `Prioritizing ${activeEvent?.vehicle_type} on approach ${activeEvent?.approach_id}`
                : 'Grant zero-delay green corridor for emergency responders'}
            </p>
          </div>
        </div>

        {isActive ? (
          <button
            type="button"
            disabled={loading}
            onClick={handleClearPriority}
            className="px-3 py-1.5 rounded-lg text-xs font-bold bg-slate-800 hover:bg-slate-700 text-rose-300 border border-rose-800/60 transition-colors"
          >
            Clear Priority Corridor
          </button>
        ) : null}
      </div>

      {isActive ? (
        <div className="bg-rose-900/20 border border-rose-800/40 rounded-lg p-3 text-xs text-rose-200 flex justify-between items-center">
          <div>
            <span className="font-semibold text-white">Target Phase {activeEvent?.target_phase_id}:</span> Green signal held safely until vehicle passes. All conflicting approaches held at RED.
          </div>
          <div className="text-right text-[11px] text-rose-300">
            Requested at step {activeEvent?.requested_at_step}
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 items-center">
          <div>
            <label className="text-[11px] font-medium text-slate-400 uppercase tracking-wider block mb-1">
              Vehicle Type
            </label>
            <select
              value={vehicleType}
              onChange={e => setVehicleType(e.target.value as any)}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-rose-500"
            >
              <option value="AMBULANCE">🚑 Ambulance</option>
              <option value="FIRE_TRUCK">🚒 Fire Engine</option>
              <option value="POLICE">🚓 Police Patrol</option>
            </select>
          </div>

          <div>
            <label className="text-[11px] font-medium text-slate-400 uppercase tracking-wider block mb-1">
              Approaching Road
            </label>
            <select
              value={approachId}
              onChange={e => setApproachId(e.target.value)}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-rose-500"
            >
              <option value="N">North Approach (From Jehangir)</option>
              <option value="S">South Approach (From Station)</option>
              <option value="E">East Approach (From Ruby Hall)</option>
              <option value="W">West Approach (From Maldhakka)</option>
            </select>
          </div>

          <div className="sm:self-end">
            <button
              type="button"
              disabled={loading}
              onClick={handleRequestPriority}
              className="w-full py-2 px-3 rounded-lg text-xs font-bold text-white bg-rose-600 hover:bg-rose-500 active:scale-95 shadow-lg shadow-rose-950/40 flex items-center justify-center gap-1.5 transition-all"
            >
              <span>🚨</span> Dispatch Priority Green
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

