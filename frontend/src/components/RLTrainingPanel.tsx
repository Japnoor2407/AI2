/**
 * RLTrainingPanel — displays PPO training status and controls.
 *
 * Shows:
 *  - Current model availability
 *  - Training progress (when running)
 *  - Training metadata (timesteps, reward, duration)
 *  - Start training / refresh buttons
 */

import { useEffect, useState, useCallback } from 'react'
import { Brain, Play, RefreshCw, CheckCircle, AlertTriangle, Clock } from 'lucide-react'
import { api } from '../services/api'
import type { RLStatus, RLTrainingStatus } from '../types'

interface Props {
  onModelReady?: () => void
}

export function RLTrainingPanel({ onModelReady }: Props) {
  const [status, setStatus] = useState<RLStatus | null>(null)
  const [trainingStatus, setTrainingStatus] = useState<RLTrainingStatus | null>(null)
  const [simulator, setSimulator] = useState<'auto' | 'junction_sim' | 'sumo'>('auto')
  const [loading, setLoading] = useState(false)
  const [startingTrain, setStartingTrain] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [pollingInterval, setPollingInterval] = useState<ReturnType<typeof setInterval> | null>(null)

  const fetchStatus = useCallback(async () => {
    setLoading(true)
    try {
      const [s, ts] = await Promise.all([api.rlStatus(), api.rlTrainingStatus()])
      setStatus(s)
      setTrainingStatus(ts)
      if (s.model_available && onModelReady) onModelReady()
      setError(null)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to fetch RL status')
    } finally {
      setLoading(false)
    }
  }, [onModelReady])

  useEffect(() => {
    fetchStatus()
  }, [fetchStatus])

  // Poll while training is running
  useEffect(() => {
    if (trainingStatus?.is_running) {
      const id = setInterval(fetchStatus, 3000)
      setPollingInterval(id)
      return () => clearInterval(id)
    } else {
      if (pollingInterval) {
        clearInterval(pollingInterval)
        setPollingInterval(null)
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trainingStatus?.is_running])

  const handleStartTraining = async () => {
    setStartingTrain(true)
    setError(null)
    try {
      await api.rlTrain({ total_timesteps: 50000, seed: 42, simulator })
      await fetchStatus()
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to start training')
    } finally {
      setStartingTrain(false)
    }
  }

  const modelAvailable = status?.model_available ?? false
  const isRunning = trainingStatus?.is_running ?? false
  const progress = trainingStatus?.progress_percent ?? 0
  const trainingState = trainingStatus?.status ?? 'idle'
  const meta = status?.metadata ?? {}
  const sumoDetected = status?.sumo_available ?? false

  return (
    <div className="card space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Brain size={18} className="text-purple-400" />
          <span className="font-bold text-white text-sm">AI (PPO) Controller</span>
          {modelAvailable && (
            <span className="text-[10px] bg-purple-900/60 text-purple-300 border border-purple-700/40 px-2 py-0.5 rounded-full">
              Model Ready
            </span>
          )}
        </div>
        <button
          onClick={fetchStatus}
          disabled={loading}
          className="text-slate-400 hover:text-white transition-colors"
          title="Refresh status"
        >
          <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
        </button>
      </div>

      {/* Error */}
      {error && (
        <div className="flex items-center gap-2 text-xs text-rose-400 bg-rose-950/40 border border-rose-800/40 rounded-lg px-3 py-2">
          <AlertTriangle size={13} />
          {error}
        </div>
      )}

      {/* Model status */}
      <div className="grid grid-cols-2 gap-2">
        <div className="bg-slate-900/60 rounded-lg p-3 border border-slate-800">
          <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">Model</div>
          <div className={`text-sm font-semibold flex items-center gap-1 ${modelAvailable ? 'text-emerald-400' : 'text-slate-400'}`}>
            {modelAvailable ? <CheckCircle size={12} /> : <Clock size={12} />}
            {modelAvailable ? 'Trained' : 'Not trained'}
          </div>
        </div>
        <div className="bg-slate-900/60 rounded-lg p-3 border border-slate-800">
          <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">Status</div>
          <div className={`text-sm font-semibold ${
            trainingState === 'running' ? 'text-amber-400'
            : trainingState === 'done' ? 'text-emerald-400'
            : trainingState === 'error' ? 'text-rose-400'
            : 'text-slate-400'
          }`}>
            {trainingState === 'running' ? 'Training…' : trainingState.charAt(0).toUpperCase() + trainingState.slice(1)}
          </div>
        </div>
      </div>

      {/* Training progress bar */}
      {isRunning && (
        <div className="space-y-1.5">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span>Training progress</span>
            <span className="font-mono">{progress.toFixed(1)}%</span>
          </div>
          <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-purple-500 rounded-full transition-all duration-500"
              style={{ width: `${progress}%` }}
            />
          </div>
          {trainingStatus?.elapsed_sec != null && (
            <div className="text-[11px] text-slate-500 font-mono">
              {Math.floor(trainingStatus.elapsed_sec / 60)}m {Math.floor(trainingStatus.elapsed_sec % 60)}s elapsed
            </div>
          )}
        </div>
      )}

      {/* Training metadata */}
      {modelAvailable && meta.total_timesteps && (
        <div className="grid grid-cols-3 gap-2 text-xs">
          {[
            { label: 'Timesteps', val: Number(meta.total_timesteps).toLocaleString() },
            { label: 'Avg Reward', val: meta.mean_ep_reward_last20 != null ? Number(meta.mean_ep_reward_last20).toFixed(3) : '—' },
            { label: 'Train Time', val: meta.training_time_sec != null ? `${Math.round(Number(meta.training_time_sec) / 60)}m` : '—' },
          ].map(({ label, val }) => (
            <div key={label} className="bg-slate-900/60 rounded-lg p-2 border border-slate-800 text-center">
              <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-0.5">{label}</div>
              <div className="text-white font-mono font-semibold">{val}</div>
            </div>
          ))}
        </div>
      )}

      {/* Simulator configuration */}
      <div className="bg-slate-900/60 rounded-lg p-3 border border-slate-800 space-y-2">
        <div className="flex items-center justify-between text-xs">
          <span className="text-slate-400 font-semibold">Simulation Engine</span>
          <span className={`text-[10px] px-2 py-0.5 rounded font-mono ${
            sumoDetected
              ? 'bg-emerald-950 text-emerald-300 border border-emerald-800'
              : 'bg-slate-800 text-slate-300 border border-slate-700'
          }`}>
            {sumoDetected ? 'SUMO Microscopic Active' : 'Python Replay Engine (Standalone)'}
          </span>
        </div>
        <div className="grid grid-cols-3 gap-2 text-xs">
          {[
            { id: 'auto' as const, label: 'Auto (Best)' },
            { id: 'junction_sim' as const, label: 'Python Engine' },
            { id: 'sumo' as const, label: 'SUMO TraCI' },
          ].map(opt => (
            <button
              key={opt.id}
              onClick={() => setSimulator(opt.id)}
              disabled={isRunning || startingTrain}
              className={`py-1.5 px-2 rounded border text-center transition-all ${
                simulator === opt.id
                  ? 'bg-purple-900/50 border-purple-500 text-white font-bold'
                  : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      {/* Actions */}
      <div className="flex gap-2">
        <button
          onClick={handleStartTraining}
          disabled={isRunning || startingTrain}
          className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold bg-purple-700 hover:bg-purple-600 text-white transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <Play size={13} />
          {isRunning ? 'Training…' : modelAvailable ? 'Retrain' : 'Train PPO (50k steps)'}
        </button>
      </div>

      {/* Explanation */}
      <div className="text-[11px] text-slate-500 leading-relaxed border-t border-slate-800 pt-3">
        PPO (Proximal Policy Optimization) is trained by running thousands of simulated episodes.
        After training, select <span className="text-purple-300 font-semibold">AI (PPO)</span> as the controller mode to use it live.
        Training takes 5–20 minutes on CPU.
      </div>
    </div>
  )
}

