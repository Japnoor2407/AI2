import { useEffect, useState } from 'react'
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer,
} from 'recharts'
import { BarChart3, Clock, TrendingUp, Layers, RefreshCw } from 'lucide-react'
import { api } from '../services/api'
import type { AggregatedInterval, AggregatedResponse } from '../types'

interface AggregatedTrafficChartsProps {
  intervalSec?: number
}

export function AggregatedTrafficCharts({ intervalSec = 600 }: AggregatedTrafficChartsProps) {
  const [data, setData] = useState<AggregatedResponse | null>(null)
  const [loading, setLoading] = useState<boolean>(true)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  const loadData = async () => {
    setLoading(true)
    try {
      const res = await api.trafficAggregated(intervalSec)
      setData(res)
      setErrorMessage(null)
    } catch (e: unknown) {
      setErrorMessage(e instanceof Error ? e.message : 'Failed to load 10-minute traffic data')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadData()
  }, [intervalSec])

  const intervals = data?.intervals || []
  const simIntervals = data?.simulation_intervals || []

  // Find peak 10-min interval
  let peakInterval: AggregatedInterval | null = null
  let totalVehiclesOverall = 0
  let totalPcuOverall = 0
  for (const it of intervals) {
    totalVehiclesOverall += it.total_vehicles
    totalPcuOverall += it.pcu_arrivals
    if (!peakInterval || it.pcu_arrivals > peakInterval.pcu_arrivals) {
      peakInterval = it
    }
  }

  return (
    <div className="space-y-6">
      {errorMessage && (
        <div className="p-3 bg-rose-950/50 border border-rose-800 text-rose-300 text-xs rounded-lg">
          {errorMessage}
        </div>
      )}
      {/* Header & KPI Summary */}
      <div className="card space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-3">
          <div className="flex items-center gap-2">
            <BarChart3 className="text-cyan-400" size={18} />
            <span className="font-bold text-white text-base">Traffic Flow Analysis (10-Minute Units)</span>
            <span className="text-xs bg-cyan-900/40 text-cyan-400 border border-cyan-700 px-2 py-0.5 rounded font-mono font-semibold">
              Δt = 10 min (600s)
            </span>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={loadData}
              disabled={loading}
              className="flex items-center gap-1.5 px-3 py-1 rounded bg-slate-800 hover:bg-slate-700 text-xs text-slate-300 border border-slate-700"
            >
              <RefreshCw size={12} className={loading ? 'animate-spin' : ''} />
              Refresh 10-Min Bins
            </button>
          </div>
        </div>

        {/* 10-Minute Statistics Cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div className="bg-slate-900/60 border border-slate-800 rounded-lg p-3">
            <div className="flex items-center gap-1.5 text-slate-400 text-xs mb-1">
              <Clock size={13} className="text-cyan-400" />
              <span>10-Min Intervals</span>
            </div>
            <div className="text-xl font-bold font-mono text-white">{intervals.length} blocks</div>
            <div className="text-[11px] text-slate-500">
              {intervals[0]?.start_time.slice(0, 5)} to {intervals[intervals.length - 1]?.end_time.slice(0, 5)}
            </div>
          </div>

          <div className="bg-slate-900/60 border border-slate-800 rounded-lg p-3">
            <div className="flex items-center gap-1.5 text-slate-400 text-xs mb-1">
              <TrendingUp size={13} className="text-amber-400" />
              <span>Peak 10-Min Window</span>
            </div>
            <div className="text-xl font-bold font-mono text-amber-400">
              {peakInterval ? peakInterval.label : '—'}
            </div>
            <div className="text-[11px] text-slate-400">
              {peakInterval ? `${peakInterval.total_vehicles} veh (${peakInterval.pcu_arrivals} PCU)` : '—'}
            </div>
          </div>

          <div className="bg-slate-900/60 border border-slate-800 rounded-lg p-3">
            <div className="flex items-center gap-1.5 text-slate-400 text-xs mb-1">
              <Layers size={13} className="text-emerald-400" />
              <span>Total Session Volume</span>
            </div>
            <div className="text-xl font-bold font-mono text-emerald-400">{totalVehiclesOverall} veh</div>
            <div className="text-[11px] text-slate-400 font-mono">{totalPcuOverall.toFixed(1)} PCU total</div>
          </div>

          <div className="bg-slate-900/60 border border-slate-800 rounded-lg p-3">
            <div className="flex items-center gap-1.5 text-slate-400 text-xs mb-1">
              <TrendingUp size={13} className="text-rose-400" />
              <span>Peak Hourly Flow</span>
            </div>
            <div className="text-xl font-bold font-mono text-rose-400">
              {peakInterval ? `${peakInterval.hourly_flow_pcu} PCU/h` : '—'}
            </div>
            <div className="text-[11px] text-slate-500">Scaled from peak 10 min</div>
          </div>
        </div>
      </div>

      {/* Chart 1: 10-Minute Stacked Vehicle Class Flow */}
      <div className="card space-y-3">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm font-bold text-slate-200">
              Vehicle Class Composition per 10-Minute Interval
            </div>
            <div className="text-xs text-slate-400">
              Stacked vehicle arrivals by type with total PCU demand curve
            </div>
          </div>
          <div className="flex items-center gap-3 text-xs font-mono">
            <span className="flex items-center gap-1 text-cyan-400">■ Car (1.0 PCU)</span>
            <span className="flex items-center gap-1 text-amber-400">■ Bike (0.5 PCU)</span>
            <span className="flex items-center gap-1 text-rose-400">■ Bus (2.5 PCU)</span>
            <span className="flex items-center gap-1 text-slate-400">■ Truck (2.5 PCU)</span>
          </div>
        </div>

        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={intervals} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false} />
              <XAxis
                dataKey="label"
                tick={{ fill: '#94a3b8', fontSize: 11 }}
                angle={-25}
                textAnchor="end"
                interval={0}
              />
              <YAxis
                tick={{ fill: '#94a3b8', fontSize: 11 }}
                label={{ value: 'Vehicles / 10 min', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 11 }}
              />
              <Tooltip
                contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: 6, fontSize: 12 }}
                formatter={(val: any, name: any) => [`${val} veh`, name.toUpperCase()]}
              />
              <Bar dataKey="car" stackId="a" fill="#06b6d4" name="Cars" />
              <Bar dataKey="motorbike" stackId="a" fill="#f59e0b" name="Motorbikes" />
              <Bar dataKey="bus" stackId="a" fill="#f43f5e" name="Buses" />
              <Bar dataKey="truck" stackId="a" fill="#94a3b8" name="Trucks" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Chart 2: 10-Minute PCU Demand and Equivalent Hourly Rate */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card space-y-3">
          <div>
            <div className="text-sm font-bold text-slate-200">
              10-Minute PCU Arrivals & Equivalent Hourly Rate
            </div>
            <div className="text-xs text-slate-400">
              Demand measured in Passenger Car Units per 10-minute block
            </div>
          </div>

          <div className="h-60 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={intervals} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false} />
                <XAxis dataKey="label" tick={{ fill: '#94a3b8', fontSize: 10 }} angle={-25} textAnchor="end" />
                <YAxis yAxisId="pcu" tick={{ fill: '#94a3b8', fontSize: 10 }} />
                <YAxis yAxisId="rate" orientation="right" tick={{ fill: '#10b981', fontSize: 10 }} />
                <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: 6, fontSize: 12 }} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Line yAxisId="pcu" type="monotone" dataKey="pcu_arrivals" stroke="#38bdf8" strokeWidth={2.5} name="10-Min PCU" dot={{ r: 3 }} />
                <Line yAxisId="rate" type="monotone" dataKey="hourly_flow_pcu" stroke="#10b981" strokeWidth={2} strokeDasharray="4 4" name="PCU/hr Eq." dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Chart 3: Simulated 10-Minute Queue Dynamics */}
        <div className="card space-y-3">
          <div>
            <div className="text-sm font-bold text-slate-200">
              10-Minute Mean vs Peak Queue Accumulation
            </div>
            <div className="text-xs text-slate-400">
              Queue length dynamics aggregated in 10-minute evaluation units
            </div>
          </div>

          <div className="h-60 w-full">
            {simIntervals.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={simIntervals} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false} />
                  <XAxis dataKey="label" tick={{ fill: '#94a3b8', fontSize: 10 }} angle={-25} textAnchor="end" />
                  <YAxis tick={{ fill: '#94a3b8', fontSize: 10 }} />
                  <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: 6, fontSize: 12 }} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Bar dataKey="mean_queue_pcu" fill="#06b6d4" name="Mean Queue (PCU)" />
                  <Bar dataKey="peak_queue_pcu" fill="#ef4444" name="Peak Queue (PCU)" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex flex-col items-center justify-center text-center p-6 text-slate-500 text-xs">
                <Clock size={28} className="mb-2 opacity-50" />
                <p>Run the simulation for &gt;10 minutes (or run an Evaluation) to see 10-minute simulated queue trends.</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* 10-Minute Interval Numerical Table */}
      <div className="card space-y-3">
        <div className="text-sm font-bold text-slate-200">
          10-Minute Interval Dataset Summary Table
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-slate-400 text-[11px] uppercase border-b border-slate-800 bg-slate-900/60">
                <th className="py-2.5 px-3 text-left">#</th>
                <th className="py-2.5 px-3 text-left">Time Window</th>
                <th className="py-2.5 px-3 text-right">Cars</th>
                <th className="py-2.5 px-3 text-right">Bikes</th>
                <th className="py-2.5 px-3 text-right">Buses</th>
                <th className="py-2.5 px-3 text-right">Trucks</th>
                <th className="py-2.5 px-3 text-right font-bold text-white">Total Veh</th>
                <th className="py-2.5 px-3 text-right font-bold text-cyan-400">Total PCU</th>
                <th className="py-2.5 px-3 text-right text-emerald-400">Hourly Rate</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono text-slate-300">
              {intervals.map((it, idx) => (
                <tr key={idx} className="hover:bg-slate-900/40 transition-colors">
                  <td className="py-2 px-3 text-slate-500">{it.interval_index}</td>
                  <td className="py-2 px-3 font-semibold text-white">{it.label}</td>
                  <td className="py-2 px-3 text-right">{it.car}</td>
                  <td className="py-2 px-3 text-right">{it.motorbike}</td>
                  <td className="py-2 px-3 text-right">{it.bus}</td>
                  <td className="py-2 px-3 text-right">{it.truck}</td>
                  <td className="py-2 px-3 text-right font-bold text-white">{it.total_vehicles}</td>
                  <td className="py-2 px-3 text-right font-bold text-cyan-400">{it.pcu_arrivals.toFixed(1)}</td>
                  <td className="py-2 px-3 text-right text-emerald-400">{it.hourly_flow_pcu} PCU/h</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
