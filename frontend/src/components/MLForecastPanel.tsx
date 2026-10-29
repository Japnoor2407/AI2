import { useState, useEffect } from 'react'
import { Brain, Cpu, Play, TrendingUp, AlertCircle, BarChart2 } from 'lucide-react'
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts'
import { api } from '../services/api'
import type { MLMetricsData } from '../types'

export function MLForecastPanel() {
  const [data, setData] = useState<MLMetricsData | null>(null)
  const [loading, setLoading] = useState(false)
  const [predictLoading, setPredictLoading] = useState(false)
  const [predictions, setPredictions] = useState<number[]>([])
  const [error, setError] = useState<string | null>(null)

  const fetchMetrics = async () => {
    try {
      const res = await api.mlMetrics()
      if (res && res.status === 'ok') {
        setData(res)
      }
    } catch {
      // not trained yet
    }
  }

  useEffect(() => {
    fetchMetrics()
  }, [])

  const handleTrain = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await api.mlTrain()
      if (res.status === 'ok') {
        setData({ status: 'ok', metrics: res.metrics, model: res.metrics?.model_name })
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Model training failed')
    } finally {
      setLoading(false)
    }
  }

  const handlePredict = async () => {
    setPredictLoading(true)
    try {
      const res = await api.mlPredict(30)
      if (res.status === 'ok' && res.predictions) {
        setPredictions(res.predictions)
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Prediction failed')
    } finally {
      setPredictLoading(false)
    }
  }

  const featureImportanceData = data?.metrics?.feature_importance
    ? Object.entries(data.metrics.feature_importance).map(([feature, importance]) => ({
        feature,
        importance: Number((importance * 100).toFixed(1)),
      }))
    : []

  return (
    <div className="space-y-6">
      {/* ML Overview Card */}
      <div className="card">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h2 className="text-base font-bold text-slate-100 flex items-center gap-2">
              <Brain className="w-5 h-5 text-purple-400" />
              <span>Machine Learning Short-Term Traffic Demand Forecaster</span>
              <span className="badge badge-purple text-[10px]">Gradient Boosting</span>
            </h2>
            <p className="text-xs text-slate-400 mt-1">
              Short-horizon predictive intelligence forecasting vehicle arrival rates using autoregressive lag features, rolling statistics, and chronological validation.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handlePredict}
              disabled={predictLoading || !data?.metrics}
              className="btn btn-secondary text-xs flex items-center gap-1.5"
            >
              <TrendingUp className="w-3.5 h-3.5" />
              {predictLoading ? 'Forecasting...' : 'Predict Next 30s'}
            </button>
            <button
              onClick={handleTrain}
              disabled={loading}
              className="btn btn-primary text-xs flex items-center gap-2"
            >
              <Play className="w-3.5 h-3.5 fill-current" />
              {loading ? 'Training Model...' : 'Train Forecaster (Pune CCTV)'}
            </button>
          </div>
        </div>

        {error && (
          <div className="mt-3 p-2 bg-rose-950/60 border border-rose-800 rounded text-rose-300 text-xs flex items-center gap-2">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}
      </div>

      {data && data.metrics ? (
        <>
          {/* Key Validation Metrics */}
          <div className="grid grid-cols-1 sm:grid-cols-4 gap-3">
            <div className="card">
              <div className="text-xs text-slate-400 font-semibold uppercase tracking-wider">Model Architecture</div>
              <div className="text-sm font-bold font-mono text-purple-400 mt-2 truncate">
                {data.metrics.model_name}
              </div>
              <div className="text-[11px] text-slate-500 mt-1">
                Train: {data.metrics.train_samples} | Test: {data.metrics.test_samples}
              </div>
            </div>

            <div className="card">
              <div className="text-xs text-slate-400 font-semibold uppercase tracking-wider">Test MAE (Error)</div>
              <div className="text-2xl font-bold font-mono text-emerald-400 mt-2">
                {data.metrics.mae_test.toFixed(3)} <span className="text-xs font-normal text-slate-400">PCU/s</span>
              </div>
              <div className="text-[11px] text-slate-500 mt-1">
                Baseline (Persistence): {data.metrics.baseline_persistence_mae.toFixed(3)}
              </div>
            </div>

            <div className="card">
              <div className="text-xs text-slate-400 font-semibold uppercase tracking-wider">Baseline Improvement</div>
              <div className="text-2xl font-bold font-mono text-cyan-400 mt-2">
                +{data.metrics.mae_improvement_pct.toFixed(1)}%
              </div>
              <div className="text-[11px] text-slate-500 mt-1">
                Outperforms naive zero-order hold
              </div>
            </div>

            <div className="card">
              <div className="text-xs text-slate-400 font-semibold uppercase tracking-wider">Goodness of Fit (R²)</div>
              <div className="text-2xl font-bold font-mono text-amber-400 mt-2">
                {data.metrics.r2_test.toFixed(3)}
              </div>
              <div className="text-[11px] text-slate-500 mt-1">
                RMSE: {data.metrics.rmse_test.toFixed(3)} PCU/s
              </div>
            </div>
          </div>

          {/* Model Benchmark vs Baselines Table */}
          <div className="card">
            <div className="text-xs font-bold text-slate-300 uppercase tracking-wider mb-3 flex items-center gap-2">
              <BarChart2 className="w-4 h-4 text-cyan-400" />
              <span>Scientific Baseline Comparison (Chronological Holdout Split)</span>
            </div>
            <table className="w-full text-xs text-left">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400">
                  <th className="py-2 px-3 font-semibold">Forecasting Strategy</th>
                  <th className="py-2 px-3 font-semibold text-right">Mean Absolute Error (MAE)</th>
                  <th className="py-2 px-3 font-semibold text-right">RMSE</th>
                  <th className="py-2 px-3 font-semibold text-right">Relative Improvement</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono">
                <tr>
                  <td className="py-2.5 px-3 font-sans text-slate-400">Baseline 1: Persistence (y_t = y_t-1)</td>
                  <td className="py-2.5 px-3 text-right text-slate-300">{data.metrics.baseline_persistence_mae.toFixed(4)}</td>
                  <td className="py-2.5 px-3 text-right text-slate-300">{data.metrics.baseline_persistence_rmse.toFixed(4)}</td>
                  <td className="py-2.5 px-3 text-right text-slate-500">Benchmark</td>
                </tr>
                <tr>
                  <td className="py-2.5 px-3 font-sans text-slate-400">Baseline 2: Rolling 60s Moving Average</td>
                  <td className="py-2.5 px-3 text-right text-slate-300">{data.metrics.baseline_ma_mae.toFixed(4)}</td>
                  <td className="py-2.5 px-3 text-right text-slate-500">—</td>
                  <td className="py-2.5 px-3 text-right text-slate-500">—</td>
                </tr>
                <tr className="bg-purple-950/20">
                  <td className="py-2.5 px-3 font-sans font-bold text-purple-300">Trained Machine Learning Model</td>
                  <td className="py-2.5 px-3 text-right text-emerald-400 font-bold">{data.metrics.mae_test.toFixed(4)}</td>
                  <td className="py-2.5 px-3 text-right text-emerald-400 font-bold">{data.metrics.rmse_test.toFixed(4)}</td>
                  <td className="py-2.5 px-3 text-right text-cyan-400 font-bold">+{data.metrics.mae_improvement_pct.toFixed(1)}%</td>
                </tr>
              </tbody>
            </table>
          </div>

          {/* Feature Importance */}
          {featureImportanceData.length > 0 && (
            <div className="card">
              <div className="text-xs font-bold text-slate-300 uppercase tracking-wider mb-4">
                Feature Importance Attribution
              </div>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={featureImportanceData} layout="vertical" margin={{ top: 5, right: 30, left: 60, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis type="number" stroke="#64748b" tick={{ fontSize: 10 }} unit="%" />
                    <YAxis dataKey="feature" type="category" stroke="#64748b" tick={{ fontSize: 10 }} width={70} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', fontSize: '12px' }}
                      itemStyle={{ color: '#c084fc' }}
                    />
                    <Bar dataKey="importance" fill="#a855f7" radius={[0, 4, 4, 0]} name="Importance (%)" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Live Predictions */}
          {predictions.length > 0 && (
            <div className="card">
              <div className="text-xs font-bold text-slate-300 uppercase tracking-wider mb-2 flex items-center justify-between">
                <span>Predicted Demand Trajectory (Next 30 Steps)</span>
                <span className="text-[10px] text-slate-500 font-mono">1-Sec Horizon</span>
              </div>
              <div className="flex flex-wrap gap-1.5 font-mono text-[11px] max-h-32 overflow-y-auto pt-2">
                {predictions.map((val, idx) => (
                  <span key={idx} className="px-2 py-1 bg-slate-900 border border-slate-800 rounded text-cyan-300">
                    t+{idx + 1}s: {val.toFixed(2)} PCU
                  </span>
                ))}
              </div>
            </div>
          )}
        </>
      ) : (
        <div className="card text-center py-12">
          <Cpu className="w-10 h-10 text-slate-600 mx-auto mb-3" />
          <div className="text-slate-400 text-sm font-semibold mb-1">No ML model trained yet</div>
          <p className="text-slate-500 text-xs mb-4 max-w-md mx-auto">
            Click Train to build a GradientBoostingRegressor model trained on empirical Pune CCTV timestamps with chronological splits and feature attribution.
          </p>
          <button
            onClick={handleTrain}
            disabled={loading}
            className="btn btn-primary text-xs inline-flex items-center gap-2"
          >
            <Brain className="w-3.5 h-3.5" />
            {loading ? 'Training...' : 'Train ML Forecaster'}
          </button>
        </div>
      )}
    </div>
  )
}
