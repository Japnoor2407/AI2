export interface ApproachData {
  name: string
  direction: string
  signal: 'GREEN' | 'YELLOW' | 'ALL_RED' | 'RED'
  queue_pcu: number
  queue_vehicles: number
  arrival_rate_pcu_per_min: number
  avg_wait_sec: number
  time_since_served_sec: number
  is_starving: boolean
  arrivals: { car: number; motorbike: number; bus: number; truck: number }
}

export interface ControllerDecision {
  step: number
  timestamp_str: string
  mode: 'FIXED_TIME' | 'ADAPTIVE'
  active_phase_id: number
  phase_name: string
  signal_state: string
  time_remaining_in_state: number
  elapsed_in_phase: number
  phase_demand_scores: Record<string, number>
  decision_rationale: string
  is_transitioning: boolean
  transition_stage: string | null
}

export interface TrafficCurrent {
  step: number
  timestamp: string
  total_queue_pcu: number
  total_throughput_pcu: number
  average_delay_sec: number
  approaches: Record<string, ApproachData>
  controller: ControllerDecision | null
}

export interface HistoryPoint {
  step: number
  time: string
  queue_pcu: number
  throughput_pcu: number
  delay_sec: number
  phase_id: number
  signal_state: string
}

export interface EvalMetrics {
  mean_queue_pcu: number
  max_queue_pcu: number
  average_wait_sec: number
  total_throughput_pcu: number
  starvation_occurrences: number
}

export interface EvaluationSummary {
  status: string
  message?: string
  scenario_id?: string
  junction?: string
  duration_sec?: number
  fixed_time?: EvalMetrics
  adaptive?: EvalMetrics
  improvements?: {
    queue_reduction_percent: number
    waiting_time_reduction_percent: number
    throughput_improvement_percent: number
  }
  summary?: string
  evaluated_at?: string
  time_series_length?: number
}

export interface ControllerStatus {
  mode: string
  is_running: boolean
  is_paused: boolean
  step: number
  total_steps: number
  junction: string
  date: string
  current_phase: number | null
  signal_state: string | null
  decision_rationale: string | null
}

export interface DecisionLogEntry {
  step: number
  time: string
  mode: string
  phase_id: number
  phase_name: string
  signal: string
  time_remaining: number
  demand_scores: Record<string, number>
  rationale: string
  is_transitioning: boolean
}

export interface AggregatedInterval {
  interval_index: number
  label: string
  start_time: string
  end_time: string
  duration_sec: number
  car: number
  motorbike: number
  bus: number
  truck: number
  total_vehicles: number
  pcu_arrivals: number
  hourly_flow_pcu: number
}

export interface SimAggregatedInterval {
  interval_index: number
  label: string
  start_time: string
  end_time: string
  mean_queue_pcu: number
  peak_queue_pcu: number
  mean_delay_sec: number
  throughput_pcu: number
}

export interface AggregatedResponse {
  status: string
  junction: string
  date: string
  camera: string
  unit: string
  interval_seconds: number
  intervals: AggregatedInterval[]
  simulation_intervals: SimAggregatedInterval[]
}


