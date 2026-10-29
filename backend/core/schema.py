from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class SignalState(str, Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    ALL_RED = "ALL_RED"
    RED = "RED"


class ControllerMode(str, Enum):
    FIXED_TIME = "FIXED_TIME"
    ACTUATED = "ACTUATED"
    ADAPTIVE = "ADAPTIVE"
    PREDICTION_ASSISTED = "PREDICTION_ASSISTED"
    COORDINATED = "COORDINATED"


class VehicleClassCounts(BaseModel):
    car: int = 0
    motorbike: int = 0
    bus: int = 0
    truck: int = 0

    @property
    def total_vehicles(self) -> int:
        return self.car + self.motorbike + self.bus + self.truck

    def to_pcu(self, car_w=1.0, bike_w=0.5, bus_w=2.5, truck_w=2.5) -> float:
        return (
            self.car * car_w
            + self.motorbike * bike_w
            + self.bus * bus_w
            + self.truck * truck_w
        )


class ApproachState(BaseModel):
    approach_id: str
    approach_name: str
    direction: str
    camera_id: str
    junction_id: str
    current_signal: SignalState = SignalState.RED
    arrivals_last_step: VehicleClassCounts = Field(default_factory=VehicleClassCounts)
    cumulative_arrivals_pcu: float = 0.0
    cumulative_departures_pcu: float = 0.0
    estimated_queue_pcu: float = 0.0
    estimated_queue_vehicles: int = 0
    arrival_rate_pcu_per_min: float = 0.0
    average_wait_time_sec: float = 0.0
    max_wait_time_sec: float = 0.0
    time_since_last_served_sec: float = 0.0
    is_starving: bool = False
    saturation_flow_rate_pcu_per_sec: float = 0.6  # approx 2160 PCU/hr


class PhaseConfig(BaseModel):
    phase_id: int
    name: str
    served_approaches: List[str]
    min_green_sec: int = 10
    max_green_sec: int = 60
    yellow_duration_sec: int = 3
    all_red_duration_sec: int = 2


class ControllerDecision(BaseModel):
    step: int
    timestamp_str: str
    mode: ControllerMode
    active_phase_id: int
    phase_name: str
    signal_state: SignalState
    time_remaining_in_state: int
    elapsed_in_phase: int
    phase_demand_scores: Dict[str, float] = Field(default_factory=dict)
    decision_rationale: str = ""
    is_transitioning: bool = False
    transition_stage: Optional[str] = None  # YELLOW, ALL_RED, or None
    safety_override: bool = False
    safety_override_reason: Optional[str] = None
    predicted_arrivals_pcu: Optional[float] = None
    execution_time_ms: float = 0.0


class StepObservation(BaseModel):
    step: int
    timestamp_str: str
    controller_decision: ControllerDecision
    approaches: Dict[str, ApproachState]
    total_active_queue_pcu: float
    total_throughput_pcu: float
    average_delay_sec: float


class MetricRecord(BaseModel):
    mean_queue_pcu: float
    median_wait_sec: float = 0.0
    p95_wait_sec: float = 0.0
    max_queue_pcu: float
    average_wait_sec: float
    max_wait_sec: float
    total_throughput_vehicles: int
    total_throughput_pcu: float
    total_arrived_vehicles: int
    residual_queue_pcu: float = 0.0
    completed_trips: int = 0
    unfinished_trips: int = 0
    safety_violations: int = 0
    controller_execution_time_ms: float = 0.0
    starvation_occurrences: int = 0
    per_approach_stats: Dict[str, Dict[str, float]] = Field(default_factory=dict)


class SafetyEvent(BaseModel):
    step: int
    timestamp_str: str
    intersection_id: str
    previous_phase_id: int
    requested_phase_id: int
    approved_phase_id: int
    is_overridden: bool
    override_reason: str
    active_constraints: List[str] = Field(default_factory=list)


class EvaluationReport(BaseModel):
    scenario_id: str
    scenario_name: str = "Empirical Pune CCTV"
    junction_name: str
    evaluation_duration_sec: int
    start_time: str
    end_time: str
    random_seed: Optional[int] = None
    fixed_time_baseline: MetricRecord
    actuated_baseline: Optional[MetricRecord] = None
    adaptive_controller: MetricRecord
    coordinated_controller: Optional[MetricRecord] = None
    queue_reduction_percent: float
    waiting_time_reduction_percent: float
    throughput_improvement_percent: float
    improvements_vs_actuated: Optional[Dict[str, float]] = None
    timestamp_evaluated: str
    summary_text: str
    time_series: List[Dict[str, Any]] = Field(default_factory=list)
