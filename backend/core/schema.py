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
    ADAPTIVE = "ADAPTIVE"


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
    max_queue_pcu: float
    average_wait_sec: float
    max_wait_sec: float
    total_throughput_vehicles: int
    total_throughput_pcu: float
    total_arrived_vehicles: int
    starvation_occurrences: int
    per_approach_stats: Dict[str, Dict[str, float]] = Field(default_factory=dict)


class EvaluationReport(BaseModel):
    scenario_id: str
    junction_name: str
    evaluation_duration_sec: int
    start_time: str
    end_time: str
    fixed_time_baseline: MetricRecord
    adaptive_controller: MetricRecord
    queue_reduction_percent: float
    waiting_time_reduction_percent: float
    throughput_improvement_percent: float
    timestamp_evaluated: str
    summary_text: str
    time_series: List[Dict[str, Any]] = Field(default_factory=list)

