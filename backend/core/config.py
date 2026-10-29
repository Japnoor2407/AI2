import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = BASE_DIR / "traffic" / "traffic"

# Traffic PCU Equivalencies (IRC:106 Standards for Indian Urban Conditions)
PCU_WEIGHTS = {
    "car": 1.0,
    "motorbike": 0.5,
    "bus": 2.5,
    "truck": 2.5,
}

# Discharge / Saturation Capacity per approach (PCU per second of green)
DEFAULT_SATURATION_FLOW_PCU_PER_SEC = 0.6  # ~2160 PCU/lane/hour

# Controller Safety & Performance Constraints
MIN_GREEN_SEC = 10
MAX_GREEN_SEC = 55
YELLOW_CLEARANCE_SEC = 3
ALL_RED_CLEARANCE_SEC = 2
STARVATION_THRESHOLD_SEC = 90  # Max waiting time before priority override

# Adaptive Controller Decision Weights
DEFAULT_ADAPTIVE_WEIGHTS = {
    "wq": 0.40,  # Weight on normalized queue length
    "wa": 0.25,  # Weight on incoming arrival rate
    "ww": 0.20,  # Weight on accumulated waiting time
    "wt": 0.15,  # Weight on time since last served (anti-starvation)
}

# Fixed-time baseline cycle specifications
FIXED_CYCLE_SPLIT_SEC = 30  # 30 seconds green for each phase in round-robin

