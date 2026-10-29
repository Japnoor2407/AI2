from typing import Dict, List
from backend.core.schema import PhaseConfig


class JunctionLayout:
    """
    Defines intersection topology, approach-to-camera/direction mappings,
    and valid signal phase groups to prevent conflicting greens.
    
    Supports:
    - 4_phase (Default): Protected movement where EXACTLY ONE approach is GREEN at a time.
    - 2_phase: Dual-arterial movement where North-South or East-West share green.
    """

    def __init__(self, junction_id: str = "AlankarChowk", plan_mode: str = "4_phase"):
        self.junction_id = junction_id
        self.plan_mode = plan_mode
        self.approaches = self._configure_approaches()
        self.phases = self._configure_phases()

    def _configure_approaches(self) -> Dict[str, Dict]:
        if self.junction_id == "AlankarChowk":
            return {
                "North_Approach": {
                    "name": "North (Station Road)",
                    "camera_id": "a2",
                    "direction": "UP",
                    "saturation_flow": 0.6,
                },
                "South_Approach": {
                    "name": "South (Camp Corridor)",
                    "camera_id": "a2",
                    "direction": "DOWN",
                    "saturation_flow": 0.6,
                },
                "East_Approach": {
                    "name": "East (Sassoon Arterial)",
                    "camera_id": "a3",
                    "direction": "RIGHT",
                    "saturation_flow": 0.6,
                },
                "West_Approach": {
                    "name": "West (Collectorate Rd)",
                    "camera_id": "a3",
                    "direction": "LEFT",
                    "saturation_flow": 0.6,
                },
            }
        elif self.junction_id == "JehangirChowk":
            return {
                "North_Approach": {
                    "name": "Bund Garden Approach",
                    "camera_id": "j1",
                    "direction": "DOWN",
                    "saturation_flow": 0.6,
                },
                "South_Approach": {
                    "name": "Hospital Approach",
                    "camera_id": "j2",
                    "direction": "UP",
                    "saturation_flow": 0.6,
                },
                "East_Approach": {
                    "name": "Pune Station Flow",
                    "camera_id": "j3",
                    "direction": "RIGHT",
                    "saturation_flow": 0.6,
                },
                "West_Approach": {
                    "name": "RTO Feeder",
                    "camera_id": "j2",
                    "direction": "LEFT",
                    "saturation_flow": 0.6,
                },
            }
        else:  # RTOChowk
            return {
                "North_Approach": {
                    "name": "Sangam Bridge North",
                    "camera_id": "r1",
                    "direction": "UP",
                    "saturation_flow": 0.6,
                },
                "South_Approach": {
                    "name": "RTO Office South",
                    "camera_id": "r2",
                    "direction": "DOWN",
                    "saturation_flow": 0.6,
                },
                "East_Approach": {
                    "name": "Engineering College East",
                    "camera_id": "r3",
                    "direction": "RIGHT",
                    "saturation_flow": 0.6,
                },
                "West_Approach": {
                    "name": "Railway Crossing West",
                    "camera_id": "r1",
                    "direction": "LEFT",
                    "saturation_flow": 0.6,
                },
            }

    def _configure_phases(self) -> List[PhaseConfig]:
        """
        Phase configurations:
        In 4-phase mode: Exactly ONE approach receives green at any time,
        preventing any two approaches from being green simultaneously.
        """
        if self.plan_mode == "4_phase":
            return [
                PhaseConfig(
                    phase_id=1,
                    name="Phase 1: North Protected Flow",
                    served_approaches=["North_Approach"],
                    min_green_sec=10,
                    max_green_sec=55,
                    yellow_duration_sec=3,
                    all_red_duration_sec=2,
                ),
                PhaseConfig(
                    phase_id=2,
                    name="Phase 2: East Protected Flow",
                    served_approaches=["East_Approach"],
                    min_green_sec=10,
                    max_green_sec=55,
                    yellow_duration_sec=3,
                    all_red_duration_sec=2,
                ),
                PhaseConfig(
                    phase_id=3,
                    name="Phase 3: South Protected Flow",
                    served_approaches=["South_Approach"],
                    min_green_sec=10,
                    max_green_sec=55,
                    yellow_duration_sec=3,
                    all_red_duration_sec=2,
                ),
                PhaseConfig(
                    phase_id=4,
                    name="Phase 4: West Protected Flow",
                    served_approaches=["West_Approach"],
                    min_green_sec=10,
                    max_green_sec=55,
                    yellow_duration_sec=3,
                    all_red_duration_sec=2,
                ),
            ]
        else:
            return [
                PhaseConfig(
                    phase_id=1,
                    name="Phase 1: North-South Arterial",
                    served_approaches=["North_Approach", "South_Approach"],
                    min_green_sec=10,
                    max_green_sec=55,
                    yellow_duration_sec=3,
                    all_red_duration_sec=2,
                ),
                PhaseConfig(
                    phase_id=2,
                    name="Phase 2: East-West Crossway",
                    served_approaches=["East_Approach", "West_Approach"],
                    min_green_sec=10,
                    max_green_sec=55,
                    yellow_duration_sec=3,
                    all_red_duration_sec=2,
                ),
            ]

    def validate_safety(self, active_phase_id: int) -> bool:
        """
        Safety invariant: verifies that active phase does not include conflicting approaches.
        """
        phase_map = {p.phase_id: p for p in self.phases}
        if active_phase_id not in phase_map:
            return False
        active_approaches = set(phase_map[active_phase_id].served_approaches)
        # Check conflict: North/South cannot mix with East/West
        ns = {"North_Approach", "South_Approach"}
        ew = {"East_Approach", "West_Approach"}
        if (active_approaches & ns) and (active_approaches & ew):
            return False
        return True
