"""
RL Safety wrapper.

Integrates the existing SafetySupervisor with RL-generated decisions,
ensuring all PPO actions pass through the same safety validation that
governs all other controllers.

The safety layer is NEVER bypassed — even for RL actions.
"""

import logging
from typing import Dict, List, Optional, Tuple

from backend.core.schema import (
    ApproachState,
    ControllerDecision,
    ControllerMode,
    PhaseConfig,
    SafetyEvent,
    SignalState,
)
from backend.core.safety_layer import SafetySupervisor

logger = logging.getLogger(__name__)


class RLSafetyValidator:
    """
    Thin wrapper that routes RL-generated decisions through SafetySupervisor.

    SafetySupervisor already handles:
      - Conflicting green phase detection
      - Yellow / all-red clearance enforcement
      - Starvation-prevention overrides
      - Sensor fault handling

    This class exists to keep the RL controller interface clean and to
    add RL-specific logging for safety events.
    """

    def __init__(
        self,
        intersection_id: str,
        phases: List[PhaseConfig],
    ):
        self._supervisor = SafetySupervisor(
            intersection_id=intersection_id,
            phases=phases,
        )
        self._safety_events: List[SafetyEvent] = []

    def validate(
        self,
        decision: ControllerDecision,
        approaches: Dict[str, ApproachState],
        current_step: int,
        timestamp_str: str,
    ) -> Tuple[ControllerDecision, List[SafetyEvent]]:
        """
        Validate and, if necessary, override a RL-generated decision.

        Args:
            decision:      The decision produced by RLTrafficController.
            approaches:    Current intersection approach states.
            current_step:  Simulation step index.
            timestamp_str: Wall-clock time string.

        Returns:
            (validated_decision, new_safety_events)
        """
        approved, events = self._supervisor.validate_and_enforce(
            decision=decision,
            approaches=approaches,
            current_step=current_step,
            timestamp_str=timestamp_str,
        )

        if events:
            for ev in events:
                if ev.is_overridden:
                    logger.warning(
                        f"[RL Safety Override] step={current_step} "
                        f"requested_phase={ev.requested_phase_id} → "
                        f"approved_phase={ev.approved_phase_id} "
                        f"reason='{ev.override_reason}'"
                    )
            self._safety_events.extend(events)

        return approved, events

    def get_fallback_decision(
        self,
        current_step: int,
        timestamp_str: str,
        reason: str = "RL inference failure",
    ) -> Tuple[ControllerDecision, SafetyEvent]:
        """Return a safe fallback decision when RL inference fails."""
        return self._supervisor.safe_fallback_decision(
            current_step=current_step,
            timestamp_str=timestamp_str,
            exception_reason=reason,
        )

    def reset(self):
        self._supervisor.reset()
        self._safety_events.clear()

    @property
    def safety_events(self) -> List[SafetyEvent]:
        return list(self._safety_events)

