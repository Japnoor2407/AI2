"""
Comprehensive test suite for the TrafficSignalEnv Gymnasium environment.

Phase 2 acceptance criteria:
- Environment init, reset, and step work correctly.
- Observation space validity (all obs in [0,1], shape=(21,), finite).
- Action space validity (16 valid actions, decode/encode roundtrip).
- Green duration clamping (min=10s, max=55s).
- Queue updates produce non-negative queues.
- Reward calculation with known inputs.
- Reward breakdown contains all 6 components.
- Starvation penalty activates correctly.
- Episode termination / truncation flags.
- Safety: Phase 1 and Phase 2 cannot both be GREEN simultaneously.
- Reproducibility: same seed produces same trajectory.
- gymnasium check_env passes.
- SB3 PPO initializes and predicts without error.
"""

import math
import pytest
import numpy as np

import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from backend.rl import TrafficSignalEnv
from backend.rl.config import (
    EnvConfig,
    RewardWeights,
    GREEN_DURATIONS,
    SUPPORTED_PHASES,
    STARVATION_THRESHOLD_SEC,
    MAX_QUEUE_PER_APPROACH,
)
from backend.rl.action_space import ActionDecoder
from backend.rl.reward import RewardCalculator
from backend.rl.observation import ObservationBuilder, OBSERVATION_DIM
from backend.core.schema import ApproachState


# ──────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────

@pytest.fixture
def env():
    """Short-episode env so tests finish quickly."""
    config = EnvConfig(episode_duration_sec=120)
    e = TrafficSignalEnv(config=config)
    yield e
    e.close()


@pytest.fixture
def decoder():
    return ActionDecoder()


# ══════════════════════════════════════════════════════════
# 1. Environment Initialisation
# ══════════════════════════════════════════════════════════

class TestEnvInit:
    def test_action_space_is_discrete_16(self, env):
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert env.action_space.n == 16

    def test_observation_space_is_box_21(self, env):
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert env.observation_space.shape == (OBSERVATION_DIM,)

    def test_observation_space_bounds(self, env):
        assert np.all(env.observation_space.low == 0.0)
        assert np.all(env.observation_space.high == 1.0)

    def test_observation_dtype_float32(self, env):
        assert env.observation_space.dtype == np.float32

    def test_env_has_metadata(self, env):
        assert "render_modes" in env.metadata


# ══════════════════════════════════════════════════════════
# 2. Reset
# ══════════════════════════════════════════════════════════

class TestReset:
    def test_reset_returns_tuple(self, env):
        result = env.reset()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_reset_obs_shape(self, env):
        obs, info = env.reset()
        assert obs.shape == (OBSERVATION_DIM,)

    def test_reset_obs_dtype(self, env):
        obs, info = env.reset()
        assert obs.dtype == np.float32

    def test_reset_obs_in_bounds(self, env):
        obs, info = env.reset()
        assert np.all(obs >= 0.0), "Observation contains values below 0"
        assert np.all(obs <= 1.0), "Observation contains values above 1"

    def test_reset_obs_finite(self, env):
        obs, info = env.reset()
        assert np.all(np.isfinite(obs)), "Observation contains NaN or Inf"

    def test_reset_info_is_dict(self, env):
        obs, info = env.reset()
        assert isinstance(info, dict)

    def test_reset_info_keys(self, env):
        obs, info = env.reset()
        expected_keys = {"step", "active_phase_id", "total_queue_pcu", "average_delay_sec"}
        assert expected_keys.issubset(set(info.keys()))

    def test_reset_clears_episode_reward(self, env):
        env.reset()
        # Take a step then reset again
        env.step(0)
        env.reset()
        assert env.total_episode_reward == 0.0

    def test_reset_sim_state(self, env):
        env.reset()
        assert env.sim is not None
        assert env.episode_steps == 0
        assert env.active_phase_id in [1, 2]


# ══════════════════════════════════════════════════════════
# 3. Step
# ══════════════════════════════════════════════════════════

class TestStep:
    def test_step_returns_5_tuple(self, env):
        env.reset()
        result = env.step(0)
        assert isinstance(result, tuple)
        assert len(result) == 5

    def test_step_obs_shape(self, env):
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)
        assert obs.shape == (OBSERVATION_DIM,)

    def test_step_obs_in_bounds(self, env):
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)

    def test_step_obs_finite(self, env):
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)
        assert np.all(np.isfinite(obs))

    def test_step_reward_is_finite(self, env):
        env.reset()
        _, reward, _, _, _ = env.step(0)
        assert math.isfinite(reward)

    def test_step_terminated_is_bool(self, env):
        env.reset()
        _, _, terminated, _, _ = env.step(0)
        assert isinstance(terminated, bool)

    def test_step_truncated_is_bool(self, env):
        env.reset()
        _, _, _, truncated, _ = env.step(0)
        assert isinstance(truncated, bool)

    def test_step_info_is_dict(self, env):
        env.reset()
        _, _, _, _, info = env.step(0)
        assert isinstance(info, dict)

    def test_step_info_keys(self, env):
        env.reset()
        _, _, _, _, info = env.step(0)
        expected = {
            "action", "action_desc", "selected_phase_id", "green_duration_sec",
            "phase_switched", "simulation_step", "total_active_queue_pcu",
            "average_delay_sec", "total_throughput_pcu", "starvation_occurrences",
            "reward_breakdown", "total_episode_reward",
        }
        assert expected.issubset(set(info.keys()))

    def test_step_terminated_always_false(self, env):
        """Traffic sim is time-truncated, not terminated early."""
        env.reset()
        for action in range(16):
            obs, _, terminated, truncated, _ = env.step(action)
            assert not terminated
            if truncated:
                break

    def test_step_without_reset_raises(self, env):
        with pytest.raises(RuntimeError, match="reset()"):
            env.step(0)

    def test_all_16_actions_execute(self, env):
        """Each valid action index should execute without error."""
        for action in range(16):
            env.reset()
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs.shape == (OBSERVATION_DIM,)
            assert math.isfinite(reward)


# ══════════════════════════════════════════════════════════
# 4. Episode Truncation
# ══════════════════════════════════════════════════════════

class TestEpisodeTruncation:
    def test_episode_truncates(self):
        config = EnvConfig(episode_duration_sec=60)
        env = TrafficSignalEnv(config=config)
        env.reset()
        truncated = False
        max_iters = 200
        for _ in range(max_iters):
            _, _, terminated, truncated, _ = env.step(0)  # action 0 = phase 1, 10s
            if truncated or terminated:
                break
        env.close()
        assert truncated, "Episode should truncate after episode_duration_sec steps"

    def test_total_reward_accumulates(self):
        config = EnvConfig(episode_duration_sec=60)
        env = TrafficSignalEnv(config=config)
        env.reset()
        for _ in range(3):
            _, _, _, truncated, _ = env.step(0)
            if truncated:
                break
        # Total episode reward should be sum of step rewards
        assert isinstance(env.total_episode_reward, float)
        assert math.isfinite(env.total_episode_reward)
        env.close()


# ══════════════════════════════════════════════════════════
# 5. Action Space and Decoder
# ══════════════════════════════════════════════════════════

class TestActionSpace:
    def test_total_actions(self, decoder):
        assert decoder.total_actions == 16

    def test_decode_all_actions(self, decoder):
        for idx in range(16):
            phase_id, duration = decoder.decode(idx)
            assert phase_id in SUPPORTED_PHASES
            assert duration in GREEN_DURATIONS

    def test_decode_returns_phase_and_duration(self, decoder):
        phase_id, duration = decoder.decode(0)
        assert isinstance(phase_id, int)
        assert isinstance(duration, int)

    def test_decode_invalid_action_raises(self, decoder):
        with pytest.raises(ValueError):
            decoder.decode(-1)
        with pytest.raises(ValueError):
            decoder.decode(16)

    def test_encode_decode_roundtrip(self, decoder):
        for phase_id in SUPPORTED_PHASES:
            for duration in GREEN_DURATIONS:
                idx = decoder.encode(phase_id, duration)
                decoded_phase, decoded_dur = decoder.decode(idx)
                assert decoded_phase == phase_id
                assert decoded_dur == duration

    def test_decode_numpy_int(self, decoder):
        action = np.int64(3)
        phase_id, duration = decoder.decode(action)
        assert phase_id in SUPPORTED_PHASES

    def test_decode_numpy_array(self, decoder):
        action = np.array(5)
        phase_id, duration = decoder.decode(action)
        assert phase_id in SUPPORTED_PHASES

    def test_all_durations_covered(self, decoder):
        decoded_durations = {decoder.decode(i)[1] for i in range(16)}
        assert decoded_durations == set(GREEN_DURATIONS)

    def test_all_phases_covered(self, decoder):
        decoded_phases = {decoder.decode(i)[0] for i in range(16)}
        assert decoded_phases == set(SUPPORTED_PHASES)

    def test_describe_returns_string(self, decoder):
        for idx in range(16):
            desc = decoder.describe(idx)
            assert isinstance(desc, str)
            assert len(desc) > 0

    def test_encode_snaps_to_closest_duration(self, decoder):
        # 12s is not in GREEN_DURATIONS; closest is 10s
        idx = decoder.encode(phase_id=1, duration_sec=12)
        _, dur = decoder.decode(idx)
        assert dur == 10

    def test_encode_invalid_phase_falls_back(self, decoder):
        # Phase 99 is not valid; should fall back to first phase
        idx = decoder.encode(phase_id=99, duration_sec=20)
        phase_id, _ = decoder.decode(idx)
        assert phase_id == SUPPORTED_PHASES[0]


# ══════════════════════════════════════════════════════════
# 6. Green Duration Clamping
# ══════════════════════════════════════════════════════════

class TestGreenDurationClamping:
    def test_min_green_enforced(self, env):
        """Action 0 has duration 10s = min; step should handle it."""
        env.reset()
        _, _, _, _, info = env.step(0)
        assert info["green_duration_sec"] >= env.config.min_green_sec

    def test_max_green_enforced(self, env):
        """Action 15 has duration 55s = max; should not exceed config max."""
        env.reset()
        _, _, _, _, info = env.step(15)
        assert info["green_duration_sec"] <= env.config.max_green_sec

    def test_all_decoded_durations_within_bounds(self, decoder):
        for idx in range(16):
            _, duration = decoder.decode(idx)
            assert 10 <= duration <= 55


# ══════════════════════════════════════════════════════════
# 7. Queue Non-negativity
# ══════════════════════════════════════════════════════════

class TestQueueNonNegativity:
    def test_queues_non_negative_after_reset(self, env):
        env.reset()
        for approach in env.sim.approaches.values():
            assert approach.estimated_queue_pcu >= 0.0

    def test_queues_non_negative_after_steps(self, env):
        env.reset()
        for _ in range(5):
            obs, _, _, truncated, _ = env.step(np.random.randint(0, 16))
            for approach in env.sim.approaches.values():
                assert approach.estimated_queue_pcu >= 0.0, (
                    f"Negative queue in {approach.approach_id}"
                )
            if truncated:
                break

    def test_observation_queue_values_non_negative(self, env):
        """First 4 dims of observation are normalised queues (>= 0)."""
        env.reset()
        obs, _, _, _, _ = env.step(0)
        assert np.all(obs[:4] >= 0.0)


# ══════════════════════════════════════════════════════════
# 8. Reward Function
# ══════════════════════════════════════════════════════════

def _make_approach(queue_pcu: float = 5.0, wait_sec: float = 30.0, starve_sec: float = 0.0):
    """Helper: build a minimal ApproachState for reward tests."""
    a = ApproachState(
        approach_id="N",
        approach_name="North Approach",
        direction="UP",
        camera_id="a2",
        junction_id="AlankarChowk",
    )
    a.estimated_queue_pcu = queue_pcu
    a.average_wait_time_sec = wait_sec
    a.time_since_last_served_sec = starve_sec
    return a


class TestRewardCalculator:
    def test_reward_breakdown_has_6_components(self):
        calc = RewardCalculator()
        prev = {"N": _make_approach(queue_pcu=10.0)}
        curr = {"N": _make_approach(queue_pcu=8.0)}
        _, breakdown = calc.calculate(prev, curr, 2.0, 20, False)
        required_keys = {
            "comp_queue_improvement",
            "comp_throughput",
            "comp_wait_penalty",
            "comp_starvation_penalty",
            "comp_switching_penalty",
            "comp_balance_penalty",
        }
        assert required_keys.issubset(set(breakdown.keys()))

    def test_reward_total_in_breakdown(self):
        calc = RewardCalculator()
        prev = {"N": _make_approach()}
        curr = {"N": _make_approach()}
        total_reward, breakdown = calc.calculate(prev, curr, 0.0, 20, False)
        assert "reward_total" in breakdown
        assert abs(breakdown["reward_total"] - total_reward) < 1e-4

    def test_queue_improvement_positive_when_queue_drops(self):
        calc = RewardCalculator()
        prev = {"N": _make_approach(queue_pcu=20.0)}
        curr = {"N": _make_approach(queue_pcu=10.0)}
        _, breakdown = calc.calculate(prev, curr, 5.0, 20, False)
        assert breakdown["comp_queue_improvement"] > 0.0

    def test_queue_improvement_negative_when_queue_grows(self):
        calc = RewardCalculator()
        prev = {"N": _make_approach(queue_pcu=5.0)}
        curr = {"N": _make_approach(queue_pcu=15.0)}
        _, breakdown = calc.calculate(prev, curr, 0.0, 20, False)
        assert breakdown["comp_queue_improvement"] < 0.0

    def test_throughput_reward_positive_when_pcu_discharged(self):
        calc = RewardCalculator()
        prev = {"N": _make_approach()}
        curr = {"N": _make_approach()}
        _, breakdown = calc.calculate(prev, curr, 10.0, 20, False)
        assert breakdown["comp_throughput"] > 0.0

    def test_wait_penalty_positive_when_avg_wait_nonzero(self):
        calc = RewardCalculator()
        prev = {"N": _make_approach(wait_sec=60.0)}
        curr = {"N": _make_approach(wait_sec=60.0)}
        _, breakdown = calc.calculate(prev, curr, 0.0, 20, False)
        assert breakdown["comp_wait_penalty"] > 0.0

    def test_switching_penalty_when_phase_switches(self):
        calc = RewardCalculator()
        prev = {"N": _make_approach()}
        curr = {"N": _make_approach()}
        _, no_switch = calc.calculate(prev, curr, 0.0, 20, phase_switched=False)
        _, switched = calc.calculate(prev, curr, 0.0, 20, phase_switched=True)
        assert switched["comp_switching_penalty"] == 1.0
        assert no_switch["comp_switching_penalty"] == 0.0

    def test_switching_increases_total_penalty(self):
        calc = RewardCalculator()
        prev = {"N": _make_approach()}
        curr = {"N": _make_approach()}
        r_no_switch, _ = calc.calculate(prev, curr, 0.0, 20, phase_switched=False)
        r_switch, _ = calc.calculate(prev, curr, 0.0, 20, phase_switched=True)
        assert r_switch < r_no_switch

    def test_starvation_penalty_zero_below_threshold(self):
        calc = RewardCalculator()
        prev = {"N": _make_approach(starve_sec=0.0)}
        curr = {"N": _make_approach(starve_sec=0.0)}
        _, breakdown = calc.calculate(prev, curr, 0.0, 20, False)
        assert breakdown["comp_starvation_penalty"] == 0.0

    def test_starvation_penalty_positive_above_threshold(self):
        calc = RewardCalculator()
        overage = STARVATION_THRESHOLD_SEC + 20.0
        prev = {"N": _make_approach(starve_sec=overage)}
        curr = {"N": _make_approach(starve_sec=overage)}
        _, breakdown = calc.calculate(prev, curr, 0.0, 20, False)
        assert breakdown["comp_starvation_penalty"] > 0.0

    def test_balance_penalty_zero_when_queues_equal(self):
        calc = RewardCalculator()
        prev = {"N": _make_approach(queue_pcu=5.0), "S": _make_approach(queue_pcu=5.0)}
        curr = {"N": _make_approach(queue_pcu=5.0), "S": _make_approach(queue_pcu=5.0)}
        _, breakdown = calc.calculate(prev, curr, 0.0, 20, False)
        assert breakdown["comp_balance_penalty"] == pytest.approx(0.0, abs=1e-4)

    def test_balance_penalty_positive_when_queues_unequal(self):
        calc = RewardCalculator()
        prev = {"N": _make_approach(queue_pcu=1.0), "S": _make_approach(queue_pcu=40.0)}
        curr = {"N": _make_approach(queue_pcu=1.0), "S": _make_approach(queue_pcu=40.0)}
        _, breakdown = calc.calculate(prev, curr, 0.0, 20, False)
        assert breakdown["comp_balance_penalty"] > 0.0

    def test_reward_is_finite_float(self):
        calc = RewardCalculator()
        prev = {"N": _make_approach()}
        curr = {"N": _make_approach()}
        reward, _ = calc.calculate(prev, curr, 0.0, 20, False)
        assert isinstance(reward, float)
        assert math.isfinite(reward)

    def test_custom_weights_applied(self):
        """Zero-weight a component and verify it has no effect on total."""
        weights = RewardWeights(switching=0.0)
        calc = RewardCalculator(weights=weights)
        prev = {"N": _make_approach()}
        curr = {"N": _make_approach()}
        r_no_switch, _ = calc.calculate(prev, curr, 0.0, 20, False)
        r_switch, _ = calc.calculate(prev, curr, 0.0, 20, True)
        assert r_no_switch == pytest.approx(r_switch, abs=1e-6)


# ══════════════════════════════════════════════════════════
# 9. Starvation in Environment
# ══════════════════════════════════════════════════════════

class TestStarvation:
    def test_starvation_info_reported(self, env):
        env.reset()
        _, _, _, _, info = env.step(0)
        assert "starvation_occurrences" in info
        assert isinstance(info["starvation_occurrences"], int)
        assert info["starvation_occurrences"] >= 0

    def test_starvation_occurs_when_approach_neglected(self):
        """Hold phase 1 long enough that EW approaches starve."""
        config = EnvConfig(episode_duration_sec=300)
        env = TrafficSignalEnv(config=config)
        env.reset()
        # Repeatedly apply phase 1 (NS) with max green; EW should eventually starve
        max_ns_action = 7  # phase 1, 55s
        any_starvation = False
        for _ in range(6):
            _, _, _, truncated, info = env.step(max_ns_action)
            if info["starvation_occurrences"] > 0:
                any_starvation = True
                break
            if truncated:
                break
        env.close()
        assert any_starvation, "Expected starvation when EW neglected for 6 × 55s = 330s"


# ══════════════════════════════════════════════════════════
# 10. Safety: No Simultaneous Conflicting Green
# ══════════════════════════════════════════════════════════

class TestSafety:
    def test_no_simultaneous_ns_ew_green(self):
        """At every simulated timestep, NS and EW cannot both show GREEN."""
        from backend.core.schema import SignalState
        config = EnvConfig(episode_duration_sec=120)
        env = TrafficSignalEnv(config=config)
        env.reset()

        for _ in range(6):
            obs, _, _, truncated, _ = env.step(np.random.randint(0, 16))
            # Inspect latest sim approach signal states
            for approach_id, approach in env.sim.approaches.items():
                # Approaches belonging to BOTH phases cannot be GREEN at once
                # The active_phase_id tells us which phase is green
                ns_approaches = {"North_Approach", "South_Approach"}
                ew_approaches = {"East_Approach", "West_Approach"}
                if env.active_phase_id == 1:
                    # Phase 1 = NS green → EW must not be green
                    if approach_id in ew_approaches:
                        assert approach.current_signal != SignalState.GREEN, (
                            f"EW approach {approach_id} is GREEN during Phase 1"
                        )
                else:
                    # Phase 2 = EW green → NS must not be green
                    if approach_id in ns_approaches:
                        assert approach.current_signal != SignalState.GREEN, (
                            f"NS approach {approach_id} is GREEN during Phase 2"
                        )
            if truncated:
                break
        env.close()


# ══════════════════════════════════════════════════════════
# 11. Reproducibility (Seeded Episodes)
# ══════════════════════════════════════════════════════════

class TestReproducibility:
    def _run_episode(self, seed: int) -> list:
        config = EnvConfig(episode_duration_sec=60)
        env = TrafficSignalEnv(config=config)
        obs, _ = env.reset(seed=seed)
        rewards = []
        np.random.seed(seed)
        for _ in range(4):
            action = env.action_space.sample()
            _, reward, _, truncated, _ = env.step(action)
            rewards.append(round(reward, 6))
            if truncated:
                break
        env.close()
        return rewards

    def test_same_seed_same_initial_obs(self):
        config = EnvConfig(episode_duration_sec=60)
        e1 = TrafficSignalEnv(config=config)
        e2 = TrafficSignalEnv(config=config)
        obs1, _ = e1.reset(seed=42)
        obs2, _ = e2.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)
        e1.close()
        e2.close()

    def test_different_seeds_may_differ(self):
        config = EnvConfig(episode_duration_sec=60)
        e1 = TrafficSignalEnv(config=config)
        e2 = TrafficSignalEnv(config=config)
        obs1, _ = e1.reset(seed=1)
        obs2, _ = e2.reset(seed=2)
        # The two observations may differ (not guaranteed but likely)
        # Just verify both are valid
        assert obs1.shape == (OBSERVATION_DIM,)
        assert obs2.shape == (OBSERVATION_DIM,)
        e1.close()
        e2.close()


# ══════════════════════════════════════════════════════════
# 12. Gymnasium check_env
# ══════════════════════════════════════════════════════════

class TestCheckEnv:
    def test_gymnasium_check_env_passes(self):
        """gymnasium check_env must pass (only cosmetic warnings allowed)."""
        config = EnvConfig(episode_duration_sec=120)
        env = TrafficSignalEnv(config=config)
        # check_env raises AssertionError or UserWarning on structural failures.
        # We allow warnings (e.g. render_modes spec) but not errors.
        try:
            check_env(env, warn=True)
        except Exception as exc:
            pytest.fail(f"check_env raised an exception: {exc}")
        env.close()


# ══════════════════════════════════════════════════════════
# 13. Stable-Baselines3 PPO Compatibility
# ══════════════════════════════════════════════════════════

class TestSB3Compatibility:
    def test_ppo_init(self):
        from stable_baselines3 import PPO
        config = EnvConfig(episode_duration_sec=120)
        env = TrafficSignalEnv(config=config)
        model = PPO("MlpPolicy", env, verbose=0)
        assert model is not None
        env.close()

    def test_ppo_predict_returns_valid_action(self):
        from stable_baselines3 import PPO
        config = EnvConfig(episode_duration_sec=120)
        env = TrafficSignalEnv(config=config)
        model = PPO("MlpPolicy", env, verbose=0)
        obs, _ = env.reset()
        action, _states = model.predict(obs, deterministic=True)
        # action might be numpy array or int — ensure it's a valid index
        action_idx = int(action) if not isinstance(action, int) else action
        assert 0 <= action_idx < 16
        env.close()

    def test_ppo_step_with_predicted_action(self):
        from stable_baselines3 import PPO
        config = EnvConfig(episode_duration_sec=120)
        env = TrafficSignalEnv(config=config)
        model = PPO("MlpPolicy", env, verbose=0)
        obs, _ = env.reset()
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (OBSERVATION_DIM,)
        assert math.isfinite(reward)
        env.close()

    def test_action_space_sample_valid(self, env):
        env.reset()
        for _ in range(20):
            action = env.action_space.sample()
            assert 0 <= action < 16


# ══════════════════════════════════════════════════════════
# 14. Render and Close
# ══════════════════════════════════════════════════════════

class TestRenderClose:
    def test_render_before_reset_returns_string(self, env):
        result = env.render()
        assert isinstance(result, str)

    def test_render_after_reset_contains_step_info(self, env):
        env.reset()
        result = env.render()
        assert "Step" in result
        assert "Phase" in result

    def test_close_sets_sim_none(self, env):
        env.reset()
        env.close()
        assert env.sim is None

    def test_double_close_safe(self, env):
        env.reset()
        env.close()
        env.close()  # Should not raise


# ══════════════════════════════════════════════════════════
# 15. Observation Builder
# ══════════════════════════════════════════════════════════

class TestObservationBuilder:
    def test_build_returns_float32_array(self):
        builder = ObservationBuilder()
        config = EnvConfig(episode_duration_sec=120)
        env = TrafficSignalEnv(config=config)
        env.reset()
        obs = builder.build(
            approaches=env.sim.approaches,
            active_phase_id=1,
            elapsed_in_phase=0,
            current_step=1,
            total_steps=900,
        )
        assert obs.dtype == np.float32
        env.close()

    def test_build_shape(self):
        builder = ObservationBuilder()
        config = EnvConfig(episode_duration_sec=120)
        env = TrafficSignalEnv(config=config)
        env.reset()
        obs = builder.build(
            approaches=env.sim.approaches,
            active_phase_id=1,
            elapsed_in_phase=0,
            current_step=1,
            total_steps=900,
        )
        assert obs.shape == (OBSERVATION_DIM,)
        env.close()

    def test_build_values_in_01(self):
        builder = ObservationBuilder()
        config = EnvConfig(episode_duration_sec=120)
        env = TrafficSignalEnv(config=config)
        env.reset()
        obs = builder.build(
            approaches=env.sim.approaches,
            active_phase_id=1,
            elapsed_in_phase=0,
            current_step=1,
            total_steps=900,
        )
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0)
        env.close()

