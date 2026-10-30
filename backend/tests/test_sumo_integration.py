"""
Unit and integration tests for SUMO and TraCI integration, network generation,
demand processing, and SimulationAdapter lifecycle.
"""

from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import pytest

from backend.core.schema import ControllerDecision, ControllerMode, SignalState
from backend.simulation.junction_layout import JunctionLayout
from backend.simulation.sumo.network_generator import NetworkGenerator
from backend.simulation.sumo.demand_generator import DemandGenerator, VEHICLE_TYPES
from backend.rl.sumo_adapter import SimulationAdapter, is_sumo_available
from backend.rl.environment import TrafficSignalEnv
from backend.rl.config import EnvConfig


# ──────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────

@pytest.fixture
def tmp_sumo_dir(tmp_path):
    d = tmp_path / "sumo"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def sample_arrivals_df():
    return pd.DataFrame([
        {"Time": "09:00:00", "Direction": "UP", "car_arrivals": 2, "motorbike_arrivals": 3, "bus_arrivals": 1, "truck_arrivals": 0},
        {"Time": "09:00:01", "Direction": "DOWN", "car_arrivals": 1, "motorbike_arrivals": 2, "bus_arrivals": 0, "truck_arrivals": 1},
        {"Time": "09:00:02", "Direction": "RIGHT", "car_arrivals": 3, "motorbike_arrivals": 4, "bus_arrivals": 0, "truck_arrivals": 0},
        {"Time": "09:00:03", "Direction": "LEFT", "car_arrivals": 1, "motorbike_arrivals": 1, "bus_arrivals": 1, "truck_arrivals": 0},
    ])


# ──────────────────────────────────────────────────────────
# 1. Network Generator Tests
# ──────────────────────────────────────────────────────────

class TestNetworkGenerator:
    def test_network_generation_produces_valid_xmls(self, tmp_sumo_dir):
        gen = NetworkGenerator(output_dir=tmp_sumo_dir)
        net_path = gen.generate_all()

        assert net_path.exists()
        assert (tmp_sumo_dir / "alankar_chowk.nod.xml").exists()
        assert (tmp_sumo_dir / "alankar_chowk.edg.xml").exists()
        assert (tmp_sumo_dir / "alankar_chowk.con.xml").exists()
        assert (tmp_sumo_dir / "alankar_chowk.tll.xml").exists()
        assert (tmp_sumo_dir / "alankar_chowk.sumocfg").exists()

        # Parse nodes XML
        nod_tree = ET.parse(tmp_sumo_dir / "alankar_chowk.nod.xml")
        nodes = {n.get("id") for n in nod_tree.getroot().findall("node")}
        assert {"J_center", "N", "S", "E", "W"}.issubset(nodes)

        # Parse edges XML
        edg_tree = ET.parse(tmp_sumo_dir / "alankar_chowk.edg.xml")
        edges = {e.get("id") for e in edg_tree.getroot().findall("edge")}
        assert {"N2C", "C2N", "S2C", "C2S", "E2C", "C2E", "W2C", "C2W"}.issubset(edges)

        # Parse net.xml
        net_tree = ET.parse(net_path)
        assert net_tree.getroot().tag == "net"


# ──────────────────────────────────────────────────────────
# 2. Demand Generator Tests
# ──────────────────────────────────────────────────────────

class TestDemandGenerator:
    def test_demand_generation_with_arrivals(self, tmp_sumo_dir, sample_arrivals_df):
        gen = DemandGenerator(output_dir=tmp_sumo_dir)
        rou_path = gen.generate_routes_xml(sample_arrivals_df, output_filename="test.rou.xml")

        assert rou_path.exists()
        tree = ET.parse(rou_path)
        root = tree.getroot()
        assert root.tag == "routes"

        # Check vehicle types defined
        vtypes = {vt.get("id") for vt in root.findall("vType")}
        assert {"car", "motorbike", "bus", "truck"}.issubset(vtypes)

        # Check vehicles generated
        vehicles = root.findall("vehicle")
        assert len(vehicles) > 0
        for v in vehicles:
            assert v.get("id") is not None
            assert v.get("type") in VEHICLE_TYPES
            assert float(v.get("depart")) >= 0.0

    def test_demand_generation_empty_df_fallback(self, tmp_sumo_dir):
        gen = DemandGenerator(output_dir=tmp_sumo_dir)
        rou_path = gen.generate_routes_xml(pd.DataFrame(), output_filename="fallback.rou.xml")

        assert rou_path.exists()
        tree = ET.parse(rou_path)
        vehicles = tree.getroot().findall("vehicle")
        assert len(vehicles) > 0, "Fallback baseline traffic should be generated on empty input"


# ──────────────────────────────────────────────────────────
# 3. SimulationAdapter Tests
# ──────────────────────────────────────────────────────────

class TestSimulationAdapter:
    def test_adapter_initialization_junction_sim(self):
        layout = JunctionLayout("AlankarChowk", plan_mode="2_phase")
        adapter = SimulationAdapter(layout=layout, use_sumo=False)

        assert adapter.backend == "junction_sim"
        assert adapter.step_count == 0
        assert adapter.total_steps > 0
        assert len(adapter.approaches) == 4

    def test_adapter_graceful_sumo_fallback_when_unavailable(self):
        layout = JunctionLayout("AlankarChowk", plan_mode="2_phase")
        # Request SUMO explicitly: should either succeed (if installed) or gracefully fallback to junction_sim
        adapter = SimulationAdapter(layout=layout, use_sumo=True)

        assert adapter.backend in ("sumo", "junction_sim")
        assert adapter.total_steps > 0

    def test_adapter_step_produces_valid_observation(self):
        layout = JunctionLayout("AlankarChowk", plan_mode="2_phase")
        adapter = SimulationAdapter(layout=layout, use_sumo=False)

        decision = ControllerDecision(
            step=0,
            timestamp_str="09:00:00",
            mode=ControllerMode.DEEP_RL,
            active_phase_id=1,
            phase_name="Phase 1: North-South Arterial",
            signal_state=SignalState.GREEN,
            time_remaining_in_state=25,
            elapsed_in_phase=0,
            decision_rationale="PPO Green",
        )

        obs = adapter.step(decision)
        assert obs.step == 0
        assert obs.total_active_queue_pcu >= 0.0
        assert obs.total_throughput_pcu >= 0.0
        assert len(obs.approaches) == 4

    def test_adapter_reset_and_close(self):
        layout = JunctionLayout("AlankarChowk", plan_mode="2_phase")
        adapter = SimulationAdapter(layout=layout, use_sumo=False)

        decision = ControllerDecision(
            step=0,
            timestamp_str="09:00:00",
            mode=ControllerMode.DEEP_RL,
            active_phase_id=1,
            phase_name="Phase 1: North-South Arterial",
            signal_state=SignalState.GREEN,
            time_remaining_in_state=25,
            elapsed_in_phase=0,
        )
        adapter.step(decision)
        assert adapter.step_count == 1

        adapter.reset()
        assert adapter.step_count == 0

        adapter.close()
        assert adapter._sim is None


# ──────────────────────────────────────────────────────────
# 4. TrafficSignalEnv with Adapter Integration Tests
# ──────────────────────────────────────────────────────────

class TestTrafficSignalEnvAdapter:
    def test_env_reset_contains_simulator_backend(self):
        config = EnvConfig(episode_duration_sec=60, simulator="junction_sim")
        env = TrafficSignalEnv(config=config)
        _, info = env.reset()

        assert "simulator_backend" in info
        assert info["simulator_backend"] == "junction_sim"
        env.close()

    def test_env_steps_with_adapter(self):
        config = EnvConfig(episode_duration_sec=60, simulator="auto")
        env = TrafficSignalEnv(config=config)
        obs, _ = env.reset()

        assert obs.shape == (21,)
        for action in [0, 8, 15]:
            obs, reward, term, trunc, info = env.step(action)
            assert obs.shape == (21,)
            assert np.all(obs >= 0.0) and np.all(obs <= 1.0)
            assert np.isfinite(reward)
            assert "simulator_backend" in info
            if trunc:
                break
        env.close()

