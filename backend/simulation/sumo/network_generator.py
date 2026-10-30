"""
SUMO Network Generator for Alankar Chowk.

Generates SUMO network XML files for a realistic 4-way intersection:
- Nodes (junction center + 4 perimeter boundaries)
- Edges (incoming and outgoing dual-lane approaches)
- Connections (through, left, right turns with lane mappings)
- Traffic light logic (2-phase dual-arterial control matching RL action space)
- Pre-compiled .net.xml and .sumocfg configuration
"""

import logging
import os
from pathlib import Path
import shutil
import subprocess

logger = logging.getLogger(__name__)


class NetworkGenerator:
    """Creates all required SUMO XML network definitions."""

    def __init__(self, output_dir: Path = None):
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).parent
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all(self, net_filename: str = "alankar_chowk.net.xml") -> Path:
        """Generates all XML network artifacts and compiles or writes .net.xml."""
        nod_path = self._write_nodes()
        edg_path = self._write_edges()
        con_path = self._write_connections()
        tll_path = self._write_traffic_lights()
        cfg_path = self._write_sumocfg()

        net_path = self.output_dir / net_filename

        # Attempt compilation via netconvert if installed
        netconvert_bin = shutil.which("netconvert") or shutil.which("netconvert.exe")
        if netconvert_bin:
            try:
                cmd = [
                    netconvert_bin,
                    "--node-files", str(nod_path),
                    "--edge-files", str(edg_path),
                    "--connection-files", str(con_path),
                    "--tllogic-files", str(tll_path),
                    "--output-file", str(net_path),
                    "--no-turnarounds", "true",
                    "--default.lanenumber", "2",
                    "--default.speed", "13.89",
                ]
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                logger.info(f"Successfully compiled SUMO network with netconvert: {net_path}")
                return net_path
            except Exception as e:
                logger.warning(f"netconvert failed ({e}). Writing direct compliant .net.xml.")

        # Fallback: write self-contained valid .net.xml directly
        self._write_direct_net_xml(net_path)
        logger.info(f"Generated standalone compliant SUMO network: {net_path}")
        return net_path

    def _write_nodes(self) -> Path:
        path = self.output_dir / "alankar_chowk.nod.xml"
        content = """<nodes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/nodes_file.xsd">
    <node id="J_center" x="0.0" y="0.0" type="traffic_light" tl="J_center"/>
    <node id="N" x="0.0" y="150.0" type="priority"/>
    <node id="S" x="0.0" y="-150.0" type="priority"/>
    <node id="E" x="150.0" y="0.0" type="priority"/>
    <node id="W" x="-150.0" y="0.0" type="priority"/>
</nodes>"""
        path.write_text(content, encoding="utf-8")
        return path

    def _write_edges(self) -> Path:
        path = self.output_dir / "alankar_chowk.edg.xml"
        content = """<edges xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/edges_file.xsd">
    <!-- North Approach -->
    <edge id="N2C" from="N" to="J_center" numLanes="2" speed="13.89" priority="2"/>
    <edge id="C2N" from="J_center" to="N" numLanes="2" speed="13.89" priority="2"/>
    <!-- South Approach -->
    <edge id="S2C" from="S" to="J_center" numLanes="2" speed="13.89" priority="2"/>
    <edge id="C2S" from="J_center" to="S" numLanes="2" speed="13.89" priority="2"/>
    <!-- East Approach -->
    <edge id="E2C" from="E" to="J_center" numLanes="2" speed="13.89" priority="2"/>
    <edge id="C2E" from="J_center" to="E" numLanes="2" speed="13.89" priority="2"/>
    <!-- West Approach -->
    <edge id="W2C" from="W" to="J_center" numLanes="2" speed="13.89" priority="2"/>
    <edge id="C2W" from="J_center" to="W" numLanes="2" speed="13.89" priority="2"/>
</edges>"""
        path.write_text(content, encoding="utf-8")
        return path

    def _write_connections(self) -> Path:
        path = self.output_dir / "alankar_chowk.con.xml"
        content = """<connections xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/connections_file.xsd">
    <!-- North Incoming Connections -->
    <connection from="N2C" to="C2W" fromLane="0" toLane="0"/>
    <connection from="N2C" to="C2S" fromLane="1" toLane="1"/>
    <connection from="N2C" to="C2E" fromLane="1" toLane="1"/>

    <!-- South Incoming Connections -->
    <connection from="S2C" to="C2E" fromLane="0" toLane="0"/>
    <connection from="S2C" to="C2N" fromLane="1" toLane="1"/>
    <connection from="S2C" to="C2W" fromLane="1" toLane="1"/>

    <!-- East Incoming Connections -->
    <connection from="E2C" to="C2N" fromLane="0" toLane="0"/>
    <connection from="E2C" to="C2W" fromLane="1" toLane="1"/>
    <connection from="E2C" to="C2S" fromLane="1" toLane="1"/>

    <!-- West Incoming Connections -->
    <connection from="W2C" to="C2S" fromLane="0" toLane="0"/>
    <connection from="W2C" to="C2E" fromLane="1" toLane="1"/>
    <connection from="W2C" to="C2N" fromLane="1" toLane="1"/>
</connections>"""
        path.write_text(content, encoding="utf-8")
        return path

    def _write_traffic_lights(self) -> Path:
        path = self.output_dir / "alankar_chowk.tll.xml"
        content = """<tlLogics xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/tllogic_file.xsd">
    <tlLogic id="J_center" type="static" programID="2_phase_rl" offset="0">
        <!-- Phase 1: North-South Green (duration default 25s) -->
        <phase duration="25" state="GGGGrrrrGGGGrrrr"/>
        <!-- Phase 1 Yellow Clearance (3s) -->
        <phase duration="3"  state="yyyyrrrryyyyrrrr"/>
        <!-- All-Red Clearance (2s) -->
        <phase duration="2"  state="rrrrrrrrrrrrrrrr"/>
        <!-- Phase 2: East-West Green (duration default 25s) -->
        <phase duration="25" state="rrrrGGGGrrrrGGGG"/>
        <!-- Phase 2 Yellow Clearance (3s) -->
        <phase duration="3"  state="rrrryyyyrrrryyyy"/>
        <!-- All-Red Clearance (2s) -->
        <phase duration="2"  state="rrrrrrrrrrrrrrrr"/>
    </tlLogic>
</tlLogics>"""
        path.write_text(content, encoding="utf-8")
        return path

    def _write_sumocfg(self) -> Path:
        path = self.output_dir / "alankar_chowk.sumocfg"
        content = """<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="alankar_chowk.net.xml"/>
        <route-files value="alankar_chowk.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <step-length value="1.0"/>
    </time>
    <processing>
        <time-to-teleport value="-1"/>
        <waiting-time-memory value="1000"/>
    </processing>
    <report>
        <verbose value="false"/>
        <no-step-log value="true"/>
        <no-warnings value="true"/>
    </report>
</configuration>"""
        path.write_text(content, encoding="utf-8")
        return path

    def _write_direct_net_xml(self, path: Path):
        """Constructs a fully formed standalone SUMO .net.xml file directly."""
        content = """<?xml version="1.0" encoding="UTF-8"?>
<net version="1.20" junctionCornerDetail="5" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/net_file.xsd">
    <location netOffset="0.00,0.00" convBoundary="-150.00,-150.00,150.00,150.00" origBoundary="-150.00,-150.00,150.00,150.00" projParameter="!"/>

    <!-- Incoming and outgoing edges with dual lanes -->
    <edge id="N2C" from="N" to="J_center" priority="2">
        <lane id="N2C_0" index="0" speed="13.89" length="140.0" shape="1.6,150.0 1.6,10.0"/>
        <lane id="N2C_1" index="1" speed="13.89" length="140.0" shape="4.8,150.0 4.8,10.0"/>
    </edge>
    <edge id="C2N" from="J_center" to="N" priority="2">
        <lane id="C2N_0" index="0" speed="13.89" length="140.0" shape="-4.8,10.0 -4.8,150.0"/>
        <lane id="C2N_1" index="1" speed="13.89" length="140.0" shape="-1.6,10.0 -1.6,150.0"/>
    </edge>

    <edge id="S2C" from="S" to="J_center" priority="2">
        <lane id="S2C_0" index="0" speed="13.89" length="140.0" shape="-1.6,-150.0 -1.6,-10.0"/>
        <lane id="S2C_1" index="1" speed="13.89" length="140.0" shape="-4.8,-150.0 -4.8,-10.0"/>
    </edge>
    <edge id="C2S" from="J_center" to="S" priority="2">
        <lane id="C2S_0" index="0" speed="13.89" length="140.0" shape="4.8,-10.0 4.8,-150.0"/>
        <lane id="C2S_1" index="1" speed="13.89" length="140.0" shape="1.6,-10.0 1.6,-150.0"/>
    </edge>

    <edge id="E2C" from="E" to="J_center" priority="2">
        <lane id="E2C_0" index="0" speed="13.89" length="140.0" shape="150.0,-1.6 10.0,-1.6"/>
        <lane id="E2C_1" index="1" speed="13.89" length="140.0" shape="150.0,-4.8 10.0,-4.8"/>
    </edge>
    <edge id="C2E" from="J_center" to="E" priority="2">
        <lane id="C2E_0" index="0" speed="13.89" length="140.0" shape="10.0,4.8 150.0,4.8"/>
        <lane id="C2E_1" index="1" speed="13.89" length="140.0" shape="10.0,1.6 150.0,1.6"/>
    </edge>

    <edge id="W2C" from="W" to="J_center" priority="2">
        <lane id="W2C_0" index="0" speed="13.89" length="140.0" shape="-150.0,1.6 -10.0,1.6"/>
        <lane id="W2C_1" index="1" speed="13.89" length="140.0" shape="-150.0,4.8 -10.0,4.8"/>
    </edge>
    <edge id="C2W" from="J_center" to="W" priority="2">
        <lane id="C2W_0" index="0" speed="13.89" length="140.0" shape="-10.0,-4.8 -150.0,-4.8"/>
        <lane id="C2W_1" index="1" speed="13.89" length="140.0" shape="-10.0,-1.6 -150.0,-1.6"/>
    </edge>

    <tlLogic id="J_center" type="static" programID="2_phase_rl" offset="0">
        <phase duration="25" state="GGGGrrrrGGGGrrrr"/>
        <phase duration="3"  state="yyyyrrrryyyyrrrr"/>
        <phase duration="2"  state="rrrrrrrrrrrrrrrr"/>
        <phase duration="25" state="rrrrGGGGrrrrGGGG"/>
        <phase duration="3"  state="rrrryyyyrrrryyyy"/>
        <phase duration="2"  state="rrrrrrrrrrrrrrrr"/>
    </tlLogic>

    <junction id="J_center" type="traffic_light" x="0.00" y="0.00" incLanes="N2C_0 N2C_1 S2C_0 S2C_1 E2C_0 E2C_1 W2C_0 W2C_1" intLanes="" shape="-10.00,10.00 10.00,10.00 10.00,-10.00 -10.00,-10.00"/>
    <junction id="N" type="priority" x="0.00" y="150.00" incLanes="C2N_0 C2N_1" intLanes="" shape="-6.40,150.00 6.40,150.00"/>
    <junction id="S" type="priority" x="0.00" y="-150.00" incLanes="C2S_0 C2S_1" intLanes="" shape="6.40,-150.00 -6.40,-150.00"/>
    <junction id="E" type="priority" x="150.00" y="0.00" incLanes="C2E_0 C2E_1" intLanes="" shape="150.00,6.40 150.00,-6.40"/>
    <junction id="W" type="priority" x="-150.00" y="0.00" incLanes="C2W_0 C2W_1" intLanes="" shape="-150.00,-6.40 -150.00,6.40"/>

    <connection from="N2C" to="C2S" fromLane="0" toLane="0" tl="J_center" linkIndex="0" dir="s" state="M"/>
    <connection from="N2C" to="C2W" fromLane="0" toLane="0" tl="J_center" linkIndex="1" dir="r" state="M"/>
    <connection from="N2C" to="C2E" fromLane="1" toLane="1" tl="J_center" linkIndex="2" dir="l" state="M"/>
    <connection from="N2C" to="C2S" fromLane="1" toLane="1" tl="J_center" linkIndex="3" dir="s" state="M"/>

    <connection from="S2C" to="C2N" fromLane="0" toLane="0" tl="J_center" linkIndex="4" dir="s" state="M"/>
    <connection from="S2C" to="C2E" fromLane="0" toLane="0" tl="J_center" linkIndex="5" dir="r" state="M"/>
    <connection from="S2C" to="C2W" fromLane="1" toLane="1" tl="J_center" linkIndex="6" dir="l" state="M"/>
    <connection from="S2C" to="C2N" fromLane="1" toLane="1" tl="J_center" linkIndex="7" dir="s" state="M"/>

    <connection from="E2C" to="C2W" fromLane="0" toLane="0" tl="J_center" linkIndex="8" dir="s" state="M"/>
    <connection from="E2C" to="C2N" fromLane="0" toLane="0" tl="J_center" linkIndex="9" dir="r" state="M"/>
    <connection from="E2C" to="C2S" fromLane="1" toLane="1" tl="J_center" linkIndex="10" dir="l" state="M"/>
    <connection from="E2C" to="C2W" fromLane="1" toLane="1" tl="J_center" linkIndex="11" dir="s" state="M"/>

    <connection from="W2C" to="C2E" fromLane="0" toLane="0" tl="J_center" linkIndex="12" dir="s" state="M"/>
    <connection from="W2C" to="C2S" fromLane="0" toLane="0" tl="J_center" linkIndex="13" dir="r" state="M"/>
    <connection from="W2C" to="C2N" fromLane="1" toLane="1" tl="J_center" linkIndex="14" dir="l" state="M"/>
    <connection from="W2C" to="C2E" fromLane="1" toLane="1" tl="J_center" linkIndex="15" dir="s" state="M"/>
</net>"""
        path.write_text(content, encoding="utf-8")


network_generator = NetworkGenerator()

