"""Generate and compile an N x N SUMO grid with a traffic light at each node."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from generate_traffic import generate_routefile

SUMO_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SUMO_DIR.parent


def _tl_id(row: int, col: int, grid_size: int) -> str:
    return f"TL_{row * grid_size + col}"


def write_grid_files(grid_size: int = 5, spacing: float = 200.0) -> None:
    """Write the node and directed edge XML files for a square grid.

    Every signalized intersection has exactly four incoming approaches. At the
    outside border, a missing grid neighbor is replaced by an external entry /
    exit road. This also keeps the same four-approach signal structure at edge
    and corner intersections as at interior intersections.
    """
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    if spacing <= 0:
        raise ValueError("spacing must be positive")

    node_root = ET.Element("nodes")
    edge_root = ET.Element("edges")
    connection_root = ET.Element("connections")
    directions = {
        "N": (-1, 0),
        "E": (0, 1),
        "S": (1, 0),
        "W": (0, -1),
    }
    external_offset = spacing * 0.75

    def add_node(node_id: str, x: float, y: float, node_type: str) -> None:
        ET.SubElement(
            node_root,
            "node",
            {"id": node_id, "x": f"{x:g}", "y": f"{y:g}", "type": node_type},
        )

    def add_edge(edge_id: str, source: str, target: str) -> None:
        ET.SubElement(
            edge_root,
            "edge",
            {
                "id": edge_id,
                "from": source,
                "to": target,
                "numLanes": "2",
                "speed": "13.89",
            },
        )

    def incoming_edge(row: int, col: int, direction: str) -> str:
        dr, dc = directions[direction]
        nr, nc = row + dr, col + dc
        tl = _tl_id(row, col, grid_size)
        if 0 <= nr < grid_size and 0 <= nc < grid_size:
            source = _tl_id(nr, nc, grid_size)
            return f"edge_{source}_to_{tl}"
        return f"in_{tl}_{direction}"

    def outgoing_edge(row: int, col: int, direction: str) -> str:
        dr, dc = directions[direction]
        nr, nc = row + dr, col + dc
        tl = _tl_id(row, col, grid_size)
        if 0 <= nr < grid_size and 0 <= nc < grid_size:
            target = _tl_id(nr, nc, grid_size)
            return f"edge_{tl}_to_{target}"
        return f"out_{tl}_{direction}"

    # Row 0 is the northernmost row, and columns increase eastward.
    for row in range(grid_size):
        for col in range(grid_size):
            x = col * spacing
            y = (grid_size - 1 - row) * spacing
            add_node(_tl_id(row, col, grid_size), x, y, "traffic_light")

    # One external node and one in/out edge pair for every missing neighbor.
    for row in range(grid_size):
        for col in range(grid_size):
            tl = _tl_id(row, col, grid_size)
            x = col * spacing
            y = (grid_size - 1 - row) * spacing
            for direction, (dr, dc) in directions.items():
                nr, nc = row + dr, col + dc
                if 0 <= nr < grid_size and 0 <= nc < grid_size:
                    neighbor = _tl_id(nr, nc, grid_size)
                    add_edge(f"edge_{tl}_to_{neighbor}", tl, neighbor)
                    continue

                external_id = f"EXT_{tl}_{direction}"
                external_x = x + dc * external_offset
                external_y = y - dr * external_offset
                add_node(external_id, external_x, external_y, "priority")
                add_edge(f"in_{tl}_{direction}", external_id, tl)
                add_edge(f"out_{tl}_{direction}", tl, external_id)

    # Recreate the same two-lane turning pattern as the original 2x2 project:
    # lane 1 serves the left turn; lane 0 serves straight and right turns.
    # SUMO derives each junction's signal phase program from these connections.
    left_turn = {"N": "E", "E": "S", "S": "W", "W": "N"}
    straight = {"N": "S", "E": "W", "S": "N", "W": "E"}
    right_turn = {"N": "W", "E": "N", "S": "E", "W": "S"}
    for row in range(grid_size):
        for col in range(grid_size):
            for approach in ("N", "E", "S", "W"):
                source_edge = incoming_edge(row, col, approach)
                for destination_direction, from_lane, to_lane in (
                    (left_turn[approach], "1", "1"),
                    (straight[approach], "0", "0"),
                    (right_turn[approach], "0", "0"),
                ):
                    ET.SubElement(
                        connection_root,
                        "connection",
                        {
                            "from": source_edge,
                            "to": outgoing_edge(row, col, destination_direction),
                            "fromLane": from_lane,
                            "toLane": to_lane,
                        },
                    )

    for root, filename, schema in (
        (node_root, SUMO_DIR / "square_grid.nod.xml", "http://sumo.dlr.de/xsd/nodes_file.xsd"),
        (edge_root, SUMO_DIR / "square_grid.edg.xml", "http://sumo.dlr.de/xsd/edges_file.xsd"),
        (connection_root, SUMO_DIR / "square_grid.con.xml", "http://sumo.dlr.de/xsd/connections_file.xsd"),
    ):
        ET.register_namespace("xsi", "http://www.w3.org/2001/XMLSchema-instance")
        root.set("{http://www.w3.org/2001/XMLSchema-instance}noNamespaceSchemaLocation", schema)
        tree = ET.ElementTree(root)
        ET.indent(tree, space="    ")
        tree.write(filename, encoding="utf-8", xml_declaration=True)

    tls_count = grid_size * grid_size
    external_approaches = 4 * grid_size
    print(f"Wrote {tls_count} traffic-light nodes and {external_approaches} boundary approaches.")
    print(f"Nodes: {SUMO_DIR / 'square_grid.nod.xml'}")
    print(f"Edges: {SUMO_DIR / 'square_grid.edg.xml'}")
    print(f"Connections: {SUMO_DIR / 'square_grid.con.xml'}")


def _find_netconvert() -> str:
    executable_name = "netconvert.exe" if os.name == "nt" else "netconvert"
    located = shutil.which(executable_name) or shutil.which("netconvert")
    if located:
        return located

    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        candidate = Path(sumo_home) / "bin" / executable_name
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        "Could not find netconvert. Install SUMO and add its bin directory to PATH, "
        "or set the SUMO_HOME environment variable."
    )


def validate_signal_programs(net_file: Path, grid_size: int) -> None:
    """Check that all signalized nodes have the phase indexes expected by env.py."""
    tree = ET.parse(net_file)
    root = tree.getroot()
    tls_by_id = {tls.get("id"): tls for tls in root.findall("tlLogic")}
    expected_ids = {f"TL_{i}" for i in range(grid_size * grid_size)}
    missing = sorted(expected_ids - set(tls_by_id))
    if missing:
        raise RuntimeError(
            f"netconvert did not create traffic-light programs for: {', '.join(missing[:10])}"
        )

    invalid = []
    for tls_id in sorted(expected_ids):
        phases = tls_by_id[tls_id].findall("phase")
        # env.py controls phases 0/1 (NS green/yellow) and 4/5 (EW green/yellow).
        if len(phases) < 6:
            invalid.append(f"{tls_id}: only {len(phases)} phases")
            continue
        states = [phase.get("state", "") for phase in phases]
        if not any(ch in "Gg" for ch in states[0]) or not any(ch in "yY" for ch in states[1]):
            invalid.append(f"{tls_id}: phases 0/1 do not look like green/yellow")
        if not any(ch in "Gg" for ch in states[4]) or not any(ch in "yY" for ch in states[5]):
            invalid.append(f"{tls_id}: phases 4/5 do not look like green/yellow")
    if invalid:
        details = "; ".join(invalid[:8])
        raise RuntimeError(
            "The generated signal phase order may not match env.py's phase_map. "
            f"Check the generated network before training. Details: {details}"
        )


def build(grid_size: int = 5, max_steps: int = 3600) -> None:
    """Generate the grid inputs, compile the SUMO network, and create routes."""
    write_grid_files(grid_size=grid_size)
    node_file = SUMO_DIR / "square_grid.nod.xml"
    edge_file = SUMO_DIR / "square_grid.edg.xml"
    connection_file = SUMO_DIR / "square_grid.con.xml"
    net_file = SUMO_DIR / "square_grid.net.xml"

    print("\nCompiling SUMO network via netconvert...")
    cmd = [
        _find_netconvert(),
        "--node-files", str(node_file),
        "--edge-files", str(edge_file),
        "--connection-files", str(connection_file),
        "--output-file", str(net_file),
        "--tls.default-type", "static",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "netconvert failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip())

    validate_signal_programs(net_file, grid_size)
    route_file = SUMO_DIR / "square_grid.rou.xml"
    print(f"\nGenerating traffic routes for the {grid_size}x{grid_size} network...")
    generate_routefile(
        filename=str(route_file),
        max_steps=max_steps,
        grid_size=grid_size,
    )
    print("\nNetwork build complete.")
    print(f"  Intersections: {grid_size * grid_size}")
    print(f"  Network:       {net_file}")
    print(f"  Routes:        {route_file}")
    print("To open the simulation GUI, run from the project root:")
    print("  sumo-gui -c sumo_env/square_grid.sumocfg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=5, help="number of rows/columns (default: 5)")
    parser.add_argument("--max-steps", type=int, default=3600, help="traffic generation horizon in seconds")
    args = parser.parse_args()
    try:
        build(grid_size=args.grid_size, max_steps=args.max_steps)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
