"""Generate and compile an NxN SUMO grid for the Neighbor-CTDE project.

Run from the project root:
    python sumo_env/build_network.py --grid-size 5

For generating XML inputs without a local SUMO/netconvert installation:
    python sumo_env/build_network.py --grid-size 5 --generate-only
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET

try:
    from .generate_traffic import generate_routefile
except ImportError:  # allows direct execution as: python sumo_env/build_network.py
    from generate_traffic import generate_routefile

SUMO_DIR = Path(__file__).resolve().parent
DIRECTIONS = ("N", "E", "S", "W")


def node_id(index: int) -> str:
    return f"TL_{index}"


def neighbor_index(row: int, col: int, direction: str, grid_size: int):
    if direction == "N" and row > 0:
        return (row - 1) * grid_size + col
    if direction == "E" and col < grid_size - 1:
        return row * grid_size + (col + 1)
    if direction == "S" and row < grid_size - 1:
        return (row + 1) * grid_size + col
    if direction == "W" and col > 0:
        return row * grid_size + (col - 1)
    return None


def incoming_edge(index: int, direction: str, grid_size: int) -> str:
    row, col = divmod(index, grid_size)
    upstream = neighbor_index(row, col, direction, grid_size)
    return f"edge_{upstream}_to_{index}" if upstream is not None else f"in_TL_{index}_{direction}"


def outgoing_edge(index: int, direction: str, grid_size: int) -> str:
    row, col = divmod(index, grid_size)
    downstream = neighbor_index(row, col, direction, grid_size)
    return f"edge_{index}_to_{downstream}" if downstream is not None else f"out_TL_{index}_{direction}"


def generate_grid_files(grid_size: int = 5, spacing: float = 200.0, boundary_offset: float = 150.0, lanes: int = 2, speed: float = 13.89):
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")

    nodes_root = ET.Element("nodes", {
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/nodes_file.xsd",
    })
    edges_root = ET.Element("edges", {
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/edges_file.xsd",
    })
    connections_root = ET.Element("connections", {
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/connections_file.xsd",
    })

    # Main signalized intersections. Row 0 is the northernmost row.
    for row in range(grid_size):
        for col in range(grid_size):
            index = row * grid_size + col
            x = col * spacing
            y = (grid_size - 1 - row) * spacing
            ET.SubElement(nodes_root, "node", {
                "id": node_id(index), "x": f"{x:.2f}", "y": f"{y:.2f}", "type": "traffic_light"
            })

    # Bidirectional roads between horizontal and vertical grid neighbors.
    for row in range(grid_size):
        for col in range(grid_size):
            source = row * grid_size + col
            for direction in ("E", "S"):
                destination = neighbor_index(row, col, direction, grid_size)
                if destination is None:
                    continue
                ET.SubElement(edges_root, "edge", {
                    "id": f"edge_{source}_to_{destination}", "from": node_id(source), "to": node_id(destination),
                    "numLanes": str(lanes), "speed": str(speed)
                })
                ET.SubElement(edges_root, "edge", {
                    "id": f"edge_{destination}_to_{source}", "from": node_id(destination), "to": node_id(source),
                    "numLanes": str(lanes), "speed": str(speed)
                })

    # Ensure every signal has four approaches: internal roads where present,
    # and external entry/exit roads on the perimeter where a neighbor is absent.
    for row in range(grid_size):
        for col in range(grid_size):
            index = row * grid_size + col
            x = col * spacing
            y = (grid_size - 1 - row) * spacing
            for direction in DIRECTIONS:
                if neighbor_index(row, col, direction, grid_size) is not None:
                    continue
                if direction == "N":
                    bx, by = x, y + boundary_offset
                elif direction == "E":
                    bx, by = x + boundary_offset, y
                elif direction == "S":
                    bx, by = x, y - boundary_offset
                else:
                    bx, by = x - boundary_offset, y
                boundary_node = f"boundary_TL_{index}_{direction}"
                ET.SubElement(nodes_root, "node", {
                    "id": boundary_node, "x": f"{bx:.2f}", "y": f"{by:.2f}", "type": "priority"
                })
                ET.SubElement(edges_root, "edge", {
                    "id": f"in_TL_{index}_{direction}", "from": boundary_node, "to": node_id(index),
                    "numLanes": str(lanes), "speed": str(speed)
                })
                ET.SubElement(edges_root, "edge", {
                    "id": f"out_TL_{index}_{direction}", "from": node_id(index), "to": boundary_node,
                    "numLanes": str(lanes), "speed": str(speed)
                })

    # Use the same two-lane movement convention as the original project:
    # lane 0 supports right/straight, lane 1 supports the left turn.
    target_direction = {
        "N": {"right": "W", "straight": "S", "left": "E"},
        "E": {"right": "N", "straight": "W", "left": "S"},
        "S": {"right": "E", "straight": "N", "left": "W"},
        "W": {"right": "S", "straight": "E", "left": "N"},
    }
    for row in range(grid_size):
        for col in range(grid_size):
            index = row * grid_size + col
            for approach in DIRECTIONS:
                from_edge = incoming_edge(index, approach, grid_size)
                # Keep consistent per-junction connection order for generated TLS programs.
                movements = (("right", 0, 0), ("straight", 0, 0), ("left", 1, 1))
                for movement, from_lane, to_lane in movements:
                    destination_direction = target_direction[approach][movement]
                    to_edge = outgoing_edge(index, destination_direction, grid_size)
                    ET.SubElement(connections_root, "connection", {
                        "from": from_edge,
                        "to": to_edge,
                        "fromLane": str(from_lane),
                        "toLane": str(to_lane),
                    })

    def write_xml(root, name):
        tree = ET.ElementTree(root)
        ET.indent(tree, space="    ")
        tree.write(SUMO_DIR / name, encoding="utf-8", xml_declaration=True)

    write_xml(nodes_root, "square_grid.nod.xml")
    write_xml(edges_root, "square_grid.edg.xml")
    write_xml(connections_root, "square_grid.con.xml")

    return {
        "nodes": len(nodes_root.findall("node")),
        "traffic_lights": grid_size * grid_size,
        "directed_internal_edges": sum(1 for e in edges_root.findall("edge") if e.get("id", "").startswith("edge_")),
        "directed_boundary_edges": sum(1 for e in edges_root.findall("edge") if not e.get("id", "").startswith("edge_")),
        "connections": len(connections_root.findall("connection")),
    }


def build(grid_size: int = 5, max_steps: int = 3600, ev_probability: float = 0.04, spawn_probability: float = 0.35, generate_only: bool = False):
    print(f"Generating {grid_size}x{grid_size} SUMO grid inputs...")
    stats = generate_grid_files(grid_size=grid_size)
    print(
        f"Generated {stats['traffic_lights']} traffic lights, "
        f"{stats['directed_internal_edges']} directed internal edges, "
        f"{stats['directed_boundary_edges']} directed boundary edges, "
        f"{stats['connections']} turning connections."
    )

    generate_routefile(
        str(SUMO_DIR / "square_grid.rou.xml"),
        max_steps=max_steps,
        ev_probability=ev_probability,
        spawn_probability=spawn_probability,
        grid_size=grid_size,
    )

    if generate_only:
        print("XML inputs generated only. Run without --generate-only on a machine with SUMO/netconvert to compile the .net.xml file.")
        return

    command = [
        "netconvert",
        "--node-files=square_grid.nod.xml",
        "--edge-files=square_grid.edg.xml",
        "--connection-files=square_grid.con.xml",
        "--output-file=square_grid.net.xml",
        "--tls.join", "true",
        "--tls.default-type", "static",
    ]
    print("Compiling SUMO network via netconvert...")
    try:
        result = subprocess.run(command, cwd=SUMO_DIR, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError("netconvert was not found. Install SUMO and add its bin directory to PATH.") from exc
    if result.returncode != 0:
        raise RuntimeError(f"netconvert failed:\n{result.stdout}\n{result.stderr}")

    network_path = SUMO_DIR / "square_grid.net.xml"
    if not network_path.exists():
        raise RuntimeError("netconvert exited successfully but square_grid.net.xml was not created")
    print(f"Network compiled: {network_path}")
    print("Setup complete. Visualize with: sumo-gui -c sumo_env/square_grid.sumocfg")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=5, help="Number of intersections per side (default: 5)")
    parser.add_argument("--max-steps", type=int, default=3600, help="Traffic generation horizon in simulation seconds")
    parser.add_argument("--ev-probability", type=float, default=0.04, help="Probability that a spawned vehicle is an emergency vehicle")
    parser.add_argument("--spawn-probability", type=float, default=0.35, help="Per-second vehicle spawn probability (retains original demand setting)")
    parser.add_argument("--generate-only", action="store_true", help="Write .nod/.edg/.con/.rou XML files without calling netconvert")
    args = parser.parse_args()
    build(args.grid_size, args.max_steps, args.ev_probability, args.spawn_probability, args.generate_only)


if __name__ == "__main__":
    main()
