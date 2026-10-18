"""Generate SUMO routes for an N x N grid of signalized intersections.

The generated trips use the edge IDs created by build_network.py:
  * in_TL_<id>_<direction>  : boundary road into an intersection
  * edge_TL_<from>_to_TL_<to>: road between adjacent intersections
  * out_TL_<id>_<direction> : road out of an intersection
"""
from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

DIRECTIONS = {
    "N": (-1, 0),
    "E": (0, 1),
    "S": (1, 0),
    "W": (0, -1),
}


def _tl_id(row: int, col: int, grid_size: int) -> str:
    return f"TL_{row * grid_size + col}"


def _edge_between(source: str, target: str) -> str:
    return f"edge_{source}_to_{target}"


def _build_boundary_accesses(grid_size: int) -> Tuple[List[Tuple[str, int, int, str]], List[Tuple[int, int, str, str]]]:
    """Return entry tuples (edge, row, col, direction) and exit tuples
    (row, col, direction, edge) for all outward-facing grid sides.
    """
    entries: List[Tuple[str, int, int, str]] = []
    exits: List[Tuple[int, int, str, str]] = []

    for row in range(grid_size):
        for col in range(grid_size):
            tl = _tl_id(row, col, grid_size)
            for direction, (dr, dc) in DIRECTIONS.items():
                nr, nc = row + dr, col + dc
                if 0 <= nr < grid_size and 0 <= nc < grid_size:
                    continue
                entries.append((f"in_{tl}_{direction}", row, col, direction))
                exits.append((row, col, direction, f"out_{tl}_{direction}"))
    return entries, exits


def _make_route(
    start_row: int,
    start_col: int,
    entry_direction: str,
    end_row: int,
    end_col: int,
    exit_direction: str,
    grid_size: int,
    rng: random.Random,
) -> Tuple[str, ...]:
    """Create a valid Manhattan path from one boundary entry to one exit."""
    start_tl = _tl_id(start_row, start_col, grid_size)
    edges: List[str] = [f"in_{start_tl}_{entry_direction}"]
    row, col = start_row, start_col

    # Randomize whether a trip makes its horizontal or vertical progress first.
    axes: Sequence[str] = ("row", "col") if rng.random() < 0.5 else ("col", "row")
    for axis in axes:
        while (row != end_row) if axis == "row" else (col != end_col):
            old_tl = _tl_id(row, col, grid_size)
            if axis == "row":
                row += 1 if end_row > row else -1
            else:
                col += 1 if end_col > col else -1
            new_tl = _tl_id(row, col, grid_size)
            edges.append(_edge_between(old_tl, new_tl))

    end_tl = _tl_id(end_row, end_col, grid_size)
    edges.append(f"out_{end_tl}_{exit_direction}")
    return tuple(edges)


def generate_routefile(
    filename: str = "square_grid.rou.xml",
    max_steps: int = 3600,
    ev_probability: float = 0.04,
    grid_size: int = 5,
    spawn_probability: float = 0.35,
    seed: int = 42,
) -> int:
    """Write a route file for a grid of signalized intersections.

    Approximately ``spawn_probability * max_steps`` vehicles are scheduled.
    All routes start on an external inbound edge and end on an external
    outbound edge.  The same seeded random generator makes comparisons
    repeatable when the parameters are unchanged.
    """
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if not 0.0 <= ev_probability <= 1.0:
        raise ValueError("ev_probability must be between 0 and 1")
    if not 0.0 <= spawn_probability <= 1.0:
        raise ValueError("spawn_probability must be between 0 and 1")

    output_path = Path(filename).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    entries, exits = _build_boundary_accesses(grid_size)

    # Precompute eligible exit choices for each entry, excluding a U-turn back
    # out of the same side of the same intersection.
    vehicles: List[Tuple[str, str, str, int]] = []
    route_lookup: Dict[Tuple[str, ...], str] = {}
    next_vehicle_id = 0

    for depart in range(max_steps):
        if rng.random() >= spawn_probability:
            continue

        entry_edge, start_row, start_col, entry_direction = rng.choice(entries)
        possible_exits = [
            item for item in exits
            if not (
                item[0] == start_row
                and item[1] == start_col
                and item[2] == entry_direction
            )
        ]
        end_row, end_col, exit_direction, _ = rng.choice(possible_exits)
        route_edges = _make_route(
            start_row, start_col, entry_direction,
            end_row, end_col, exit_direction,
            grid_size, rng,
        )
        # Ensure the first edge agrees with the sampled boundary access.
        if route_edges[0] != entry_edge:
            raise RuntimeError("Generated route does not match selected entry edge")

        if route_edges not in route_lookup:
            route_lookup[route_edges] = f"route_{len(route_lookup)}"
        vehicle_type = "emergency_car" if rng.random() < ev_probability else "standard_car"
        vehicles.append((f"veh_{next_vehicle_id}", vehicle_type, route_lookup[route_edges], depart))
        next_vehicle_id += 1

    root = ET.Element("routes")
    ET.SubElement(
        root,
        "vType",
        {
            "id": "standard_car", "accel": "2.6", "decel": "4.5",
            "sigma": "0.5", "length": "4.5", "minGap": "2.5",
            "maxSpeed": "13.89", "guiShape": "passenger",
        },
    )
    ET.SubElement(
        root,
        "vType",
        {
            "id": "emergency_car", "vClass": "emergency", "accel": "4.0",
            "decel": "6.0", "sigma": "0.1", "length": "6.5",
            "minGap": "1.5", "maxSpeed": "22.22", "speedDev": "0.1",
            "impatience": "1.0", "color": "red", "guiShape": "emergency",
        },
    )

    for route_edges, route_id in route_lookup.items():
        ET.SubElement(root, "route", {"id": route_id, "edges": " ".join(route_edges)})

    for vehicle_id, vehicle_type, route_id, depart in vehicles:
        ET.SubElement(
            root,
            "vehicle",
            {
                "id": vehicle_id,
                "type": vehicle_type,
                "route": route_id,
                "depart": str(depart),
                "departLane": "best",
                "departSpeed": "max",
            },
        )

    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    print(
        f"Generated {len(vehicles)} vehicles and {len(route_lookup)} unique routes "
        f"for a {grid_size}x{grid_size} grid: {output_path}"
    )
    return len(vehicles)


if __name__ == "__main__":
    generate_routefile()
