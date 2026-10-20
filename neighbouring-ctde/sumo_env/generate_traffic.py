"""Generate reproducible vehicle routes for an NxN SUMO signalized grid."""
from __future__ import annotations

import random
from pathlib import Path
from typing import List, Sequence, Tuple

DIRECTIONS = ("N", "E", "S", "W")


def _neighbor(row: int, col: int, direction: str, grid_size: int):
    if direction == "N" and row > 0:
        return row - 1, col
    if direction == "E" and col < grid_size - 1:
        return row, col + 1
    if direction == "S" and row < grid_size - 1:
        return row + 1, col
    if direction == "W" and col > 0:
        return row, col - 1
    return None


def _boundary_ports(grid_size: int) -> List[Tuple[int, str]]:
    """Return (intersection index, side) pairs with an external road."""
    ports = []
    for row in range(grid_size):
        for col in range(grid_size):
            index = row * grid_size + col
            for direction in DIRECTIONS:
                if _neighbor(row, col, direction, grid_size) is None:
                    ports.append((index, direction))
    return ports


def _manhattan_path(start: int, destination: int, grid_size: int, rng: random.Random) -> List[int]:
    """Build a randomized shortest path containing only adjacent grid nodes."""
    row, col = divmod(start, grid_size)
    dest_row, dest_col = divmod(destination, grid_size)
    moves = (["V"] * abs(dest_row - row)) + (["H"] * abs(dest_col - col))
    rng.shuffle(moves)

    path = [start]
    for move in moves:
        if move == "V":
            row += 1 if dest_row > row else -1
        else:
            col += 1 if dest_col > col else -1
        path.append(row * grid_size + col)
    return path


def _route_edges(start: Tuple[int, str], end: Tuple[int, str], grid_size: int, rng: random.Random) -> List[str]:
    start_index, entry_side = start
    end_index, exit_side = end
    path = _manhattan_path(start_index, end_index, grid_size, rng)

    edges = [f"in_TL_{start_index}_{entry_side}"]
    for source, destination in zip(path, path[1:]):
        edges.append(f"edge_{source}_to_{destination}")
    edges.append(f"out_TL_{end_index}_{exit_side}")
    return edges


def generate_routefile(
    filename="square_grid.rou.xml",
    max_steps: int = 3600,
    ev_probability: float = 0.04,
    spawn_probability: float = 0.35,
    grid_size: int = 5,
    route_count: int = 120,
    seed: int = 42,
):
    """Write route definitions and scheduled vehicles for the requested grid.

    Traffic demand and emergency-vehicle probability intentionally retain the
    values used by the original 2x2 project. The network geometry and route
    pool are the parts that scale with ``grid_size``.
    """
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    if not 0.0 <= ev_probability <= 1.0:
        raise ValueError("ev_probability must be between 0 and 1")
    if not 0.0 <= spawn_probability <= 1.0:
        raise ValueError("spawn_probability must be between 0 and 1")

    rng = random.Random(seed)
    ports = _boundary_ports(grid_size)
    if len(ports) < 2:
        raise ValueError("The grid must have at least two boundary ports")

    routes: List[List[str]] = []
    seen = set()
    attempts = 0
    max_attempts = max(route_count * 20, 1000)
    while len(routes) < route_count and attempts < max_attempts:
        attempts += 1
        start = rng.choice(ports)
        possible_ends = [p for p in ports if p[0] != start[0]]
        if not possible_ends:
            continue
        end = rng.choice(possible_ends)
        route_edges = _route_edges(start, end, grid_size, rng)
        key = tuple(route_edges)
        if key not in seen:
            seen.add(key)
            routes.append(route_edges)

    if not routes:
        raise RuntimeError("Could not generate any valid routes")

    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as out:
        out.write('''<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    <!-- Standard passenger vehicle: original project settings retained. -->
    <vType id="standard_car" accel="2.6" decel="4.5" sigma="0.5" length="4.5" minGap="2.5" maxSpeed="13.89" guiShape="passenger"/>
    <!-- Emergency vehicle: original project settings retained. -->
    <vType id="emergency_car" vClass="emergency" accel="4.0" decel="6.0" sigma="0.1" length="6.5" minGap="1.5" maxSpeed="22.22" speedDev="0.1" impatience="1.0" color="red" guiShape="emergency"/>
''')
        for route_index, route_edges in enumerate(routes):
            out.write(f'    <route id="r_{route_index}" edges="{" ".join(route_edges)}"/>\n')

        vehicle_id = 0
        for step in range(max_steps):
            if rng.random() < spawn_probability:
                route_index = rng.randrange(len(routes))
                vehicle_type = "emergency_car" if rng.random() < ev_probability else "standard_car"
                out.write(
                    f'    <vehicle id="veh_{vehicle_id}" type="{vehicle_type}" '
                    f'route="r_{route_index}" depart="{step}" departLane="best" departSpeed="max"/>\n'
                )
                vehicle_id += 1
        out.write("</routes>\n")

    print(
        f"Traffic routes generated: {output_path} | grid={grid_size}x{grid_size} | "
        f"routes={len(routes)} | vehicles={vehicle_id} | EV probability={ev_probability:.3f}"
    )
    return len(routes), vehicle_id


if __name__ == "__main__":
    generate_routefile()
