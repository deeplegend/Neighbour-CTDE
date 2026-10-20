import os
import random

def generate_routefile(filename="square_grid.rou.xml", max_steps=3600, ev_probability=0.04, seed=42):
    rng = random.Random(seed)
    
    # Valid Origin-Destination path sequences traversing through intersections
    routes_pool = [
        # Horizontal & Vertical straight corridors
        ["in_TL0_W", "edge_0_to_1", "out_TL1_E"],
        ["in_TL1_E", "edge_1_to_0", "out_TL0_W"],
        ["in_TL0_N", "edge_0_to_3", "out_TL3_S"],
        ["in_TL3_S", "edge_3_to_0", "out_TL0_N"],
        ["in_TL1_N", "edge_1_to_2", "out_TL2_S"],
        ["in_TL2_S", "edge_2_to_1", "out_TL1_N"],
        ["in_TL3_W", "edge_3_to_2", "out_TL2_E"],
        ["in_TL2_E", "edge_2_to_3", "out_TL3_W"],
        
        # Turning trajectories through the square
        ["in_TL0_W", "edge_0_to_3", "edge_3_to_2", "out_TL2_E"],
        ["in_TL0_N", "edge_0_to_1", "edge_1_to_2", "out_TL2_S"],
        ["in_TL2_E", "edge_2_to_1", "edge_1_to_0", "out_TL0_W"],
        ["in_TL3_S", "edge_3_to_2", "edge_2_to_1", "out_TL1_N"]
    ]

    with open(filename, "w") as routes:
        routes.write("""<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    <!-- Standard Passenger Vehicle -->
    <vType id="standard_car" accel="2.6" decel="4.5" sigma="0.5" length="4.5" minGap="2.5" maxSpeed="13.89" guiShape="passenger"/>

    <!-- Emergency Vehicle (High Priority, High Speed, Siren Highlight) -->
    <vType id="emergency_car" vClass="emergency" accel="4.0" decel="6.0" sigma="0.1" length="6.5" minGap="1.5" maxSpeed="22.22" speedDev="0.1" impatience="1.0" color="red" guiShape="emergency"/>
""")
        # Register Route IDs
        for idx, r in enumerate(routes_pool):
            routes.write(f'    <route id="r_{idx}" edges="{" ".join(r)}"/>\n')

        veh_id = 0
        for step in range(max_steps):
            # Spawn probability per second
            if rng.random() < 0.35:
                route_idx = rng.randint(0, len(routes_pool) - 1)
                is_ev = rng.random() < ev_probability
                vtype = "emergency_car" if is_ev else "standard_car"
                
                routes.write(f'    <vehicle id="veh_{veh_id}" type="{vtype}" route="r_{route_idx}" depart="{step}" departLane="best" departSpeed="max"/>\n')
                veh_id += 1

        routes.write("</routes>\n")
    print(f"Traffic routes successfully generated: {filename} with {veh_id} total vehicles.")

if __name__ == "__main__":
    generate_routefile()