import subprocess
import os
from generate_traffic import generate_routefile

def build():
    print("Step 1: Compiling SUMO Network via netconvert...")
    cmd = [
        "netconvert",
        "--node-files=square_grid.nod.xml",
        "--edge-files=square_grid.edg.xml",
        "--connection-files=square_grid.con.xml",
        "--output-file=square_grid.net.xml",
        "--tls.join", "true",
        "--tls.default-type", "static"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("netconvert failed with error:")
        print(res.stderr)
        return
    print("Network successfully compiled: square_grid.net.xml")

    print("\nStep 2: Generating Traffic Routes with Emergency Vehicles...")
    generate_routefile("square_grid.rou.xml")
    print("\nPhase 1 Setup Complete! You can test visualization with:")
    print("sumo-gui -c square_grid.sumocfg")

if __name__ == "__main__":
    build()