import os
import sys
import time
import subprocess
import xml.etree.ElementTree as ET

# Always work in script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

NET_FILE = 'dataset.net.xml'
ROUTE_FILE = 'random_trips.rou.xml'
SUMOCFG_FILE = 'random_simulation.sumocfg'
RANDOM_SEED = int(time.time())
TRIP_NUMBER = 400
BEGIN_TIME = 0
END_TIME = 3600
USE_GUI = True

SUMO_HOME = os.environ.get('SUMO_HOME', r'C:\Program Files (x86)\Eclipse\Sumo')
TOOLS_PATH = os.path.join(SUMO_HOME, 'tools')
RANDOMTRIPS_PATH = os.path.join(TOOLS_PATH, 'randomTrips.py')

def generate_randomtrips():
    cmd = [
        sys.executable, RANDOMTRIPS_PATH,
        '-n', NET_FILE,
        '-o', ROUTE_FILE,
        '--seed', str(RANDOM_SEED),
        '--trip-attributes', 'departLane="best" departSpeed="max" departPos="random_free"',
        '-b', str(BEGIN_TIME),
        '-e', str(END_TIME),
        '-p', str(max(1, END_TIME // TRIP_NUMBER)),
        '--validate'
    ]
    print("[INFO] Generating random trips with command:")
    print(" ".join(cmd))
    subprocess.check_call(cmd)
    print("[INFO] Random trips generated:", ROUTE_FILE)

# ...rest of your script unchanged....
# ==== GENERATE SUMO CONFIG FILE ====
def build_sumocfg():
    cfg = ET.Element('configuration')
    input_elem = ET.SubElement(cfg, 'input')
    netfile_elem = ET.SubElement(input_elem, 'net-file')
    netfile_elem.set('value', NET_FILE)
    routefile_elem = ET.SubElement(input_elem, 'route-files')
    routefile_elem.set('value', ROUTE_FILE)
    time_elem = ET.SubElement(cfg, 'time')
    ET.SubElement(time_elem, 'begin').set('value', str(BEGIN_TIME))
    ET.SubElement(time_elem, 'end').set('value', str(END_TIME))

    tree = ET.ElementTree(cfg)
    with open(SUMOCFG_FILE, 'wb') as f:
        tree.write(f, encoding='utf-8', xml_declaration=True)
    print(f"[INFO] Generated SUMO config: {SUMOCFG_FILE}")

# ==== RUN RL MAIN SCRIPT ====
def run_rl_script(extra_args=None):
    rl_script = 'Lane.py'  # Change if your RL main file has different name
    cmd = [sys.executable, rl_script, '--sumo', SUMOCFG_FILE]
    if USE_GUI:
        cmd.append('--gui')
    # Add any extra arguments
    if extra_args:
        cmd.extend(extra_args)
    print("[INFO] Running RL script with command:")
    print(" ".join(cmd))
    subprocess.check_call(cmd)

if __name__ == '__main__':
    generate_randomtrips()
    build_sumocfg()
    # You can add args like ['--max-steps', '1500', '--episodes', '2']
    run_rl_script(extra_args=['--max-steps', '1500', '--episodes', '2'])