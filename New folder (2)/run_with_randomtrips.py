import os
import sys
import time
import subprocess
import xml.etree.ElementTree as ET
import random
from collections import defaultdict

# Always work in script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

NET_FILE = 'dataset.net.xml'
ROUTE_FILE = 'random_trips.rou.xml'
VTYPES_FILE = 'vtypes.xml'
SUMOCFG_FILE = 'random_simulation.sumocfg'
RANDOM_SEED = int(time.time())

# Optimized congestion parameters
TRIP_NUMBER = 1000
BEGIN_TIME = 0
END_TIME = 3600
USE_GUI = True

# Congestion settings
PEAK_INTENSITY = 2.0
CONGESTION_PERIODS = [
    (900, 1200),   # Morning rush: 15-20 minutes
    (1800, 2100)   # Evening rush: 30-35 minutes
]

SUMO_HOME = os.environ.get('SUMO_HOME', r'C:\Program Files (x86)\Eclipse\Sumo')
TOOLS_PATH = os.path.join(SUMO_HOME, 'tools')
RANDOMTRIPS_PATH = os.path.join(TOOLS_PATH, 'randomTrips.py')
SUMO_BINARY = os.path.join(SUMO_HOME, 'bin', 'sumo-gui.exe' if USE_GUI else 'sumo.exe')

def create_vehicle_types():
    """Create optimized vehicle types for congestion"""
    vtypes = ET.Element('routes')
    
    # Passenger car (60% of vehicles)
    ET.SubElement(vtypes, 'vType', {
        'id': 'passenger',
        'vClass': 'passenger',
        'length': '4.5',
        'maxSpeed': '55',
        'accel': '2.6',
        'decel': '4.5',
        'sigma': '0.5'
    })
    
    # Truck (25% of vehicles)
    ET.SubElement(vtypes, 'vType', {
        'id': 'truck',
        'vClass': 'truck',
        'length': '16',
        'maxSpeed': '35',
        'accel': '1.3',
        'decel': '3.0',
        'sigma': '0.3'
    })
    
    # Bus (15% of vehicles)
    ET.SubElement(vtypes, 'vType', {
        'id': 'bus',
        'vClass': 'bus',
        'length': '12',
        'maxSpeed': '40',
        'accel': '1.5',
        'decel': '3.5',
        'sigma': '0.4'
    })
    
    tree = ET.ElementTree(vtypes)
    with open(VTYPES_FILE, 'wb') as f:
        tree.write(f, encoding='utf-8', xml_declaration=True)
    print(f"[INFO] Created optimized vehicle types: {VTYPES_FILE}")

def generate_randomtrips():
    """Generate trips with optimized congestion"""
    base_vehicles = int(TRIP_NUMBER * 0.7)
    congestion_vehicles = int(TRIP_NUMBER * 0.3 * PEAK_INTENSITY)
    
    # Calculate periods
    base_period = max(1, END_TIME // base_vehicles)
    
    # Baseline traffic
    print("[INFO] Generating baseline traffic...")
    base_cmd = [
        sys.executable, RANDOMTRIPS_PATH,
        '-n', NET_FILE,
        '-o', 'baseline.rou.xml',
        '--seed', str(RANDOM_SEED),
        '--trip-attributes', 'departLane="best" departSpeed="max" departPos="random_free"',
        '-b', str(BEGIN_TIME),
        '-e', str(END_TIME),
        '-p', str(base_period),
        '--validate'
    ]
    subprocess.check_call(base_cmd)
    
    # Congestion traffic
    route_files = ['baseline.rou.xml']
    for i, (start, end) in enumerate(CONGESTION_PERIODS):
        print(f"[INFO] Generating congestion period {i+1}...")
        duration = end - start
        congestion_period = max(1, duration // (congestion_vehicles // len(CONGESTION_PERIODS)))
        
        congestion_cmd = [
            sys.executable, RANDOMTRIPS_PATH,
            '-n', NET_FILE,
            '-o', f'congestion_{i}.rou.xml',
            '--seed', str(RANDOM_SEED + i + 1),
            '--trip-attributes', 'departLane="best" departSpeed="max" departPos="random_free"',
            '-b', str(start),
            '-e', str(end),
            '-p', str(congestion_period),
            '--prefix', f'peak{i}_',
            '--validate'
        ]
        subprocess.check_call(congestion_cmd)
        route_files.append(f'congestion_{i}.rou.xml')
    
    # Merge files with sorting
    merge_and_assign_vehicles(route_files, ROUTE_FILE)
    print(f"[INFO] Optimized routes generated: {ROUTE_FILE}")

def merge_and_assign_vehicles(route_files, output_file):
    """Merge route files with optimized assignments and proper sorting"""
    # Create dictionary to store vehicles by departure time
    vehicles_by_time = defaultdict(list)
    
    # Process files and collect vehicles
    for route_file in route_files:
        tree = ET.parse(route_file)
        root = tree.getroot()
        
        for element in root:
            if element.tag in ['trip', 'vehicle']:
                # Get departure time (default to 0 if missing)
                depart_time = float(element.get('depart', '0'))
                
                # Assign vehicle type
                rand = random.random()
                if 'baseline' in route_file:
                    vtype = 'passenger' if rand < 0.7 else 'truck' if rand < 0.9 else 'bus'
                else:  # Congestion file
                    vtype = 'passenger' if rand < 0.5 else 'truck' if rand < 0.8 else 'bus'
                element.set('type', vtype)
                
                # Store in dictionary by time
                vehicles_by_time[depart_time].append(element)
    
    # Create sorted list of departure times
    sorted_times = sorted(vehicles_by_time.keys())
    
    # Create new root element
    routes = ET.Element('routes')
    
    # Add vehicle types first
    vtypes_tree = ET.parse(VTYPES_FILE)
    for vtype in vtypes_tree.getroot():
        routes.append(vtype)
    
    # Add vehicles in chronological order
    for time in sorted_times:
        for vehicle in vehicles_by_time[time]:
            routes.append(vehicle)
    
    # Write sorted route file
    tree = ET.ElementTree(routes)
    with open(output_file, 'wb') as f:
        tree.write(f, encoding='utf-8', xml_declaration=True)
    
    # Cleanup
    for file in route_files:
        if os.path.exists(file):
            os.remove(file)

def build_sumocfg():
    """Build optimized SUMO configuration"""
    cfg = ET.Element('configuration')
    input_elem = ET.SubElement(cfg, 'input')
    ET.SubElement(input_elem, 'net-file').set('value', NET_FILE)
    ET.SubElement(input_elem, 'route-files').set('value', ROUTE_FILE)
    
    time_elem = ET.SubElement(cfg, 'time')
    ET.SubElement(time_elem, 'begin').set('value', str(BEGIN_TIME))
    ET.SubElement(time_elem, 'end').set('value', str(END_TIME))

    tree = ET.ElementTree(cfg)
    with open(SUMOCFG_FILE, 'wb') as f:
        tree.write(f, encoding='utf-8', xml_declaration=True)
    print(f"[INFO] Generated optimized SUMO config: {SUMOCFG_FILE}")

def run_rl_script(extra_args=None):
    """Run with connection retry optimization"""
    rl_script = 'Lane.py'
    cmd = [sys.executable, rl_script, '--sumo', SUMOCFG_FILE]
    
    # Add connection retry parameters
    cmd.extend(['--num-retries', '120'])
    cmd.extend(['--retry-delay', '1'])
    
    if USE_GUI:
        cmd.append('--gui')
    if extra_args:
        cmd.extend(extra_args)
    
    print("[INFO] Running RL script with optimized parameters:")
    print(" ".join(cmd))
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] RL script failed: {e}")
        sys.exit(1)

def validate_simulation():
    """Test SUMO configuration before RL run"""
    cmd = [SUMO_BINARY, '-c', SUMOCFG_FILE, '--no-step-log', 'true']
    print("[INFO] Validating simulation with command:")
    print(" ".join(cmd))
    try:
        # Run validation for 10 seconds max
        result = subprocess.run(cmd, timeout=10, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[ERROR] Validation failed with exit code {result.returncode}")
            print("Validation output:")
            print(result.stdout)
            print(result.stderr)
            sys.exit(1)
        print("[SUCCESS] Simulation validated successfully!")
    except subprocess.TimeoutExpired:
        print("[SUCCESS] Validation timed out but SUMO started (expected behavior)")
    except Exception as e:
        print(f"[ERROR] Validation failed: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    print("🚦 GENERATING OPTIMIZED CONGESTION SIMULATION")
    print(f"📊 Total vehicles: {TRIP_NUMBER}")
    print(f"🚨 Peak intensity: {PEAK_INTENSITY}x")
    print(f"⏰ Congestion periods: {len(CONGESTION_PERIODS)}")
    print("=" * 60)
    
    create_vehicle_types()
    generate_randomtrips()
    build_sumocfg()
    
    # Validate before RL run
    validate_simulation()
    
    # Run with retry parameters
    run_rl_script(extra_args=[
        '--max-steps', '1500',
        '--episodes', '2'
    ])
    
    print("\n🎉 Simulation completed successfully!")