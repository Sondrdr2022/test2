import os
import sys
import time
import random
import subprocess
import xml.etree.ElementTree as ET

# Always work in script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

NET_FILE = 'dataset.net.xml'
ROUTE_FILE = 'random_trips.rou.xml'
SUMOCFG_FILE = 'random_simulation.sumocfg'
VTYPE_FILE = 'vehicle_types.add.xml'

# Generate a new random seed for each run
RANDOM_SEED = random.randint(1, 1000000)
TRIP_NUMBER = 400
BEGIN_TIME = 0
END_TIME = 3600
USE_GUI = True

SUMO_HOME = os.environ.get('SUMO_HOME', r'C:\Program Files (x86)\Eclipse\Sumo')
TOOLS_PATH = os.path.join(SUMO_HOME, 'tools')
RANDOMTRIPS_PATH = os.path.join(TOOLS_PATH, 'randomTrips.py')

def create_vehicle_types():
    """Create diverse vehicle types with different characteristics"""
    root = ET.Element('additional')
    
    # Define various vehicle types with different properties
    vehicle_types = [
        {
            'id': 'car_normal',
            'accel': '2.6',
            'decel': '4.5',
            'sigma': '0.5',
            'length': '5.0',
            'maxSpeed': '50.0',
            'vClass': 'passenger'
        },
        {
            'id': 'car_aggressive',
            'accel': '3.5',
            'decel': '5.0',
            'sigma': '0.3',
            'length': '4.5',
            'maxSpeed': '60.0',
            'vClass': 'passenger'
        },
        {
            'id': 'car_slow',
            'accel': '1.8',
            'decel': '3.5',
            'sigma': '0.8',
            'length': '5.5',
            'maxSpeed': '40.0',
            'vClass': 'passenger'
        },
        {
            'id': 'truck',
            'accel': '1.2',
            'decel': '3.0',
            'sigma': '0.6',
            'length': '12.0',
            'maxSpeed': '35.0',
            'vClass': 'truck'
        },
        {
            'id': 'bus',
            'accel': '1.0',
            'decel': '2.8',
            'sigma': '0.7',
            'length': '15.0',
            'maxSpeed': '30.0',
            'vClass': 'bus'
        },
        {
            'id': 'motorcycle',
            'accel': '4.0',
            'decel': '6.0',
            'sigma': '0.4',
            'length': '2.5',
            'maxSpeed': '70.0',
            'vClass': 'motorcycle'
        }
    ]
    
    # Add vehicle types to XML
    for vtype in vehicle_types:
        vtype_elem = ET.SubElement(root, 'vType')
        for attr, value in vtype.items():
            vtype_elem.set(attr, value)
    
    # Write vehicle types file
    tree = ET.ElementTree(root)
    with open(VTYPE_FILE, 'wb') as f:
        tree.write(f, encoding='utf-8', xml_declaration=True)
    print(f"[INFO] Created vehicle types file: {VTYPE_FILE}")

def generate_randomtrips():
    """Generate random trips with varied vehicle types and congestion"""
    # Vary trip density for different congestion levels
    congestion_factor = random.uniform(0.5, 2.0)  # 0.5 = light traffic, 2.0 = heavy traffic
    adjusted_trip_number = int(TRIP_NUMBER * congestion_factor)
    
    # Calculate period between trips
    period = max(1, END_TIME // adjusted_trip_number)
    
    # Create weighted vehicle type distribution
    vehicle_weights = [
        ('car_normal', 0.4),
        ('car_aggressive', 0.2),
        ('car_slow', 0.2),
        ('truck', 0.1),
        ('bus', 0.05),
        ('motorcycle', 0.05)
    ]
    
    # Build vehicle type string for randomTrips.py
    vtypes = []
    for vtype, weight in vehicle_weights:
        vtypes.extend([vtype] * int(weight * 100))
    
    vtype_string = ','.join(vtypes)
    
    cmd = [
        sys.executable, RANDOMTRIPS_PATH,
        '-n', NET_FILE,
        '-o', ROUTE_FILE,
        '--seed', str(RANDOM_SEED),
        '--trip-attributes', 'departLane="best" departSpeed="max" departPos="random_free"',
        '-b', str(BEGIN_TIME),
        '-e', str(END_TIME),
        '-p', str(period),
        '--vehicle-class', vtype_string,
        '--validate'
    ]
    
    print(f"[INFO] Generating random trips with seed: {RANDOM_SEED}")
    print(f"[INFO] Congestion factor: {congestion_factor:.2f}")
    print(f"[INFO] Adjusted trip number: {adjusted_trip_number}")
    print(f"[INFO] Period between trips: {period}s")
    print("[INFO] Command:")
    print(" ".join(cmd))
    
    subprocess.check_call(cmd)
    print(f"[INFO] Random trips generated: {ROUTE_FILE}")

def modify_route_file_for_vtypes():
    """Modify the generated route file to assign random vehicle types"""
    try:
        tree = ET.parse(ROUTE_FILE)
        root = tree.getroot()
        
        vehicle_types = ['car_normal', 'car_aggressive', 'car_slow', 'truck', 'bus', 'motorcycle']
        weights = [0.4, 0.2, 0.2, 0.1, 0.05, 0.05]
        
        # Find all vehicle/trip elements and assign random types
        for elem in root.iter():
            if elem.tag in ['vehicle', 'trip']:
                # Assign random vehicle type based on weights
                vtype = random.choices(vehicle_types, weights=weights)[0]
                elem.set('type', vtype)
        
        # Write back the modified file
        tree.write(ROUTE_FILE, encoding='utf-8', xml_declaration=True)
        print(f"[INFO] Modified route file with vehicle types: {ROUTE_FILE}")
        
    except Exception as e:
        print(f"[WARNING] Could not modify route file: {e}")

def build_sumocfg():
    """Build SUMO configuration file including vehicle types"""
    cfg = ET.Element('configuration')
    
    # Input section
    input_elem = ET.SubElement(cfg, 'input')
    netfile_elem = ET.SubElement(input_elem, 'net-file')
    netfile_elem.set('value', NET_FILE)
    routefile_elem = ET.SubElement(input_elem, 'route-files')
    routefile_elem.set('value', ROUTE_FILE)
    
    # Add vehicle types file
    addfile_elem = ET.SubElement(input_elem, 'additional-files')
    addfile_elem.set('value', VTYPE_FILE)
    
    # Time section
    time_elem = ET.SubElement(cfg, 'time')
    ET.SubElement(time_elem, 'begin').set('value', str(BEGIN_TIME))
    ET.SubElement(time_elem, 'end').set('value', str(END_TIME))
    
    # Random number generation
    random_elem = ET.SubElement(cfg, 'random_number')
    ET.SubElement(random_elem, 'seed').set('value', str(RANDOM_SEED))
    
    tree = ET.ElementTree(cfg)
    with open(SUMOCFG_FILE, 'wb') as f:
        tree.write(f, encoding='utf-8', xml_declaration=True)
    print(f"[INFO] Generated SUMO config: {SUMOCFG_FILE}")

def run_rl_script(extra_args=None):
    """Run the RL main script"""
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
    print(f"[INFO] Starting simulation with random seed: {RANDOM_SEED}")
    create_vehicle_types()
    generate_randomtrips()
    modify_route_file_for_vtypes()
    build_sumocfg()
    
    # You can add args like ['--max-steps', '1500', '--episodes', '2']
    run_rl_script(extra_args=['--max-steps', '1500', '--episodes', '2'])