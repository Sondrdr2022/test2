import traci
from SmartTrafficSystem import SmartTrafficSystem

SUMO_CFG = r"C:\Users\Admin\Downloads\sumo test\New folder\dataset1.sumocfg"
sumo_binary = "sumo-gui"
sumo_cmd = [sumo_binary, "-c", SUMO_CFG]
thresholds = {"status_limit": 50}

def main():
    traci.start(sumo_cmd)
    try:
        # Get all lane IDs from SUMO network
        lane_ids = traci.lane.getIDList()
        print("Lanes detected in SUMO:", lane_ids)
        system = SmartTrafficSystem(lane_ids, cycle_time=120, thresholds=thresholds)

        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()

            # Gather live sensor data for each lane at this time step
            realtime_data = {}
            for lane_id in lane_ids:
                queue_length = traci.lane.getLastStepHaltingNumber(lane_id)     # Number of stopped vehicles
                avg_wait_time = 0  # (Custom: SUMO doesn't provide directly)
                density = traci.lane.getLastStepOccupancy(lane_id)
                avg_speed = traci.lane.getLastStepMeanSpeed(lane_id)
                flow_rate = traci.lane.getLastStepVehicleNumber(lane_id)
                realtime_data[lane_id] = {
                    "l": queue_length,
                    "tđ": avg_wait_time,
                    "m": density,
                    "v": avg_speed,
                    "g": flow_rate,
                    "ambulance": False  # Add logic if you wish to detect special vehicles
                }

            # Update the smart system and run cycle
            system.update_sensors(realtime_data)
            system.run_cycle()

            # If you want to set the traffic light, do it here using system.current_green
            # traci.trafficlight.setPhase("YOUR_JUNCTION_ID", system.current_green)

    finally:
        traci.close()

if __name__ == "__main__":
    main()