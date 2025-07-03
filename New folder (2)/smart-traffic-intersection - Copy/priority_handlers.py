import traci

def handle_priority_vehicles():
    for veh_id in traci.vehicle.getIDList():
        try:
            if traci.vehicle.getTypeID(veh_id) == "ambulance":
                lane_id = traci.vehicle.getLaneID(veh_id)
                traci.trafficlight.setPhase("J0", EMERGENCY_GREEN_PHASE)
                while traci.vehicle.getLaneID(veh_id) == lane_id:
                    traci.simulationStep()
        except traci.exceptions.TraCIException:
            # Vehicle is no longer in the simulation; skip
            continue