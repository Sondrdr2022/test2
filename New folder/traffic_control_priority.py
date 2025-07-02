def detect_ambulance(traci, lane_id):
    # Giả sử dùng TraCI để kiểm tra xe có attribute 'type' == 'emergency'
    for veh_id in traci.lane.getLastStepVehicleIDs(lane_id):
        if traci.vehicle.getTypeID(veh_id) == 'emergency':
            return True
    return False