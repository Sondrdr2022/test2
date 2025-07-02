import traci
from traffic_control_config import LANES, MAX_CYCLE
from traffic_control_lane import Lane
from traffic_control_controller import TrafficController
from traffic_control_priority import detect_ambulance

def run():
    traci.start(["sumo-gui", "-c", "your_sumocfg.sumocfg"])
    lanes = [Lane(lane_id) for lane_id in LANES]
    controller = TrafficController(lanes)

    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        for lane in lanes:
            lane.update(
                l=traci.lane.getLastStepHaltingNumber(lane.lane_id),
                td=traci.lane.getWaitingTime(lane.lane_id),
                m=traci.lane.getLastStepOccupancy(lane.lane_id),
                v=traci.lane.getLastStepMeanSpeed(lane.lane_id),
                g=traci.lane.getLastStepVehicleNumber(lane.lane_id),
                ambulance_detected=detect_ambulance(traci, lane.lane_id)
            )
        controller.update_lane_scores()
        green_time = controller.distribute_green_time(MAX_CYCLE)
        # Áp dụng green_time cho từng phase của đèn
        # (Bạn cần ánh xạ các lane sang phase logic của SUMO)
        # Ví dụ: traci.trafficlight.setPhaseDuration(tl_id, green_time[lane_id])
        print(f"Step {step}, green_times: {green_time}, scores: {[lane.score for lane in lanes]}")
        traci.simulationStep()
        step += 1

    traci.close()

if __name__ == "__main__":
    run()