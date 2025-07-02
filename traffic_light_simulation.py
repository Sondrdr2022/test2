import traci

class LaneStateTracker:
    def __init__(self, lane_ids, limits, weights):
        self.scores = {lane: 0 for lane in lane_ids}
        self.states = {lane: 'GOOD' for lane in lane_ids}
        self.limits = limits  # dict: {'l':.., 'tđ':.., ...}
        self.weights = weights  # dict: {'l':w1, 'tđ':w2, ...}
    
    def compute_status(self, params):
        # params: dict {lane: {'l':..., 'tđ':..., 'm':..., 'v':..., 'g':...}}
        status = {}
        for lane, vals in params.items():
            s = (self.weights['l'] * (vals['l']/self.limits['l']) +
                 self.weights['tđ'] * (vals['tđ']/self.limits['tđ']) +
                 self.weights['m'] * (vals['m']/self.limits['m']) -
                 self.weights['v'] * (vals['v']/self.limits['v']) -
                 self.weights['g'] * (vals['g']/self.limits['g']))
            status[lane] = 'GOOD' if s < 1 else 'BAD'
        return status
    
    def update_scores(self, current_status):
        for lane, st in current_status.items():
            prev = self.states[lane]
            score = self.scores[lane]
            if st == prev:
                if st == 'GOOD':
                    self.scores[lane] = min(score + 1, 5)
                else:
                    self.scores[lane] = max(score - 1, -5)
            else:
                self.scores[lane] = 0
            self.states[lane] = st
    
    def get_scores(self):
        return self.scores

class PriorityDetector:
    def __init__(self, priority_vehicle_types=['ambulance']):
        self.priority_vehicle_types = priority_vehicle_types

    def detect_priority(self, lane_id):
        vehs = traci.lane.getLastStepVehicleIDs(lane_id)
        for v in vehs:
            vtype = traci.vehicle.getTypeID(v)
            if vtype in self.priority_vehicle_types:
                return True
        return False

class TrafficLightController:
    def __init__(self, tls_id, lane_ids, base_green=10, alpha=20):
        self.tls_id = tls_id
        self.lane_ids = lane_ids
        self.base_green = base_green
        self.alpha = alpha
    
    def set_phase_times(self, lane_scores, priority_lane=None):
        if priority_lane:
            # All red, except priority lane green max
            phases = []
            for lane in self.lane_ids:
                phases.append(self.base_green*10 if lane == priority_lane else 0)
            # Set green for priority_lane, others red
            self.apply_phase(priority_lane, max(self.base_green*10, 60))
            return
        # Calculate green time per lane based on score
        total_score = sum([max(1, abs(s)) for s in lane_scores.values()])
        for lane, score in lane_scores.items():
            green_time = int(self.base_green + self.alpha * score / total_score)
            self.apply_phase(lane, green_time)
    
    def apply_phase(self, lane, green_time):
        # TODO: mapping lane to phase index, depends on TLS logic in SUMO .net.xml
        # Example: traci.trafficlight.setPhaseDuration(self.tls_id, green_time)
        pass

# Main simulation loop
def run_simulation():
    lane_ids = ["lane1", "lane2", "lane3", "lane4"]
    limits = {'l': 15, 'tđ': 20, 'm': 0.5, 'v': 10, 'g': 20}
    weights = {'l': 0.3, 'tđ': 0.3, 'm': 0.2, 'v': 0.1, 'g': 0.1}
    tracker = LaneStateTracker(lane_ids, limits, weights)
    detector = PriorityDetector()
    controller = TrafficLightController("junction_id", lane_ids)

    while traci.simulation.getMinExpectedNumber() > 0:
        params = {}
        priority_lane = None
        for lane in lane_ids:
            params[lane] = {
                'l': traci.lane.getLastStepHaltingNumber(lane),
                'tđ': traci.lane.getWaitingTime(lane),  # hoặc custom tính
                'm': traci.lane.getLastStepOccupancy(lane),
                'v': traci.lane.getLastStepMeanSpeed(lane),
                'g': traci.lane.getLastStepVehicleNumber(lane)
            }
            if detector.detect_priority(lane):
                priority_lane = lane
        status = tracker.compute_status(params)
        tracker.update_scores(status)
        lane_scores = tracker.get_scores()
        controller.set_phase_times(lane_scores, priority_lane)
        traci.simulationStep()
    traci.close()

if __name__ == "__main__":
    import traci
    traci.start(["sumo-gui", "-c", r"C:\Users\Admin\Downloads\sumo test\New folder\dataset1.sumocfg"])
    run_simulation()