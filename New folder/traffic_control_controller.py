from config import LIMIT, AMBULANCE_WEIGHT

class TrafficController:
    def __init__(self, lanes):
        self.lanes = lanes

    def update_lane_scores(self):
        for lane in self.lanes:
            prev_status = lane.status
            lane.compute_status(LIMIT)
            # Cập nhật điểm chuỗi
            if lane.status == "GOOD":
                if prev_status == "GOOD":
                    lane.score = min(lane.score + 1, 5)
                else:
                    lane.score = 1  # reset về 1 khi chuyển BAD→GOOD
            else:  # BAD
                if prev_status == "BAD":
                    lane.score = max(lane.score - 1, -5)
                else:
                    lane.score = -1  # reset về -1 khi chuyển GOOD→BAD

    def distribute_green_time(self, total_cycle):
        # Nếu có xe ưu tiên, phân toàn bộ đèn xanh cho làn đó
        for lane in self.lanes:
            if lane.ambulance_detected:
                return {lane.lane_id: total_cycle}
        # Ngược lại, phân theo điểm
        total_score = sum([abs(lane.score) for lane in self.lanes])
        green_times = {}
        for lane in self.lanes:
            if total_score == 0:
                green_times[lane.lane_id] = total_cycle // len(self.lanes)
            else:
                weight = abs(lane.score) + 1  # +1 để tránh chia 0
                green_times[lane.lane_id] = int(total_cycle * weight / (total_score + len(self.lanes)))
        return green_times