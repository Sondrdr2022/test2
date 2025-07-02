class SmartTrafficSystem:
    def __init__(self, lanes, cycle_time=120, thresholds={}):
        self.lanes = {id: Lane(id, params) for id in lanes}  # Các làn đường
        self.cycle_time = cycle_time  # Chu kỳ đèn cơ bản (giây)
        self.thresholds = thresholds  # Ngưỡng quyết định trạng thái
        self.priority_flag = False    # Cờ ưu tiên khẩn cấp
        self.current_green = None      # Làn đang xanh

    def update_sensors(self, realtime_data):
        # Cập nhật dữ liệu cảm biến thời gian thực
        for lane_id, data in realtime_data.items():
            self.lanes[lane_id].update(data)
            
    def calculate_status(self, lane):
        # Tính toán trạng thái làn (GOOD/BAD)
        status_index = (
            0.4 * lane.queue_length + 
            0.3 * lane.avg_wait_time +
            0.2 * lane.density -
            0.1 * lane.avg_speed
        )
        return "GOOD" if status_index < self.thresholds["status_limit"] else "BAD"

    def adjust_scores(self):
        # Cập nhật điểm chuỗi cho các làn
        for lane in self.lanes.values():
            current_status = self.calculate_status(lane)
            
            if current_status == lane.prev_status:
                if current_status == "GOOD":
                    lane.score = min(5, lane.score + 1)
                else:
                    lane.score = max(-5, lane.score - 1)
            else:
                lane.score = 1 if current_status == "GOOD" else -1
                
            lane.prev_status = current_status

    def handle_priority(self):
        # Xử lý ưu tiên khẩn cấp
        for lane in self.lanes.values():
            if lane.ambulance_detected:
                self.priority_flag = True
                self.current_green = lane.id
                return lane.id
        self.priority_flag = False
        return None

    def optimize_green_time(self):
        # Tối ưu phân bổ thời gian xanh
        if priority_lane := self.handle_priority():
            return {priority_lane: self.cycle_time}  # Ưu tiên toàn bộ chu kỳ
        
        total_score = sum(max(0, lane.score) for lane in self.lanes.values())
        if total_score == 0:  # Trường hợp đặc biệt
            return {id: self.cycle_time/len(self.lanes) for id in self.lanes}
        
        # Phân bổ theo tỷ lệ điểm
        green_times = {}
        for id, lane in self.lanes.items():
            weight = max(0, lane.score) / total_score
            green_times[id] = max(10, weight * self.cycle_time)  # Tối thiểu 10s
            
        return green_times

    def run_cycle(self):
        # Thực thi một chu kỳ đèn
        self.adjust_scores()
        green_plan = self.optimize_green_time()
        
        for lane_id, duration in green_plan.items():
            self.current_green = lane_id
            self.execute_green_phase(lane_id, duration)