class Lane:
    def __init__(self, lane_id):
        self.lane_id = lane_id
        self.l = 0          # Độ dài hàng đợi
        self.td = 0         # Thời gian chờ TB
        self.m = 0          # Mật độ
        self.v = 0          # Vận tốc TB
        self.g = 0          # Lưu lượng qua nút
        self.score = 0      # Điểm tích lũy chuỗi
        self.status = "GOOD"
        self.ambulance_detected = False

    def update(self, l, td, m, v, g, ambulance_detected):
        self.l = l
        self.td = td
        self.m = m
        self.v = v
        self.g = g
        self.ambulance_detected = ambulance_detected

    def compute_status(self, limit):
        # Ví dụ: Trạng thái xấu nếu l và td lớn, v nhỏ (tắc), m và g cao
        status_val = (self.l / 10.0) + (self.td / 30.0) + (self.m) + (1 - self.v / 15.0) + (self.g / 10.0)
        if status_val < limit:
            self.status = "GOOD"
        else:
            self.status = "BAD"
        return self.status