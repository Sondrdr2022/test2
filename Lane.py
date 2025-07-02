class Lane:
    def __init__(self, id, initial_params):
        self.id = id
        # Tham số động
        self.queue_length = initial_params['l']
        self.avg_wait_time = initial_params['tđ']
        self.density = initial_params['m']
        self.avg_speed = initial_params['v']
        self.flow_rate = initial_params['g']
        
        # Điểm chuỗi
        self.score = 0
        self.prev_status = "NEUTRAL"
        self.ambulance_detected = False
        
    def update(self, sensor_data):
        # Cập nhật từ cảm biến IoT
        self.queue_length = sensor_data['l']
        self.avg_wait_time = sensor_data['tđ']
        self.density = sensor_data['m']
        self.avg_speed = sensor_data['v']
        self.flow_rate = sensor_data['g']
        self.ambulance_detected = sensor_data.get('ambulance', False)