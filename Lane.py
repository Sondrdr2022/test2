class Lane:
    def __init__(self, id):
        self.id = id
        self.queue_length = 0
        self.avg_wait_time = 0
        self.density = 0
        self.avg_speed = 0
        self.flow_rate = 0
        self.score = 0
        self.prev_status = "NEUTRAL"
        self.ambulance_detected = False

    def update(self, sensor_data):
        self.queue_length = sensor_data['l']
        self.avg_wait_time = sensor_data['tđ']
        self.density = sensor_data['m']
        self.avg_speed = sensor_data['v']
        self.flow_rate = sensor_data['g']
        self.ambulance_detected = sensor_data.get('ambulance', False)