import os
import sys
import traci
from collections import defaultdict

# --- Cấu hình SUMO ---
# Kiểm tra biến môi trường SUMO_HOME
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Vui lòng khai báo biến môi trường 'SUMO_HOME'")

# --- Các tham số mô phỏng và ngưỡng ---
# Ngưỡng để xác định trạng thái làn đường (BAD/GOOD)
# Đây là một công thức ví dụ, cần được hiệu chỉnh qua thực nghiệm
# Trọng số cho các tham số: w1*độ dài hàng đợi + w2*thời gian chờ
# Bạn có thể thêm các tham số khác như m, v, g
LANE_STATUS_THRESHOLD = 25.0 
W_QUEUE = 0.7  # Trọng số cho độ dài hàng đợi
W_WAIT_TIME = 0.3 # Trọng số cho thời gian chờ

# Thời gian đèn xanh tối thiểu và tối đa
MIN_GREEN_TIME = 10
MAX_GREEN_TIME = 60

# --- Lớp quản lý trạng thái và điểm ---
class LaneManager:
    """Quản lý trạng thái và điểm cho tất cả các làn đường."""
    def __init__(self):
        self.lane_scores = defaultdict(float)
        self.consecutive_status = defaultdict(int) # Đếm số chu kỳ liên tiếp ở một trạng thái
        self.current_status = defaultdict(str)

    def get_lane_status(self, lane_id):
        """
        Tính toán trạng thái của một làn (GOOD/BAD) dựa trên các tham số thời gian thực.
        Đây là nơi bạn có thể tối ưu hóa công thức tính 'Status_t'.
        """
        try:
            # l: Độ dài hàng đợi (số xe)
            queue_length = traci.lane.getLastStepHaltingNumber(lane_id)
            # tđ: Thời gian chờ trung bình của xe đầu hàng
            waiting_time = traci.lane.getWaitingTime(lane_id)
            
            # Công thức tính điểm trạng thái (Status_t)
            status_score = (W_QUEUE * queue_length) + (W_WAIT_TIME * waiting_time)
            
            if status_score < LANE_STATUS_THRESHOLD:
                return "GOOD", status_score
            else:
                return "BAD", status_score
        except traci.TraCIException:
            return "UNKNOWN", 0

    def update_all_lane_scores(self, lanes):
        """
        Cập nhật điểm chuỗi cho tất cả các làn dựa trên trạng thái hiện tại.
        """
        for lane_id in lanes:
            new_status, _ = self.get_lane_status(lane_id)
            if new_status == "UNKNOWN":
                continue

            last_status = self.current_status[lane_id]

            if new_status == last_status:
                self.consecutive_status[lane_id] += 1
            else:
                # Trạng thái thay đổi, reset bộ đếm
                self.current_status[lane_id] = new_status
                self.consecutive_status[lane_id] = 1

            # Tính điểm cộng/trừ
            # Điểm tăng/giảm từ 1 đến 5 dựa trên số chu kỳ liên tiếp
            score_change = min(self.consecutive_status[lane_id], 5)

            if new_status == "GOOD":
                self.lane_scores[lane_id] += score_change
            elif new_status == "BAD":
                self.lane_scores[lane_id] -= score_change


class TrafficLightController:
    """Điều khiển logic cho một nút giao thông."""
    def __init__(self, tls_id, lane_manager):
        self.id = tls_id
        self.lanes = traci.trafficlight.getControlledLanes(self.id)
        self.lane_manager = lane_manager
        # Lưu trữ chương trình đèn mặc định để quay lại sau khi xe ưu tiên đi qua
        self.default_logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.id)[0]

    def has_emergency_vehicle(self):
        """Kiểm tra xem có xe ưu tiên nào trên các làn được kiểm soát không."""
        for lane_id in self.lanes:
            vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
            for vehicle_id in vehicles:
                # Giả sử xe ưu tiên có vClass là "emergency" hoặc "ambulance"
                if traci.vehicle.getVehicleClass(vehicle_id) in ["emergency", "ambulance"]:
                    print(f"Phát hiện xe ưu tiên {vehicle_id} trên làn {lane_id} tại nút {self.id}")
                    return lane_id, vehicle_id
        return None, None
        
    def grant_priority_to_lane(self, priority_lane):
        """Bật đèn xanh cho làn ưu tiên và đèn đỏ cho các làn khác."""
        # Tìm phase tương ứng với việc bật đèn xanh cho làn ưu tiên
        # Đây là một logic đơn giản, có thể cần phức tạp hơn tùy vào cấu trúc nút giao
        green_phase_found = False
        for i, phase in enumerate(self.default_logic.phases):
            state = phase.state
            # Tìm phase mà làn ưu tiên có đèn xanh (G hoặc g)
            try:
                lane_index = self.lanes.index(priority_lane)
                if state[lane_index].lower() == 'g':
                    traci.trafficlight.setPhase(self.id, i)
                    green_phase_found = True
                    break
            except (ValueError, IndexError):
                continue # Làn không được kiểm soát trực tiếp bởi phase này
        
        if not green_phase_found:
            print(f"Cảnh báo: Không tìm thấy phase đèn xanh phù hợp cho làn ưu tiên {priority_lane}")


    def adjust_lights_based_on_scores(self):
        """
        Phân phối lại thời gian đèn xanh dựa trên điểm tích lũy của các làn.
        Đây là nơi có thể tích hợp mô hình Reinforcement Learning.
        """
        total_score = 0
        lane_scores = {}
        for lane_id in self.lanes:
            score = abs(self.lane_manager.lane_scores[lane_id])
            lane_scores[lane_id] = score
            total_score += score
        
        if total_score == 0:
            return # Không có gì để điều chỉnh

        # Phân phối lại tổng thời gian của một chu kỳ (ví dụ: 120 giây)
        total_cycle_time = sum([p.duration for p in self.default_logic.phases if 'y' not in p.state.lower()])
        
        # Logic phân phối thời gian (ví dụ)
        # Làn nào có điểm 'BAD' (điểm âm, abs() thành dương) cao hơn sẽ được nhiều thời gian xanh hơn
        # Cần một logic phức tạp hơn để nhóm các làn thành các pha tương thích
        # Ví dụ đơn giản: chỉ điều chỉnh thời gian của các pha hiện có
        current_logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(self.id)[0]
        
        for i, phase in enumerate(current_logic.phases):
            # Chỉ điều chỉnh pha đèn xanh (không có đèn vàng)
            if 'y' in phase.state.lower() or 'g' not in phase.state.lower():
                continue

            # Tính điểm trung bình cho các làn trong pha này
            phase_score = 1.0 # Điểm cơ sở
            green_lanes_in_phase = [self.lanes[j] for j, s in enumerate(phase.state) if s.lower() == 'g']
            
            if not green_lanes_in_phase:
                continue

            for lane_id in green_lanes_in_phase:
                phase_score += lane_scores.get(lane_id, 0)
            
            # Phân bổ thời gian mới
            new_duration = (phase_score / total_score) * total_cycle_time
            # Giới hạn thời gian trong khoảng cho phép
            phase.duration = max(MIN_GREEN_TIME, min(new_duration, MAX_GREEN_TIME))

        # Áp dụng logic mới
        traci.trafficlight.setCompleteRedYellowGreenDefinition(self.id, current_logic)


def run_simulation():
    """Hàm chính để chạy mô phỏng."""
    # Bước 1: Khởi động SUMO GUI và kết nối với Traci
    # Thay 'your_config_file.sumocfg' bằng tệp cấu hình của bạn
    traci.start(["sumo-gui", "-c", r"C:\Users\Admin\Downloads\sumo test\New folder\dataset1.sumocfg", "--step-length", "1"])

    # Bước 2: Tự động nhận diện các thành phần mạng lưới
    all_lanes = traci.lane.getIDList()
    all_tls_ids = traci.trafficlight.getIDList()
    
    # Khởi tạo các đối tượng quản lý
    lane_manager = LaneManager()
    tls_controllers = {tls_id: TrafficLightController(tls_id, lane_manager) for tls_id in all_tls_ids}
    
    step = 0
    # Chu kỳ để cập nhật điểm và điều chỉnh đèn (ví dụ: mỗi 60 bước)
    update_cycle = 60 

    # --- Vòng lặp mô phỏng chính ---
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

        if step % 10 == 0: # In ra mỗi 10 bước để không làm chậm mô phỏng
            test_lane_id = ':E5-3_0' # <-- Thay bằng ID làn bạn muốn kiểm tra
            if test_lane_id in all_lanes:
                q_len = traci.lane.getLastStepHaltingNumber(test_lane_id)
                wait_t = traci.lane.getWaitingTime(test_lane_id)
                status_val = (W_QUEUE * q_len) + (W_WAIT_TIME * wait_t)
                print(f"[DEBUG] Làn {test_lane_id} | Hàng đợi: {q_len} xe | T.gian chờ: {wait_t:.2f}s | Điểm trạng thái: {status_val:.2f}")
        
        # Xử lý ưu tiên cho xe cứu thương (kiểm tra mỗi bước)
        emergency_handled_this_step = False
        for tls_id, controller in tls_controllers.items():
            priority_lane, vehicle_id = controller.has_emergency_vehicle()
            if priority_lane:
                controller.grant_priority_to_lane(priority_lane)
                emergency_handled_this_step = True
        
        # Nếu có xe ưu tiên, bỏ qua logic điều chỉnh thông thường
        if emergency_handled_this_step:
            continue

        # Cập nhật điểm và điều chỉnh đèn theo chu kỳ
        if step % update_cycle == 0:
            print(f"\n--- Cập nhật tại bước {step} ---")
            # 1. Cập nhật điểm cho tất cả các làn
            lane_manager.update_all_lane_scores(all_lanes)
            
            # 2. Điều chỉnh đèn dựa trên điểm số mới
            for tls_id, controller in tls_controllers.items():
                # Quay lại logic mặc định nếu không có áp lực giao thông đáng kể
                # hoặc sau khi xử lý xe ưu tiên
                traci.trafficlight.setProgram(tls_id, controller.default_logic.programID)
                controller.adjust_lights_based_on_scores()
            
            # In ra một vài điểm số để theo dõi
            print("Điểm số một vài làn:", dict(list(lane_manager.lane_scores.items())[:5]))

        step += 1

    traci.close()
    print("Mô phỏng kết thúc.")

if __name__ == "__main__":
    # Bạn cần có một file cấu hình SUMO, ví dụ: your_config_file.sumocfg
    # File này sẽ trỏ đến các file mạng lưới (.net.xml) và luồng xe (.rou.xml)
    # Hãy chắc chắn rằng bạn đã định nghĩa vClass="emergency" cho các xe ưu tiên trong file route.
    run_simulation()
