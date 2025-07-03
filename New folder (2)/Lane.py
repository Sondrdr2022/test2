import os
import sys
import traci
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import math
import time
import datetime
import pickle
import json
import random
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set SUMO_HOME environment variable if not already set
if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = 'C:/Program Files (x86)/Eclipse/Sumo'

tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
sys.path.append(tools)

# ========================= Q-Learning Agent ==========================
class QLearningAgent:
    """Q-Learning Agent for traffic light control with persistent training data"""
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.95, epsilon=0.1, q_table_file="q_table.pkl"):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: np.zeros(action_size))
        self.training_data = []
        self.q_table_file = q_table_file
        self.load_model(self.q_table_file)
        
    def get_action(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        state_key = self._state_to_key(state)
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            return np.argmax(self.q_table[state_key])
    
    def update_q_table(self, state, action, reward, next_state):
        """Update Q-table using Q-learning formula"""
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)
        current_q = self.q_table[state_key][action]
        max_next_q = np.max(self.q_table[next_state_key])
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state_key][action] = new_q
        # Store training data for incremental dataset
        self.training_data.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'q_value': new_q
        })
    
    def _state_to_key(self, state):
        """Convert state array to hashable key"""
        return tuple(np.round(state, 2))
    
    def save_model(self, filepath=None):
        """Save Q-table and training data, appending to existing data"""
        if filepath is None:
            filepath = self.q_table_file
        # Append new training_data to the old one (persistent dataset)
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            old_training = model_data.get('training_data', [])
        else:
            old_training = []
        model_data = {
            'q_table': dict(self.q_table),
            'training_data': old_training + self.training_data,
            'params': {
                'state_size': self.state_size,
                'action_size': self.action_size,
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'epsilon': self.epsilon
            }
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        self.training_data = []
    
    def load_model(self, filepath=None):
        """Load Q-table and training data, if exists"""
        if filepath is None:
            filepath = self.q_table_file
        try:
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    model_data = pickle.load(f)
                self.q_table = defaultdict(lambda: np.zeros(self.action_size))
                self.q_table.update(model_data.get('q_table', {}))
                self.training_data = []
                params = model_data.get('params', {})
                self.learning_rate = params.get('learning_rate', self.learning_rate)
                self.discount_factor = params.get('discount_factor', self.discount_factor)
                self.epsilon = params.get('epsilon', self.epsilon)
                print(f"Loaded Q-table model with {len(self.q_table)} states from {filepath}")
                return True
        except Exception as e:
            print(f"Error loading model: {e}")
        print("No existing Q-table, starting fresh")
        return False

# ========================= Data Logger ==========================
class DataLogger:
    """Data logging system for RL training, supports continuous append"""
    def __init__(self, log_file='traffic_rl_log.csv'):
        self.log_file = log_file
        self.episode_data = []
        self.current_episode = 0
        if not os.path.exists(log_file):
            self._initialize_log_file()
    
    def _initialize_log_file(self):
        """Initialize CSV log file with headers"""
        headers = [
            'episode', 'time', 'lane_id', 'state', 'action', 'reward', 'next_state', 'ambulance_detected'
        ]
        df = pd.DataFrame(columns=headers)
        df.to_csv(self.log_file, index=False)
        print(f"Initialized log file: {self.log_file}")
    
    def log_step(self, episode, time_step, lane_id, state, action, reward, next_state, ambulance=False):
        """Log a single step of RL training data"""
        log_entry = {
            'episode': episode,
            'time': time_step,
            'lane_id': lane_id,
            'state': json.dumps(state.tolist() if isinstance(state, np.ndarray) else state),
            'action': action,
            'reward': reward,
            'next_state': json.dumps(next_state.tolist() if isinstance(next_state, np.ndarray) else next_state),
            'ambulance_detected': ambulance,
        }
        self.episode_data.append(log_entry)
    
    def save_episode(self):
        """Save accumulated episode data to CSV, append mode"""
        if self.episode_data:
            df = pd.DataFrame(self.episode_data)
            df.to_csv(self.log_file, mode='a', header=False, index=False)
            print(f"Saved {len(self.episode_data)} records for episode {self.current_episode}")
            self.episode_data = []
    
    def load_training_data(self):
        """Load all historical training data"""
        try:
            if os.path.exists(self.log_file):
                df = pd.read_csv(self.log_file)
                print(f"Loaded {len(df)} training records from {self.log_file}")
                return df
            else:
                print("No training data file found")
                return pd.DataFrame()
        except Exception as e:
            print(f"Error loading training data: {e}")
            return pd.DataFrame()

# ========================= Smart Junction Controller RL ==========================
class SmartJunctionControllerRL:
    """Traffic controller with RL, score-chaining, ambulance, dynamic green, dataset increment"""
    def __init__(self, state_size=6, action_size=4):
        self.lane_scores = defaultdict(int)
        self.lane_states = defaultdict(lambda: "UNKNOWN")
        self.consecutive_states = defaultdict(int)
        self.tls_states = {}
        self.phase_durations = {}
        self.last_phase_change = defaultdict(float)
        self.last_green_time = defaultdict(float)
        self.lane_to_tl = {}  # Map lane to traffic light
        self.rl_agent = QLearningAgent(state_size, action_size)
        self.data_logger = DataLogger()
        self.current_episode = 1
        self.step_count = 0
        self.previous_states = {}
        self.previous_actions = {}
        self.priority_vehicles = defaultdict(bool)
        self.ambulance_active = defaultdict(lambda: False)
        self.ambulance_start_time = defaultdict(float)   # or simply: dict()        # RL parameters
        self.limit = [0.7, 0.6]  # threshold for BAD vs GOOD
        self.reward_scale = 50
        self.starvation_threshold = 120  # sec
        self.min_green = 10
        self.max_green = 60
        print("SmartJunctionControllerRL initialized.")

    def run_step(self):
        lane_data = self.collect_lane_data()
        lane_status = self.update_lane_status_and_score(lane_data)
        self.adjust_traffic_lights(lane_data, lane_status)
        self.step_count += 1
    
    def find_phase_for_lane(self, tl_id, target_lane):
        """
        Returns the phase index of tl_id that gives green to target_lane.
        This is a heuristic: it checks which phase has green for the target lane.
        """
        # Get all phases for the traffic light
        logic = traci.trafficlight.getAllProgramLogics(tl_id)[0]
        phases = logic.phases
        controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
        # For each phase, check which lanes are green
        for idx, phase in enumerate(phases):
            state = phase.state  # e.g. "GrGr" for 2-way green
            # For each lane controlled by this traffic light, check if it's green in this phase
            for lane_idx, lane in enumerate(controlled_lanes):
                if lane == target_lane and (state[lane_idx].upper() == 'G'):
                    return idx
        # If not found, return 0 as fallback
        return 0

    def collect_lane_data(self):
        """Collect data for all lanes, including ambulance detection"""
        lane_data = {}
        lanes = traci.lane.getIDList()
        current_time = traci.simulation.getTime()

        for lane_id in lanes:
            try:
                lane_length = traci.lane.getLength(lane_id)
                if lane_length <= 0: continue
                queue_length = traci.lane.getLastStepHaltingNumber(lane_id)
                waiting_time = traci.lane.getWaitingTime(lane_id)
                vehicle_count = traci.lane.getLastStepVehicleNumber(lane_id)
                mean_speed = traci.lane.getLastStepMeanSpeed(lane_id)
                max_speed = traci.lane.getMaxSpeed(lane_id)
                ambulance_detected = False
                vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
                for vehicle_id in vehicles:
                    try:
                        vehicle_class = traci.vehicle.getVehicleClass(vehicle_id)
                        if vehicle_class in ['emergency', 'authority']:
                            ambulance_detected = True
                    except Exception:
                        continue
                self.priority_vehicles[lane_id] = ambulance_detected

                lane_data[lane_id] = {
                    'queue_length': queue_length,
                    'waiting_time': waiting_time,
                    'density': vehicle_count / lane_length if lane_length > 0 else 0,
                    'mean_speed': mean_speed,
                    'flow': vehicle_count,
                    'lane_id': lane_id,
                    'ambulance': ambulance_detected,
                }
            except Exception as e:
                print(f"Lane data error: {e}")
                continue
        return lane_data

    def update_lane_status_and_score(self, lane_data):
        """Update lane scores using chain system and status logic"""
        status = {}
        for lane_id, data in lane_data.items():
            # Status_t = f(l, tđ, m, v, g)
            l = data['queue_length']
            tđ = data['waiting_time']
            m = data['density']
            v = data['mean_speed']
            g = data['flow']
            # Simple normalized congestion indicator (BAD: > limit[0], GOOD < limit[1])
            norm = (l / 15) + (tđ / 60) + (1 - v / (max(v, 1))) + (1 - g / 10)
            if norm > self.limit[0]:
                status[lane_id] = "BAD"
                delta = -min(5, int(norm * 5))
            elif norm < self.limit[1]:
                status[lane_id] = "GOOD"
                delta = min(5, int((1-norm) * 5))
            else:
                status[lane_id] = "NORMAL"
                delta = 0
            # Chain logic
            if self.lane_states[lane_id] == status[lane_id]:
                self.lane_scores[lane_id] += delta
            else:
                self.lane_states[lane_id] = status[lane_id]
                # no point reset, continue chain
        return status

    def ambulance_present_on_any_lane(self, tl_id):
        """Returns True if ambulance is detected on any lane controlled by tl_id."""
        controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
        for lane in controlled_lanes:
            vehicles = traci.lane.getLastStepVehicleIDs(lane)
            for v in vehicles:
                try:
                    if traci.vehicle.getVehicleClass(v) in ['emergency', 'authority']:
                        return True
                except Exception:
                    continue
        return False

    def adjust_traffic_lights(self, lane_data, lane_status):
        tl_ids = traci.trafficlight.getIDList()
        current_time = traci.simulation.getTime()
        for tl_id in tl_ids:
            controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
            ambulance_now = self.ambulance_present_on_any_lane(tl_id)
            if ambulance_now:
                if not self.ambulance_active[tl_id]:
                    ambulance_lane = [lane for lane in controlled_lanes if self.priority_vehicles[lane]][0]
                    phase_index = self.find_phase_for_lane(tl_id, ambulance_lane)
                    traci.trafficlight.setPhase(tl_id, phase_index)
                    traci.trafficlight.setPhaseDuration(tl_id, 1000)
                    self.ambulance_active[tl_id] = True
                    self.ambulance_start_time[tl_id] = current_time
                    print(f"AMBULANCE PRIORITY: Keeping green for {ambulance_lane} at {tl_id}")
                else:
                    if current_time - self.ambulance_start_time[tl_id] > 30:
                        print(f"AMBULANCE OVERRIDE TIMEOUT: Releasing {tl_id} after 30s of stuck")
                        self.ambulance_active[tl_id] = False
                continue
            else:
                if self.ambulance_active[tl_id]:
                    self.ambulance_active[tl_id] = False
                    print(f"AMBULANCE CLEARED: Releasing override at {tl_id}")
                    # --- PATCH START ---
                    # Resume normal operation: reset to next phase or RL control
                    cur_phase = traci.trafficlight.getPhase(tl_id)
                    num_phases = len(traci.trafficlight.getAllProgramLogics(tl_id)[0].phases)
                    traci.trafficlight.setPhase(tl_id, (cur_phase + 1) % num_phases)
                    traci.trafficlight.setPhaseDuration(tl_id, self.min_green)

    def override_green_for_ambulance(self, tl_id, lane_id):
        """Ambulance override: keep green for the lane until no ambulance present"""
        print(f"AMBULANCE PRIORITY: Keeping green for {lane_id} at {tl_id}")
        phases = traci.trafficlight.getAllProgramLogics(tl_id)[0].phases
        cur_phase = traci.trafficlight.getPhase(tl_id)
        # Find phase that serves the ambulance lane
        for idx, ph in enumerate(phases):
            if lane_id in traci.trafficlight.getControlledLanes(tl_id)[idx::len(phases)]:
                traci.trafficlight.setPhase(tl_id, idx)
                traci.trafficlight.setPhaseDuration(tl_id, self.max_green)
                return

    def create_state_vector(self, lane_id, lane_data):
        """Create state vector for lane (for RL state)"""
        data = lane_data[lane_id]
        # t, l, tđ, m, v, g, ambulance
        t = traci.simulation.getTime()
        l = data['queue_length']
        tđ = data['waiting_time']
        m = data['density']
        v = data['mean_speed']
        g = data['flow']
        ambulance = int(data['ambulance'])
        return np.array([t % 1000 / 1000, l / 20, tđ / 120, m / 2, v / 20, g / 20])

    def calculate_reward(self, lane_id, lane_data, action_taken, current_time):
        """Reward: penalize queue, wait, bonus for flow, heavier bonus for ambulance"""
        data = lane_data[lane_id]
        queue_penalty = -data['queue_length'] * 2
        wait_penalty = -data['waiting_time'] * 0.1
        flow_reward = data['flow'] * 4
        ambulance_bonus = 50 if data['ambulance'] else 0
        starvation_penalty = -30 if (current_time - self.last_green_time.get(lane_id,0)) > self.starvation_threshold else 0
        total_reward = (queue_penalty + wait_penalty + flow_reward + ambulance_bonus + starvation_penalty)
        return total_reward / self.reward_scale

    def set_green_phase(self, tl_id, lane_id, green_time):
        """Set the green phase for lane_id for green_time seconds"""
        # For SUMO: cycle through phases, set green for the lane direction
        # You might need to map lane to phase index for a real scenario
        traci.trafficlight.setPhaseDuration(tl_id, green_time)
        self.last_green_time[lane_id] = traci.simulation.getTime()
        self.last_phase_change[tl_id] = traci.simulation.getTime()

    def end_episode(self):
        self.data_logger.save_episode()
        self.rl_agent.save_model()
        print(f"End episode {self.current_episode}")

# ===================== RL Simulation Entrypoint =======================
def start_rl_simulation(sumocfg_path, use_gui=True, max_steps=None, episodes=1):
    try:
        controller = SmartJunctionControllerRL()
        for episode in range(episodes):
            print(f"\n{'='*50}")
            print(f"STARTING RL EPISODE {episode + 1}/{episodes}")
            print(f"{'='*50}")
            sumo_binary = "sumo-gui" if use_gui else "sumo"
            sumo_cmd = [os.path.join(os.environ['SUMO_HOME'], 'bin', sumo_binary),
                        '-c', sumocfg_path,
                        '--start', '--quit-on-end']
            traci.start(sumo_cmd)
            controller.current_episode = episode + 1
            controller.step_count = 0
            controller.previous_states.clear()
            controller.previous_actions.clear()
            step = 0
            while traci.simulation.getMinExpectedNumber() > 0:
                if max_steps and step >= max_steps:
                    print(f"Reached max steps ({max_steps}), ending episode.")
                    break
                try:
                    traci.simulationStep()
                    controller.run_step()
                    step += 1
                    if step % 200 == 0:
                        print(f"Episode {episode + 1}: {step} steps completed...")
                except Exception as e:
                    print(f"Error in simulation step {step}: {e}")
                    break
            print(f"Episode {episode + 1} completed after {step} steps.")
            controller.end_episode()
            traci.close()
            if episode < episodes - 1:
                time.sleep(2)
        print(f"\nAll {episodes} episodes completed!")
    except Exception as e:
        print(f"Error in RL simulation: {e}")
    finally:
        try:
            traci.close()
        except:
            pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run SUMO RL smart traffic simulation")
    parser.add_argument('--sumo', required=True, help='Path to SUMO config file')
    parser.add_argument('--gui', action='store_true', help='Use SUMO GUI')
    parser.add_argument('--max-steps', type=int, help='Maximum simulation steps per episode')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes to run')
    args = parser.parse_args()
    start_rl_simulation(args.sumo, args.gui, args.max_steps, args.episodes)