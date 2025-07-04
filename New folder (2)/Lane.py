import os
import sys
import traci
import numpy as np
import pandas as pd
from collections import defaultdict
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
    os.environ['SUMO_HOME'] = r'C:\Program Files (x86)\Eclipse\Sumo'

tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
if tools not in sys.path:
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
        self._loaded_training_count = 0
        self.load_model()  # Load existing model if available

    def get_action(self, state, lane_id=None, training=True):  # CHANGE: add lane_id param
        state_key = self._state_to_key(state, lane_id)  # CHANGE: use lane_id in key
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(0, self.action_size)
        else:
            return np.argmax(self.q_table[state_key])

    def update_q_table(self, state, action, reward, next_state, lane_info=None):
        """Update Q-table using Q-learning formula, extended for lane/route info"""
        # CHANGE: Use lane_id in Q-table key for per-lane learning
        lane_id = lane_info['lane_id'] if lane_info and 'lane_id' in lane_info else None
        state_key = self._state_to_key(state, lane_id)
        next_state_key = self._state_to_key(next_state, lane_id)  # next_state for same lane
        current_q = self.q_table[state_key][action]
        max_next_q = np.max(self.q_table[next_state_key])
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state_key][action] = new_q
        # Store training data for incremental dataset (EXTENDED)
        entry = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'q_value': new_q
        }
        if lane_info:  # Include extra lane/route info if provided
            entry.update(lane_info)
        self.training_data.append(entry)

    def save_model(self, filepath=None):
        """Save Q-table and training data, appending to existing data"""
        if filepath is None:
            filepath = self.q_table_file
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

            
    def _state_to_key(self, state, lane_id=None):  # CHANGE: lane_id as part of key
        key = tuple(np.round(state, 2))
        if lane_id is not None:
            return (lane_id, key)
        return key
    
    '''
        # Initialize with current data
        model_data = {
            'q_table': dict(self.q_table),
            'training_data': self.training_data.copy(),
            'params': {
                'state_size': self.state_size,
                'action_size': self.action_size,
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'epsilon': self.epsilon
            }
        }

        # Merge with existing data if available
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    existing_data = pickle.load(f)

                # Merge Q-tables (current takes precedence)
                existing_q_table = existing_data.get('q_table', {})
                model_data['q_table'].update(existing_q_table)

                # Get existing training data and remove duplicates
                existing_training = existing_data.get('training_data', [])
                existing_keys = set()
                for entry in existing_training:
                    state_key = tuple(np.round(entry['state'], 4)) if isinstance(entry['state'], (np.ndarray, list)) else str(entry['state'])
                    action_key = entry['action']
                    next_state_key = tuple(np.round(entry['next_state'], 4)) if isinstance(entry['next_state'], (np.ndarray, list)) else str(entry['next_state'])
                    existing_keys.add((state_key, action_key, next_state_key))

                unique_new_training = []
                for entry in self.training_data:
                    state_key = tuple(np.round(entry['state'], 4)) if isinstance(entry['state'], (np.ndarray, list)) else str(entry['state'])
                    action_key = entry['action']
                    next_state_key = tuple(np.round(entry['next_state'], 4)) if isinstance(entry['next_state'], (np.ndarray, list)) else str(entry['next_state'])
                    if (state_key, action_key, next_state_key) not in existing_keys:
                        unique_new_training.append(entry)

                # Combine old and new unique training data
                model_data['training_data'] = existing_training + unique_new_training

            except Exception as e:
                print(f"Warning: Error merging with existing model data: {e}")

        # Save the combined data
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"✅ Model saved with {len(model_data['training_data'])} training entries (added {len(self.training_data)})")
        self.training_data = []
        '''
    def load_model(self, filepath=None):
        """Load Q-table and training data if exists"""
        if filepath is None:
            filepath = self.q_table_file
        try:
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    model_data = pickle.load(f)
                # CHANGE: Use default dict for lane+state Q-table
                self.q_table = defaultdict(lambda: np.zeros(self.action_size))
                loaded_q_table = model_data.get('q_table', {})
                self.q_table.update(loaded_q_table)
                self._loaded_training_count = len(model_data.get('training_data', []))
                params = model_data.get('params', {})
                self.learning_rate = params.get('learning_rate', self.learning_rate)
                self.discount_factor = params.get('discount_factor', self.discount_factor)
                self.epsilon = params.get('epsilon', self.epsilon)
                print(f"Loaded Q-table with {len(self.q_table)} states from {filepath} (prev training count: {self._loaded_training_count})")
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
        headers = [
            'episode', 'time', 'lane_id', 'edge_id',  # CHANGE: add edge_id
            'state', 'action', 'reward', 'next_state', 'q_value', 'ambulance_detected'
        ]
        df = pd.DataFrame(columns=headers)
        df.to_csv(self.log_file, index=False)
        print(f"Initialized log file: {self.log_file}")

    def log_step(self, episode, time_step, lane_id, edge_id, state, action, reward, next_state, q_value, ambulance=False):
        # CHANGE: log edge_id and q_value as well
        log_entry = {
            'episode': episode,
            'time': time_step,
            'lane_id': lane_id,
            'edge_id': edge_id,
            'state': json.dumps(state.tolist() if isinstance(state, np.ndarray) else state),
            'action': action,
            'reward': reward,
            'next_state': json.dumps(next_state.tolist() if isinstance(next_state, np.ndarray) else next_state),
            'q_value': q_value,
            'ambulance_detected': ambulance,
        }
        self.episode_data.append(log_entry)

    def save_episode(self):
        if self.episode_data:
            df = pd.DataFrame(self.episode_data)
            df.to_csv(self.log_file, mode='a', header=False, index=False)
            print(f"Saved {len(self.episode_data)} records for episode {self.current_episode}")
            self.episode_data = []

    def load_training_data(self):
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
        self.lane_to_tl = {}
        self.rl_agent = QLearningAgent(state_size, action_size)
        self.data_logger = DataLogger()
        self.current_episode = 1
        self.step_count = 0
        self.previous_states = {}
        self.previous_actions = {}
        self.priority_vehicles = defaultdict(bool)
        self.ambulance_active = defaultdict(lambda: False)
        self.ambulance_start_time = defaultdict(float)
        self.limit = [0.7, 0.6]
        self.reward_scale = 50
        self.starvation_threshold = 120
        self.min_green = 10
        self.max_green = 60
        print("SmartJunctionControllerRL initialized.")

    def run_step(self):
        lane_data = self.collect_lane_data()
        lane_status = self.update_lane_status_and_score(lane_data)
        self.adjust_traffic_lights(lane_data, lane_status)
        self.step_count += 1

        # Per-lane RL step and logging
        for lane_id, data in lane_data.items():
            state = self.create_state_vector(lane_id, lane_data)
            action = self.rl_agent.get_action(state, lane_id=lane_id)
            # (simulate action, get reward, get next_state)
            next_state = state  # replace with actual next_state computation
            reward = 0  # replace with actual reward computation
            tl_id = self.lane_to_tl.get(lane_id, None)
            c = traci.trafficlight.getPhase(tl_id) if tl_id else -1
            lane_info = {
                'lane_id': lane_id,
                'edge_id': data['edge_id'],
                'c': c,  # current signal phase
                'm': data['density'],
                'v': data['mean_speed'],
                'g': data['flow'],
                'queue_lane': data['queue_length'],
                'queue_route': data['queue_route'],
                'flow_route': data['flow_route'],
                'wait_lane': data['waiting_time'],
            }
            self.rl_agent.update_q_table(state, action, reward, next_state, lane_info=lane_info)
            # Log step with lane/edge info
            self.data_logger.log_step(
                self.current_episode, traci.simulation.getTime(), lane_id, data['edge_id'],
                state, action, reward, next_state, self.rl_agent.q_table[self.rl_agent._state_to_key(state, lane_id)][action],
                ambulance=data['ambulance']
            )

    def find_phase_for_lane(self, tl_id, target_lane):
        logic = traci.trafficlight.getAllProgramLogics(tl_id)[0]
        phases = logic.phases
        controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
        for idx, phase in enumerate(phases):
            state = phase.state
            for lane_idx, lane in enumerate(controlled_lanes):
                if lane == target_lane and (state[lane_idx].upper() == 'G'):
                    return idx
        return 0

    def collect_lane_data(self):
        """Collect data for all lanes, including ambulance detection and extra info"""
        lane_data = {}
        lanes = traci.lane.getIDList()
        current_time = traci.simulation.getTime()
        edge_queues = defaultdict(float)
        edge_flows = defaultdict(float)
        lane_to_edge = {}

        # First pass: collect all lane data, and sum queue/flow per edge
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

                # Get edge/route id for the lane
                edge_id = traci.lane.getEdgeID(lane_id)
                lane_to_edge[lane_id] = edge_id

                edge_queues[edge_id] += queue_length
                edge_flows[edge_id] += vehicle_count

                lane_data[lane_id] = {
                    'queue_length': queue_length,
                    'waiting_time': waiting_time,
                    'density': vehicle_count / lane_length if lane_length > 0 else 0,
                    'mean_speed': mean_speed,
                    'flow': vehicle_count,
                    'lane_id': lane_id,
                    'edge_id': edge_id,
                    'ambulance': ambulance_detected,
                }
            except Exception as e:
                print(f"Lane data error: {e}")
                continue

        # Add edge/route queue/flow to lane data
        for lane_id, data in lane_data.items():
            edge_id = data['edge_id']
            data['queue_route'] = edge_queues[edge_id]
            data['flow_route'] = edge_flows[edge_id]
        return lane_data

    def update_lane_status_and_score(self, lane_data):
        status = {}
        for lane_id, data in lane_data.items():
            l = data['queue_length']
            tđ = data['waiting_time']
            m = data['density']
            v = data['mean_speed']
            g = data['flow']
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
            if self.lane_states[lane_id] == status[lane_id]:
                self.lane_scores[lane_id] += delta
            else:
                self.lane_states[lane_id] = status[lane_id]
        return status

    def ambulance_present_on_any_lane(self, tl_id):
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
                    cur_phase = traci.trafficlight.getPhase(tl_id)
                    num_phases = len(traci.trafficlight.getAllProgramLogics(tl_id)[0].phases)
                    traci.trafficlight.setPhase(tl_id, (cur_phase + 1) % num_phases)
                    traci.trafficlight.setPhaseDuration(tl_id, self.min_green)

    def create_state_vector(self, lane_id, lane_data):
        data = lane_data[lane_id]
        t = traci.simulation.getTime()
        l = data['queue_length']
        tđ = data['waiting_time']
        m = data['density']
        v = data['mean_speed']
        g = data['flow']
        ambulance = int(data['ambulance'])
        return np.array([t % 1000 / 1000, l / 20, tđ / 120, m / 2, v / 20, g / 20])

    def calculate_reward(self, lane_id, lane_data, action_taken, current_time):
        data = lane_data[lane_id]
        queue_penalty = -data['queue_length'] * 2
        wait_penalty = -data['waiting_time'] * 0.1
        flow_reward = data['flow'] * 4
        ambulance_bonus = 50 if data['ambulance'] else 0
        starvation_penalty = -30 if (current_time - self.last_green_time.get(lane_id, 0)) > self.starvation_threshold else 0
        total_reward = (queue_penalty + wait_penalty + flow_reward + ambulance_bonus + starvation_penalty)
        return total_reward / self.reward_scale

    def set_green_phase(self, tl_id, lane_id, green_time):
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

def analyze_training_progression(pkl_file='q_table.pkl'):
    """Analyze if training data is actually progressing"""
    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        training_data = data.get('training_data', [])
        if not training_data:
            print("❌ No training data found!")
            return
        print(f"📊 TRAINING DATA ANALYSIS")
        print(f"="*50)
        print(f"Total entries: {len(training_data)}")
        rewards = [entry.get('reward', 0) for entry in training_data]
        q_values = [entry.get('q_value', 0) for entry in training_data]
        actions = [entry.get('action', 0) for entry in training_data]
        print(f"\n🔍 DATA VARIETY CHECK:")
        print(f"Unique reward values: {len(set(rewards))}")
        print(f"Unique Q-values: {len(set(q_values))}")
        print(f"Unique actions: {len(set(actions))}")
        if len(set(rewards)) == 1 and len(set(q_values)) == 1:
            print("⚠️  WARNING: All rewards and Q-values are identical!")
            print("   This suggests no new learning is happening.")
        print(f"\n📈 PROGRESSION ANALYSIS:")
        if len(training_data) >= 20:
            first_10_rewards = [training_data[i].get('reward', 0) for i in range(10)]
            last_10_rewards = [training_data[i].get('reward', 0) for i in range(-10, 0)]
            print(f"First 10 avg reward: {np.mean(first_10_rewards):.6f}")
            print(f"Last 10 avg reward: {np.mean(last_10_rewards):.6f}")
            if first_10_rewards == last_10_rewards:
                print("⚠️  WARNING: First and last 10 entries are identical!")
                print("   Your agent is not generating new training data.")
        print(f"\n🕐 TIMESTAMP CHECK:")
        states = []
        for entry in training_data[:5]:
            state = entry.get('state', [])
            if isinstance(state, str):
                try:
                    state = json.loads(state)
                except:
                    continue
            if len(state) > 0:
                states.append(state[0])
        if states and len(set(states)) == 1:
            print("⚠️  WARNING: All timestamps are identical!")
            print("   This confirms the simulation is not progressing.")
        else:
            print(f"✅ Found {len(set(states))} different timestamps")
        print(f"\n📋 SAMPLE ENTRIES:")
        for i in [0, len(training_data)//2, -1]:
            entry = training_data[i]
            print(f"Entry {i}: action={entry.get('action')}, "
                  f"reward={entry.get('reward'):.6f}, "
                  f"q_value={entry.get('q_value'):.6f}")
        return training_data
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run SUMO RL smart traffic simulation")
    parser.add_argument('--sumo', required=True, help='Path to SUMO config file')
    parser.add_argument('--gui', action='store_true', help='Use SUMO GUI')
    parser.add_argument('--max-steps', type=int, help='Maximum simulation steps per episode')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes to run')
    parser.add_argument('--analyze', action='store_true', help='Analyze q_table.pkl after training')
    args = parser.parse_args()
    if args.analyze:
        analyze_training_progression('q_table.pkl')
    else:
        start_rl_simulation(args.sumo, args.gui, args.max_steps, args.episodes)