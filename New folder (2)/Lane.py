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
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set SUMO_HOME environment variable
if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = r'C:\Program Files (x86)\Eclipse\Sumo'

tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
if tools not in sys.path:
    sys.path.append(tools)

class EnhancedQLearningAgent:
    """Enhanced Q-Learning Agent with adaptive learning and priority handling"""
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.95, 
                 epsilon=0.1, epsilon_decay=0.995, min_epsilon=0.01, 
                 q_table_file="enhanced_q_table.pkl"):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = defaultdict(lambda: np.zeros(action_size))
        self.training_data = []
        self.q_table_file = q_table_file
        self._loaded_training_count = 0
        
        # Adaptive learning parameters
        self.reward_history = []
        self.learning_rate_decay = 0.999
        self.min_learning_rate = 0.01
        self.consecutive_no_improvement = 0
        self.max_no_improvement = 100

    def is_valid_state(self, state):
        """Check if state contains valid values (no NaN or extreme values)"""
        if isinstance(state, (list, np.ndarray)):
            state_array = np.array(state)
            if np.isnan(state_array).any() or np.isinf(state_array).any():
                return False
            if (np.abs(state_array) > 100).any():  # Unrealistic values
                return False
        return True

    def get_action(self, state, lane_id=None, training=True):
        if not self.is_valid_state(state):
            return 0  # Default action for invalid state
        
        state_key = self._state_to_key(state, lane_id)
        
        # Exploration vs exploitation
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(0, self.action_size)
        else:
            return np.argmax(self.q_table[state_key])

    def _state_to_key(self, state, lane_id=None):
        """Convert state to hashable key"""
        try:
            if isinstance(state, np.ndarray):
                key = tuple(np.round(state, 2))
            elif isinstance(state, list):
                state_array = np.round(np.array(state), 2)
                key = tuple(state_array.tolist())
            else:
                key = tuple(state) if hasattr(state, '__iter__') else (state,)
            
            if lane_id is not None:
                return (lane_id, key)
            return key
        except Exception:
            return (lane_id, (0,)) if lane_id is not None else (0,)

    def update_q_table(self, state, action, reward, next_state, lane_info=None):
        """Enhanced Q-learning update with adaptive learning"""
        if not self.is_valid_state(state) or not self.is_valid_state(next_state):
            return
        
        if np.isnan(reward) or np.isinf(reward):
            reward = 0.0
        
        lane_id = lane_info['lane_id'] if lane_info and 'lane_id' in lane_info else None
        state_key = self._state_to_key(state, lane_id)
        next_state_key = self._state_to_key(next_state, lane_id)
        
        # Adaptive learning rate
        current_q = self.q_table[state_key][action]
        max_next_q = np.max(self.q_table[next_state_key])
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        
        if np.isnan(new_q) or np.isinf(new_q):
            new_q = current_q
        
        self.q_table[state_key][action] = new_q
        
        # Store training data with additional context
        entry = {
            'state': state.tolist() if isinstance(state, np.ndarray) else state,
            'action': action,
            'reward': reward,
            'next_state': next_state.tolist() if isinstance(next_state, np.ndarray) else next_state,
            'q_value': new_q,
            'timestamp': time.time(),
            'learning_rate': self.learning_rate,
            'epsilon': self.epsilon
        }
        if lane_info:
            entry.update(lane_info)
        self.training_data.append(entry)
        
        # Update adaptive parameters
        self._update_adaptive_parameters(reward)

    def _update_adaptive_parameters(self, reward):
        """Adjust learning parameters based on performance"""
        self.reward_history.append(reward)
        if len(self.reward_history) > 100:
            self.reward_history.pop(0)
            
        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        # Adjust learning rate based on performance
        if len(self.reward_history) >= 50:
            recent_avg = np.mean(self.reward_history[-50:])
            older_avg = np.mean(self.reward_history[-100:-50])
            
            if recent_avg <= older_avg:
                self.consecutive_no_improvement += 1
            else:
                self.consecutive_no_improvement = 0
                
            if self.consecutive_no_improvement > self.max_no_improvement:
                self.learning_rate = max(self.min_learning_rate, 
                                       self.learning_rate * self.learning_rate_decay)
                self.consecutive_no_improvement = 0

    def load_model(self, filepath=None):
        """Load Q-table, training data, and adaptive parameters if exists"""
        if filepath is None:
            filepath = self.q_table_file
        try:
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    model_data = pickle.load(f)
                # Use default dict for lane+state Q-table
                self.q_table = defaultdict(lambda: np.zeros(self.action_size))
                loaded_q_table = model_data.get('q_table', {})
                self.q_table.update(loaded_q_table)
                self._loaded_training_count = len(model_data.get('training_data', []))
                params = model_data.get('params', {})
                self.learning_rate = params.get('learning_rate', self.learning_rate)
                self.discount_factor = params.get('discount_factor', self.discount_factor)
                self.epsilon = params.get('epsilon', self.epsilon)
                
                # Return both success status and adaptive params
                adaptive_params = model_data.get('adaptive_params', {})
                print(f"Loaded Q-table with {len(self.q_table)} states from {filepath}")
                if adaptive_params:
                    print(f"📋 Loaded adaptive parameters from previous run")
                
                return True, adaptive_params
        except Exception as e:
            print(f"Error loading model: {e}")
        print("No existing Q-table, starting fresh")
        return False, {}

    def save_model(self, filepath=None, adaptive_params=None):
        """Save model with versioning, backup, and adaptive parameters"""
        if filepath is None:
            filepath = self.q_table_file
            
        try:
            # Create backup if file exists
            if os.path.exists(filepath):
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = f"{filepath}.bak_{timestamp}"
                
                # Retry backup operation
                for _ in range(3):
                    try:
                        os.rename(filepath, backup_path)
                        break
                    except Exception as e:
                        print(f"Retrying backup: {e}")
                        time.sleep(0.5)
            
            model_data = {
                'q_table': dict(self.q_table),
                'training_data': self.training_data,
                'params': {
                    'state_size': self.state_size,
                    'action_size': self.action_size,
                    'learning_rate': self.learning_rate,
                    'discount_factor': self.discount_factor,
                    'epsilon': self.epsilon,
                    'epsilon_decay': self.epsilon_decay,
                    'min_epsilon': self.min_epsilon
                },
                'metadata': {
                    'last_updated': datetime.datetime.now().isoformat(),
                    'training_count': len(self.training_data),
                    'average_reward': np.mean([x['reward'] for x in self.training_data[-100:]]) if self.training_data else 0
                }
            }
            
            # Add adaptive parameters to the saved data
            if adaptive_params:
                model_data['adaptive_params'] = adaptive_params.copy()
                print(f"💾 Saving adaptive parameters: {adaptive_params}")
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            print(f"✅ Model saved with {len(self.training_data)} training entries")
            self.training_data = []  # Clear only after successful save
            
        except Exception as e:
            print(f"Error saving model: {e}")

class EnhancedDataLogger:
    """Enhanced data logger with real-time analysis"""
    def __init__(self, log_file='enhanced_traffic_rl_log.csv'):
        self.log_file = log_file
        self.episode_data = []
        self.current_episode = 0
        self._initialize_log_file()
        
        # Real-time analysis buffers
        self.reward_buffer = []
        self.action_distribution = defaultdict(int)
        self.state_statistics = defaultdict(list)

    def _initialize_log_file(self):
        headers = [
            'episode', 'timestamp', 'simulation_time', 'lane_id', 'edge_id', 'route_id',
            'state', 'action', 'reward', 'next_state', 'q_value', 
            'ambulance_detected', 'left_turn', 'phase_id', 'tl_id',
            'queue_length', 'waiting_time', 'density', 'flow', 'speed',
            'queue_route', 'flow_route', 'learning_params'
        ]
        
        if not os.path.exists(self.log_file):
            pd.DataFrame(columns=headers).to_csv(self.log_file, index=False)
            print(f"Initialized enhanced log file: {self.log_file}")

    def log_step(self, episode, time_step, lane_info, state, action, reward, next_state, q_value):
        """Enhanced logging with more contextual information"""
        log_entry = {
            'episode': episode,
            'timestamp': datetime.datetime.now().isoformat(),
            'simulation_time': time_step,
            'lane_id': lane_info.get('lane_id', ''),
            'edge_id': lane_info.get('edge_id', ''),
            'route_id': lane_info.get('route_id', ''),
            'state': json.dumps(state.tolist() if isinstance(state, np.ndarray) else state),
            'action': action,
            'reward': reward,
            'next_state': json.dumps(next_state.tolist() if isinstance(next_state, np.ndarray) else next_state),
            'q_value': q_value,
            'ambulance_detected': lane_info.get('ambulance', False),
            'left_turn': lane_info.get('left_turn', False),
            'phase_id': lane_info.get('phase_id', -1),
            'tl_id': lane_info.get('tl_id', ''),
            'queue_length': lane_info.get('queue_length', 0),
            'waiting_time': lane_info.get('waiting_time', 0),
            'density': lane_info.get('density', 0),
            'flow': lane_info.get('flow', 0),
            'speed': lane_info.get('mean_speed', 0),
            'queue_route': lane_info.get('queue_route', 0),
            'flow_route': lane_info.get('flow_route', 0),
            'learning_params': json.dumps({
                'epsilon': lane_info.get('epsilon', 0),
                'learning_rate': lane_info.get('learning_rate', 0),
                'adaptive_params': lane_info.get('adaptive_params', {})
            })
        }
        
        self.episode_data.append(log_entry)
        
        # Update real-time analysis
        self._update_realtime_stats(action, reward, state)

    def _update_realtime_stats(self, action, reward, state):
        """Update real-time performance statistics"""
        self.reward_buffer.append(reward)
        self.action_distribution[action] += 1
        
        # Keep only recent data for statistics
        if len(self.reward_buffer) > 1000:
            self.reward_buffer.pop(0)
            
        # Track state statistics
        if isinstance(state, (list, np.ndarray)):
            for i, val in enumerate(state):
                self.state_statistics[f'state_{i}'].append(val)
                if len(self.state_statistics[f'state_{i}']) > 1000:
                    self.state_statistics[f'state_{i}'].pop(0)

    def get_performance_summary(self):
        """Return current performance metrics"""
        if not self.reward_buffer:
            return {}
            
        return {
            'avg_reward': np.mean(self.reward_buffer),
            'min_reward': np.min(self.reward_buffer),
            'max_reward': np.max(self.reward_buffer),
            'action_distribution': dict(self.action_distribution),
            'state_stats': {k: {'mean': np.mean(v), 'std': np.std(v)} 
                           for k, v in self.state_statistics.items()}
        }

    def save_episode(self):
        if self.episode_data:
            try:
                # Append to existing CSV
                df = pd.DataFrame(self.episode_data)
                df.to_csv(self.log_file, mode='a', header=False, index=False)
                
                # Save a compressed version periodically
                if self.current_episode % 10 == 0:
                    compressed_file = f"compressed_{self.log_file}"
                    df.to_csv(compressed_file, mode='a', header=not os.path.exists(compressed_file), 
                             index=False, compression='gzip')
                
                print(f"✅ Saved {len(self.episode_data)} records for episode {self.current_episode}")
                self.episode_data = []
                
            except Exception as e:
                print(f"Error saving episode data: {e}")

class SmartTrafficController:
    """Enhanced traffic controller with dynamic parameter adjustment"""
    def __init__(self, state_size=12, action_size=5):
        # Core components
        self.rl_agent = EnhancedQLearningAgent(state_size, action_size)
        self.data_logger = EnhancedDataLogger()
        
        # Traffic state tracking
        self.lane_scores = defaultdict(int)
        self.lane_states = defaultdict(lambda: "UNKNOWN")
        self.consecutive_states = defaultdict(int)
        self.lane_to_tl = {}
        self.edge_to_routes = defaultdict(set)
        
        # RL learning state tracking
        self.previous_states = {}
        self.previous_actions = {}
        self.current_episode = 0
        
        # Timing control
        self.last_phase_change = defaultdict(float)
        self.last_green_time = defaultdict(float)
        self.phase_utilization = defaultdict(int)
        
        # Priority handling
        self.priority_vehicles = defaultdict(bool)
        self.ambulance_active = defaultdict(lambda: False)
        self.ambulance_start_time = defaultdict(float)
        self.left_turn_queues = defaultdict(int)
        
        # Traffic light cache to avoid repeated API calls
        self.tl_logic_cache = {}
        
        # DEFAULT dynamic parameters (used as fallback)
        default_adaptive_params = {
            'min_green': 20,
            'max_green': 50,
            'starvation_threshold': 120,
            'reward_scale': 50,
            'queue_weight': 0.5,
            'wait_weight': 0.2,
            'flow_weight': 0.8,
            'speed_weight': 0.1,
            'left_turn_priority': 1.5
        }
        
        # Load saved adaptive parameters if they exist
        success, loaded_adaptive_params = self.rl_agent.load_model()
        if success and loaded_adaptive_params:
            self.adaptive_params = loaded_adaptive_params.copy()
            print(f"🔄 Loaded adaptive parameters: {self.adaptive_params}")
        else:
            self.adaptive_params = default_adaptive_params.copy()
            print("🆕 Using default adaptive parameters")
        
        # State normalization bounds
        self.norm_bounds = {
            'queue': 20,
            'wait': 120,
            'density': 2.0,
            'speed': 20,
            'flow': 20,
            'time_since_green': 120
        }
        
        print("Enhanced Smart Traffic Controller initialized")

    def _get_traffic_light_logic(self, tl_id):
        """Get traffic light logic with caching"""
        if tl_id not in self.tl_logic_cache:
            try:
                logic = traci.trafficlight.getAllProgramLogics(tl_id)[0]
                self.tl_logic_cache[tl_id] = logic
            except Exception as e:
                print(f"Error getting logic for TL {tl_id}: {e}")
                return None
        return self.tl_logic_cache[tl_id]

    def _get_phase_count(self, tl_id):
        """Get number of phases for traffic light"""
        try:
            logic = self._get_traffic_light_logic(tl_id)
            if logic:
                return len(logic.phases)
            else:
                # Fallback: try to get from current program
                program = traci.trafficlight.getProgram(tl_id)
                return len(traci.trafficlight.getCompleteRedYellowGreenDefinition(tl_id))
        except Exception as e:
            print(f"Error getting phase count for {tl_id}: {e}")
            return 4  # Default fallback

    def _get_phase_name(self, tl_id, phase_idx):
        """Get phase name"""
        try:
            logic = self._get_traffic_light_logic(tl_id)
            if logic and phase_idx < len(logic.phases):
                return getattr(logic.phases[phase_idx], 'name', f'phase_{phase_idx}')
        except Exception as e:
            print(f"Error getting phase name for {tl_id}[{phase_idx}]: {e}")
        return f'phase_{phase_idx}'

    def run_step(self):
        """Execute one control step"""
        try:
            current_time = traci.simulation.getTime()
            lane_data = self._collect_enhanced_lane_data()
            
            if not lane_data:
                return
                
            # Update lane status and scores
            lane_status = self._update_lane_status_and_score(lane_data)
            
            # Adjust traffic lights based on current state
            self._adjust_traffic_lights(lane_data, lane_status, current_time)
            
            # RL learning and logging
            self._process_rl_learning(lane_data, current_time)
            
        except Exception as e:
            print(f"Error in run_step: {e}")

    def _collect_enhanced_lane_data(self):
        """Collect comprehensive lane data with route information"""
        lane_data = {}
        try:
            lanes = traci.lane.getIDList()
            edge_queues = defaultdict(float)
            edge_flows = defaultdict(float)
            route_queues = defaultdict(float)
            route_flows = defaultdict(float)
            
            # First pass: collect basic lane data and aggregate edge/route metrics
            for lane_id in lanes:
                try:
                    lane_length = traci.lane.getLength(lane_id)
                    if lane_length <= 0:
                        continue
                        
                    # Basic lane metrics
                    queue_length = traci.lane.getLastStepHaltingNumber(lane_id)
                    waiting_time = traci.lane.getWaitingTime(lane_id)
                    vehicle_count = traci.lane.getLastStepVehicleNumber(lane_id)
                    mean_speed = traci.lane.getLastStepMeanSpeed(lane_id)
                    
                    # Edge and route information
                    edge_id = traci.lane.getEdgeID(lane_id)
                    route_id = self._get_route_for_lane(lane_id)
                    
                    # Priority vehicle detection
                    ambulance_detected = self._detect_priority_vehicles(lane_id)
                    
                    # Left turn detection
                    is_left_turn = self._is_left_turn_lane(lane_id)
                    
                    # Store lane data
                    lane_data[lane_id] = {
                        'queue_length': queue_length,
                        'waiting_time': waiting_time,
                        'density': vehicle_count / lane_length,
                        'mean_speed': mean_speed,
                        'flow': vehicle_count,
                        'lane_id': lane_id,
                        'edge_id': edge_id,
                        'route_id': route_id,
                        'ambulance': ambulance_detected,
                        'left_turn': is_left_turn,
                        'tl_id': self.lane_to_tl.get(lane_id, '')
                    }
                    
                    # Update edge aggregates
                    edge_queues[edge_id] += queue_length
                    edge_flows[edge_id] += vehicle_count
                    
                    # Update route aggregates if available
                    if route_id:
                        route_queues[route_id] += queue_length
                        route_flows[route_id] += vehicle_count
                        
                except Exception as e:
                    print(f"Error collecting data for lane {lane_id}: {e}")
                    continue
                    
            # Second pass: add aggregated metrics
            for lane_id, data in lane_data.items():
                data['queue_route'] = route_queues.get(data.get('route_id', ''), 0)
                data['flow_route'] = route_flows.get(data.get('route_id', ''), 0)
                
        except Exception as e:
            print(f"Error in _collect_enhanced_lane_data: {e}")
            
        return lane_data

    def _get_route_for_lane(self, lane_id):
        """Get route ID for vehicles in the lane"""
        try:
            vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
            if vehicles:
                return traci.vehicle.getRouteID(vehicles[0])
        except:
            pass
        return ""

    def _detect_priority_vehicles(self, lane_id):
        """Detect priority vehicles (ambulances, emergency) in lane"""
        try:
            vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
            for vid in vehicles:
                if traci.vehicle.getVehicleClass(vid) in ['emergency', 'authority']:
                    return True
        except:
            pass
        return False

    def _is_left_turn_lane(self, lane_id):
        """Check if lane is for left turns"""
        try:
            connections = traci.lane.getLinks(lane_id)
            for conn in connections:
                if len(conn) > 3 and conn[3] == 'l':  # SUMO connection direction
                    return True
                    
            # Check naming convention as fallback
            return 'left' in lane_id.lower() or '_l_' in lane_id.lower()
        except:
            return False

    def _update_lane_status_and_score(self, lane_data):
        """Enhanced lane status update with dynamic thresholds"""
        status = {}
        try:
            for lane_id, data in lane_data.items():
                # Get normalized metrics
                queue_norm = data['queue_length'] / self.norm_bounds['queue']
                wait_norm = data['waiting_time'] / self.norm_bounds['wait']
                speed_norm = data['mean_speed'] / self.norm_bounds['speed']
                flow_norm = data['flow'] / self.norm_bounds['flow']
                
                # Calculate composite score
                composite_score = (
                    self.adaptive_params['queue_weight'] * queue_norm +
                    self.adaptive_params['wait_weight'] * wait_norm +
                    (1 - min(speed_norm, 1.0)) +  # Inverse of speed
                    (1 - min(flow_norm, 1.0))     # Inverse of flow
                )
                
                # Apply left turn priority boost
                if data['left_turn']:
                    composite_score *= self.adaptive_params['left_turn_priority']
                
                # Determine status based on dynamic thresholds
                if composite_score > 0.7:  # BAD threshold
                    status[lane_id] = "BAD"
                    delta = -min(5, int(composite_score * 5))
                elif composite_score < 0.4:  # GOOD threshold
                    status[lane_id] = "GOOD"
                    delta = min(5, int((1 - composite_score) * 5))
                else:
                    status[lane_id] = "NORMAL"
                    delta = 0
                
                # Update lane score with momentum
                if self.lane_states[lane_id] == status[lane_id]:
                    self.consecutive_states[lane_id] += 1
                    delta *= min(2, 1 + self.consecutive_states[lane_id] / 10)
                else:
                    self.lane_states[lane_id] = status[lane_id]
                    self.consecutive_states[lane_id] = 1
                
                self.lane_scores[lane_id] += delta
                
                # Ensure score stays within reasonable bounds
                self.lane_scores[lane_id] = max(-100, min(100, self.lane_scores[lane_id]))
                
        except Exception as e:
            print(f"Error in _update_lane_status_and_score: {e}")
            
        return status

    def _adjust_traffic_lights(self, lane_data, lane_status, current_time):
        """Enhanced traffic light adjustment with priority handling"""
        try:
            tl_ids = traci.trafficlight.getIDList()
            
            for tl_id in tl_ids:
                try:
                    controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
                    
                    # Update lane to traffic light mapping
                    for lane in controlled_lanes:
                        self.lane_to_tl[lane] = tl_id
                    
                    # Check for priority conditions
                    priority_handled = self._handle_priority_conditions(tl_id, controlled_lanes, lane_data, current_time)
                    
                    # Only proceed with normal control if no priority was handled
                    if not priority_handled:
                        self._perform_normal_control(tl_id, controlled_lanes, lane_data, current_time)
                        
                except Exception as e:
                    print(f"Error adjusting traffic light {tl_id}: {e}")
                    
        except Exception as e:
            print(f"Error in _adjust_traffic_lights: {e}")

    def _handle_priority_conditions(self, tl_id, controlled_lanes, lane_data, current_time):
        """Handle priority conditions (ambulances, left turns)"""
        # Check for ambulances first
        ambulance_lanes = [lane for lane in controlled_lanes 
                         if lane in lane_data and lane_data[lane]['ambulance']]
        
        if ambulance_lanes:
            self._handle_ambulance_priority(tl_id, ambulance_lanes, current_time)
            return True
            
        # Check for left turns with significant queues
        left_turn_lanes = [lane for lane in controlled_lanes 
                         if lane in lane_data and lane_data[lane]['left_turn'] 
                         and lane_data[lane]['queue_length'] > 3]
        
        if left_turn_lanes:
            self._handle_protected_left_turn(tl_id, left_turn_lanes, lane_data, current_time)
            return True
            
        return False

    def _handle_ambulance_priority(self, tl_id, ambulance_lanes, current_time):
        """Handle ambulance priority with timeout"""
        try:
            if not self.ambulance_active[tl_id]:
                # New ambulance detected - activate priority
                ambulance_lane = ambulance_lanes[0]
                phase_index = self._find_phase_for_lane(tl_id, ambulance_lane)
                
                if phase_index is not None:
                    # Set green for ambulance lane with extended duration
                    traci.trafficlight.setPhase(tl_id, phase_index)
                    traci.trafficlight.setPhaseDuration(tl_id, 30)
                    
                    self.ambulance_active[tl_id] = True
                    self.ambulance_start_time[tl_id] = current_time
                    print(f"🚑 AMBULANCE PRIORITY: Green for {ambulance_lane} at {tl_id}")
            else:
                # Check if priority should be released
                if current_time - self.ambulance_start_time[tl_id] > 30:  # 30s timeout
                    self.ambulance_active[tl_id] = False
                    print(f"🚑 AMBULANCE CLEARED: Released priority at {tl_id}")
                    
        except Exception as e:
            print(f"Error in _handle_ambulance_priority: {e}")

    def _handle_protected_left_turn(self, tl_id, left_turn_lanes, lane_data, current_time):
        """Handle protected left turn phase activation"""
        try:
            # Find the left turn lane with highest priority
            left_turn_lanes.sort(key=lambda x: lane_data[x]['queue_length'], reverse=True)
            target_lane = left_turn_lanes[0]
            
            current_phase = traci.trafficlight.getPhase(tl_id)
            phase_name = self._get_phase_name(tl_id, current_phase)
            
            # Check if we're not already in a left turn phase
            if 'left' not in phase_name.lower() and 'protected' not in phase_name.lower():
                # Find the left turn phase
                left_turn_phase = self._find_left_turn_phase(tl_id)
                
                if left_turn_phase is not None:
                    # Calculate duration based on queue length
                    queue_length = lane_data[target_lane]['queue_length']
                    duration = min(max(5 + queue_length * 0.5, self.adaptive_params['min_green']), 
                                 self.adaptive_params['max_green'])
                    
                    # Set the left turn phase
                    traci.trafficlight.setPhase(tl_id, left_turn_phase)
                    traci.trafficlight.setPhaseDuration(tl_id, duration)
                    
                    print(f"↩️ PROTECTED LEFT TURN: Activated for {target_lane} at {tl_id} " +
                         f"(queue={queue_length}, duration={duration}s)")
                    
                    # Update last green time
                    self.last_green_time[target_lane] = current_time
                    
        except Exception as e:
            print(f"Error in _handle_protected_left_turn: {e}")

    def _find_left_turn_phase(self, tl_id):
        """Find the left turn phase index for a traffic light"""
        try:
            logic = self._get_traffic_light_logic(tl_id)
            if logic:
                for idx, phase in enumerate(logic.phases):
                    phase_name = getattr(phase, 'name', f'phase_{idx}')
                    if 'left' in phase_name.lower() or 'protected' in phase_name.lower():
                        return idx
        except Exception as e:
            print(f"Error finding left turn phase for {tl_id}: {e}")
        return None

    def _perform_normal_control(self, tl_id, controlled_lanes, lane_data, current_time):
        """Perform normal RL-based traffic control"""
        try:
            # Select target lane based on multiple factors
            target_lane = self._select_target_lane(tl_id, controlled_lanes, lane_data, current_time)
            
            if not target_lane:
                return
                
            # Get state and RL action
            state = self._create_state_vector(target_lane, lane_data)
            if not self.rl_agent.is_valid_state(state):
                return
                
            action = self.rl_agent.get_action(state, lane_id=target_lane)
            
            # Execute action with minimum phase duration enforcement
            if current_time - self.last_phase_change.get(tl_id, 0) >= 5:  # 5s minimum
                self._execute_control_action(tl_id, target_lane, action, lane_data, current_time)
                
        except Exception as e:
            print(f"Error in _perform_normal_control: {e}")

    def _select_target_lane(self, tl_id, controlled_lanes, lane_data, current_time):
        """Select target lane based on multiple factors"""
        candidate_lanes = []
        
        for lane in controlled_lanes:
            if lane in lane_data:
                data = lane_data[lane]
                
                # Base score from lane scoring system
                score = self.lane_scores.get(lane, 0)
                
                # Queue and waiting time factors
                queue_factor = data['queue_length'] * 2
                wait_factor = data['waiting_time'] * 0.1
                
                # Starvation prevention
                starvation_factor = 0
                last_green = self.last_green_time.get(lane, 0)
                if current_time - last_green > self.adaptive_params['starvation_threshold']:
                    starvation_factor = (current_time - last_green - 
                                       self.adaptive_params['starvation_threshold']) * 0.5
                
                # Left turn priority boost
                left_turn_boost = 5 if data['left_turn'] else 0
                
                # Total priority score
                total_score = (score + queue_factor + wait_factor + 
                             starvation_factor + left_turn_boost)
                
                candidate_lanes.append((lane, total_score))
        
        if not candidate_lanes:
            return None
            
        # Select lane with highest priority
        candidate_lanes.sort(key=lambda x: x[1], reverse=True)
        return candidate_lanes[0][0]

    def _execute_control_action(self, tl_id, target_lane, action, lane_data, current_time):
        """Execute the selected control action"""
        try:
            # Ensure traffic light exists
            if tl_id not in traci.trafficlight.getIDList():
                print(f"⚠️ Traffic light {tl_id} not found")
                return
            
            # Get current phase and find target phase for the lane
            current_phase = traci.trafficlight.getPhase(tl_id)
            target_phase = self._find_phase_for_lane(tl_id, target_lane)
            
            if target_phase is None:
                target_phase = current_phase  # Fallback to current phase
                
            if action == 0:  # Set green for target lane
                if current_phase != target_phase:
                    green_time = self._calculate_dynamic_green(lane_data[target_lane])
                    traci.trafficlight.setPhase(tl_id, target_phase)
                    traci.trafficlight.setPhaseDuration(tl_id, green_time)
                    self.last_phase_change[tl_id] = current_time
                    self.last_green_time[target_lane] = current_time
                    print(f"🟢 RL ACTION: Green for {target_lane} (duration={green_time}s)")
                    
            elif action == 1:  # Next phase
                phase_count = self._get_phase_count(tl_id)
                next_phase = (current_phase + 1) % phase_count
                traci.trafficlight.setPhase(tl_id, next_phase)
                traci.trafficlight.setPhaseDuration(tl_id, self.adaptive_params['min_green'])
                self.last_phase_change[tl_id] = current_time
                print(f"⏭️ RL ACTION: Next phase ({next_phase})")
                
            elif action == 2:  # Extend current phase
                try:
                    remaining = traci.trafficlight.getNextSwitch(tl_id) - current_time
                    extension = min(15, self.adaptive_params['max_green'] - remaining)
                    if extension > 0:
                        traci.trafficlight.setPhaseDuration(tl_id, remaining + extension)
                        print(f"⏱️ RL ACTION: Extended phase by {extension}s")
                except Exception as e:
                    print(f"Could not extend phase: {e}")
                    
            elif action == 3:  # Shorten current phase
                try:
                    remaining = traci.trafficlight.getNextSwitch(tl_id) - current_time
                    if remaining > self.adaptive_params['min_green'] + 5:
                        reduction = min(5, remaining - self.adaptive_params['min_green'])
                        traci.trafficlight.setPhaseDuration(tl_id, remaining - reduction)
                        print(f"⏳ RL ACTION: Shortened phase by {reduction}s")
                except Exception as e:
                    print(f"Could not shorten phase: {e}")
                    
            elif action == 4:  # Balanced phase switch
                balanced_phase = self._get_balanced_phase(tl_id, lane_data)
                if balanced_phase != current_phase:
                    traci.trafficlight.setPhase(tl_id, balanced_phase)
                    traci.trafficlight.setPhaseDuration(tl_id, self.adaptive_params['min_green'])
                    self.last_phase_change[tl_id] = current_time
                    print(f"⚖️ RL ACTION: Balanced phase ({balanced_phase})")
                    
            # Update phase utilization stats
            new_phase = traci.trafficlight.getPhase(tl_id)
            self.phase_utilization[(tl_id, new_phase)] += 1
            
        except Exception as e:
            print(f"Error in _execute_control_action: {e}")

    def _get_balanced_phase(self, tl_id, lane_data):
        """Get phase that balances traffic flow"""
        try:
            logic = self._get_traffic_light_logic(tl_id)
            if not logic:
                return 0
                
            best_phase = 0
            best_score = -1
            
            controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
            
            for phase_idx in range(len(logic.phases)):
                phase_score = 0
                
                for lane_idx, lane in enumerate(controlled_lanes):
                    if lane in lane_data:
                        # Only consider lanes that get green in this phase
                        state = logic.phases[phase_idx].state
                        if lane_idx < len(state) and state[lane_idx].upper() == 'G':
                            data = lane_data[lane]
                            phase_score += (data['queue_length'] * 0.5 + 
                                          data['waiting_time'] * 0.1 +
                                          (1 - min(data['mean_speed'] / self.norm_bounds['speed'], 1)) * 10)
                
                if phase_score > best_score:
                    best_score = phase_score
                    best_phase = phase_idx
                    
            return best_phase
        except Exception as e:
            print(f"Error in _get_balanced_phase: {e}")
            return 0

    def _calculate_dynamic_green(self, lane_data):
        """Calculate dynamic green time based on lane conditions"""
        base_time = self.adaptive_params['min_green']
        queue_factor = min(lane_data['queue_length'] * 0.7, 15)
        density_factor = min(lane_data['density'] * 5, 10)
        emergency_bonus = 10 if lane_data['ambulance'] else 0
        left_turn_bonus = 5 if lane_data['left_turn'] else 0
        
        total_time = (base_time + queue_factor + density_factor + 
                     emergency_bonus + left_turn_bonus)
        
        return min(max(total_time, self.adaptive_params['min_green']), 
                 self.adaptive_params['max_green'])

    def _find_phase_for_lane(self, tl_id, target_lane):
        """Find phase that gives green to target lane"""
        try:
            logic = self._get_traffic_light_logic(tl_id)
            if not logic:
                return 0
                
            controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
            
            for phase_idx, phase in enumerate(logic.phases):
                state = phase.state
                for lane_idx, lane in enumerate(controlled_lanes):
                    if lane == target_lane and lane_idx < len(state) and state[lane_idx].upper() == 'G':
                        return phase_idx
        except Exception as e:
            print(f"Error finding phase for lane {target_lane}: {e}")
        return 0

    def _process_rl_learning(self, lane_data, current_time):
        """Process RL learning for each lane"""
        for lane_id, data in lane_data.items():
            state = self._create_state_vector(lane_id, lane_data)
            
            if not self.rl_agent.is_valid_state(state):
                continue
                
            action = self.rl_agent.get_action(state, lane_id=lane_id)
            
            # Calculate reward if we have previous state
            reward = 0
            if lane_id in self.previous_states and lane_id in self.previous_actions:
                reward = self._calculate_reward(lane_id, lane_data, 
                                            self.previous_actions[lane_id], 
                                            current_time)
                
                # Prepare lane info for logging
                lane_info = {
                    'lane_id': lane_id,
                    'edge_id': data['edge_id'],
                    'route_id': data['route_id'],
                    'queue_length': data['queue_length'],
                    'waiting_time': data['waiting_time'],
                    'density': data['density'],
                    'mean_speed': data['mean_speed'],
                    'flow': data['flow'],
                    'queue_route': data['queue_route'],
                    'flow_route': data['flow_route'],
                    'ambulance': data['ambulance'],
                    'left_turn': data['left_turn'],
                    'tl_id': self.lane_to_tl.get(lane_id, ''),
                    'phase_id': traci.trafficlight.getPhase(self.lane_to_tl.get(lane_id, '')) if lane_id in self.lane_to_tl else -1,
                    'epsilon': self.rl_agent.epsilon,
                    'learning_rate': self.rl_agent.learning_rate,
                    'adaptive_params': self.adaptive_params.copy()
                }
                
                # Update Q-table
                self.rl_agent.update_q_table(
                    self.previous_states[lane_id],
                    self.previous_actions[lane_id],
                    reward,
                    state,
                    lane_info=lane_info
                )
                
                # Get current Q-value for logging
                state_key = self.rl_agent._state_to_key(state, lane_id)
                q_value = self.rl_agent.q_table[state_key][action]
                
                # Log this step
                self.data_logger.log_step(
                    self.current_episode,
                    current_time,
                    lane_info,
                    state,
                    action,
                    reward,
                    state,
                    q_value
                )
            
            # Store current state and action
            self.previous_states[lane_id] = state
            self.previous_actions[lane_id] = action

    def _create_state_vector(self, lane_id, lane_data):
        """Create comprehensive state vector"""
        try:
            data = lane_data[lane_id]
            tl_id = self.lane_to_tl.get(lane_id, "")
            
            # Normalized lane metrics
            queue_norm = min(data['queue_length'] / self.norm_bounds['queue'], 1.0)
            wait_norm = min(data['waiting_time'] / self.norm_bounds['wait'], 1.0)
            density_norm = min(data['density'] / self.norm_bounds['density'], 1.0)
            speed_norm = min(data['mean_speed'] / self.norm_bounds['speed'], 1.0)
            flow_norm = min(data['flow'] / self.norm_bounds['flow'], 1.0)
            
            # Route-level metrics
            route_queue_norm = min(data['queue_route'] / (self.norm_bounds['queue'] * 3), 1.0)
            route_flow_norm = min(data['flow_route'] / (self.norm_bounds['flow'] * 3), 1.0)
            
            # Traffic light context
            current_phase = 0
            phase_norm = 0.0
            if tl_id:
                try:
                    current_phase = traci.trafficlight.getPhase(tl_id)
                    num_phases = self._get_phase_count(tl_id)
                    phase_norm = current_phase / max(num_phases-1, 1)
                except:
                    pass
            
            # Time since last green
            last_green = self.last_green_time.get(lane_id, 0)
            time_since_green = min((traci.simulation.getTime() - last_green) / 
                                 self.norm_bounds['time_since_green'], 1.0)
            
            # Create state vector
            state = np.array([
                queue_norm,
                wait_norm,
                density_norm,
                speed_norm,
                flow_norm,
                route_queue_norm,
                route_flow_norm,
                phase_norm,
                time_since_green,
                float(data['ambulance']),
                float(data['left_turn']),
                self.lane_scores.get(lane_id, 0) / 100  # Normalized lane score
            ])
            
            # Ensure no invalid values
            state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=0.0)
            
            return state
            
        except Exception as e:
            print(f"Error creating state vector for {lane_id}: {e}")
            return np.zeros(12)

    def _calculate_reward(self, lane_id, lane_data, action_taken, current_time):
        """Calculate comprehensive reward signal"""
        try:
            data = lane_data[lane_id]
            
            # Core components
            queue_penalty = -data['queue_length'] * self.adaptive_params['queue_weight']
            wait_penalty = -data['waiting_time'] * self.adaptive_params['wait_weight']
            throughput_reward = data['flow'] * self.adaptive_params['flow_weight']
            speed_reward = data['mean_speed'] * self.adaptive_params['speed_weight']
            
            # Action effectiveness
            action_bonus = 0
            if action_taken == 0:  # Green light action
                if data['queue_length'] > 5:
                    action_bonus = min(data['queue_length'] * 0.5, 15)
                    
            # Starvation prevention
            starvation_penalty = 0
            last_green = self.last_green_time.get(lane_id, 0)
            if current_time - last_green > self.adaptive_params['starvation_threshold']:
                starvation_penalty = -min(30, (current_time - last_green - 
                                             self.adaptive_params['starvation_threshold']) * 0.3)
            
            # Priority vehicle handling
            ambulance_bonus = 20 if data['ambulance'] else 0
            
            # Left turn priority
            left_turn_bonus = 5 if data['left_turn'] else 0
            
            # Composite reward
            total_reward = (
                queue_penalty + 
                wait_penalty + 
                throughput_reward + 
                speed_reward + 
                action_bonus + 
                starvation_penalty + 
                ambulance_bonus +
                left_turn_bonus
            )
            
            # Normalize and validate
            normalized_reward = total_reward / self.adaptive_params['reward_scale']
            normalized_reward = max(-1.0, min(1.0, normalized_reward))
            
            if np.isnan(normalized_reward) or np.isinf(normalized_reward):
                normalized_reward = 0.0
                
            return normalized_reward
            
        except Exception as e:
            print(f"Error calculating reward for {lane_id}: {e}")
            return 0.0

    def end_episode(self):
        """Finalize episode and save data"""
        try:
            # Save collected data
            self.data_logger.save_episode()
            
            # Print performance summary
            perf = self.data_logger.get_performance_summary()
            print(f"\nEpisode {self.current_episode} Performance:")
            print(f"• Average Reward: {perf.get('avg_reward', 0):.4f}")
            print(f"• Action Distribution: {perf.get('action_distribution', {})}")
            
            # Update adaptive parameters based on performance
            self._update_adaptive_parameters(perf)
            
            # Save RL model with updated adaptive parameters
            self.rl_agent.save_model(adaptive_params=self.adaptive_params)
            
            # Reset episode-specific state
            self.previous_states.clear()
            self.previous_actions.clear()
            
            print(f"✅ Episode {self.current_episode} completed and data saved")
            
        except Exception as e:
            print(f"Error ending episode: {e}")

    def _update_adaptive_parameters(self, performance_stats):
        """Dynamically adjust control parameters based on performance"""
        try:
            avg_reward = performance_stats.get('avg_reward', 0)
            
            # Adjust green time parameters based on queue performance
            queue_stats = performance_stats.get('state_stats', {}).get('state_0', {})
            avg_queue = queue_stats.get('mean', 0)
            
            if avg_queue > 0.6:  # High queues
                self.adaptive_params['min_green'] = min(15, self.adaptive_params['min_green'] + 1)
                self.adaptive_params['max_green'] = min(90, self.adaptive_params['max_green'] + 5)
            elif avg_queue < 0.3:  # Low queues
                self.adaptive_params['min_green'] = max(5, self.adaptive_params['min_green'] - 1)
                self.adaptive_params['max_green'] = max(30, self.adaptive_params['max_green'] - 5)
                
            # Adjust reward weights based on action distribution
            action_dist = performance_stats.get('action_distribution', {})
            if action_dist.get(0, 0) > 0.5:  # Too many green actions
                self.adaptive_params['queue_weight'] = min(1.0, self.adaptive_params['queue_weight'] * 1.1)
                self.adaptive_params['wait_weight'] = max(0.1, self.adaptive_params['wait_weight'] * 0.9)
                
            print("🔄 Updated adaptive parameters:", self.adaptive_params)
            
        except Exception as e:
            print(f"Error updating adaptive parameters: {e}")

def start_enhanced_simulation(sumocfg_path, use_gui=True, max_steps=None, episodes=1, num_retries=1, retry_delay=1):
    """Run enhanced simulation with the smart controller"""
    controller = None
    try:
        controller = SmartTrafficController()
        
        for episode in range(episodes):
            print(f"\n{'='*50}")
            print(f"🚦 STARTING ENHANCED EPISODE {episode + 1}/{episodes}")
            print(f"{'='*50}")
            
            sumo_binary = "sumo-gui" if use_gui else "sumo"
            sumo_cmd = [
                os.path.join(os.environ['SUMO_HOME'], 'bin', sumo_binary),
                '-c', sumocfg_path,
                '--start', '--quit-on-end'
            ]
            
            traci.start(sumo_cmd)
            controller.current_episode = episode + 1
            
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
                        print(f"Episode {episode + 1}: Step {step} completed")
                        
                except Exception as e:
                    print(f"Error in simulation step {step}: {e}")
                    break
                    
            print(f"Episode {episode + 1} completed after {step} steps")
            controller.end_episode()
            traci.close()
            
            if episode < episodes - 1:
                time.sleep(2)  # Brief pause between episodes
                
        print(f"\n🎉 All {episodes} episodes completed!")
        
    except Exception as e:
        print(f"Error in enhanced simulation: {e}")
    finally:
        try:
            traci.close()
        except:
            pass
        print("Simulation resources cleaned up")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run enhanced SUMO RL traffic simulation")
    parser.add_argument('--sumo', required=True, help='Path to SUMO config file')
    parser.add_argument('--gui', action='store_true', help='Use SUMO GUI')
    parser.add_argument('--max-steps', type=int, help='Maximum simulation steps per episode')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes to run')
    parser.add_argument('--num-retries', type=int, default=1, help='Number of retries if connection fails')
    parser.add_argument('--retry-delay', type=int, default=1, help='Delay in seconds between retries')
    args = parser.parse_args()
    start_enhanced_simulation(args.sumo, args.gui, args.max_steps, args.episodes, args.num_retries, args.retry_delay)