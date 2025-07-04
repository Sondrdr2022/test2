import pickle
import json
import pandas as pd
import numpy as np
import os
import time
from collections import defaultdict

def clean_state_data(state_data):
    """Convert state/next_state from JSON string or array to readable columns"""
    if isinstance(state_data, str):
        try:
            state_array = json.loads(state_data)
        except:
            return {}
    elif isinstance(state_data, (list, np.ndarray)):
        state_array = state_data
    else:
        return {}
    
    if len(state_array) >= 6:
        return {
            'time_norm': round(float(state_array[0]), 4),
            'queue_norm': round(float(state_array[1]), 4),
            'wait_norm': round(float(state_array[2]), 4),
            'density_norm': round(float(state_array[3]), 4),
            'speed_norm': round(float(state_array[4]), 4),
            'flow_norm': round(float(state_array[5]), 4)
        }
    return {}

def export_training_data_xlsx_only(training_data, base_filename):
    """Export only as Excel file (.xlsx) for best compatibility"""
    
    clean_data = []
    print(f"Processing {len(training_data)} training entries...")
    
    for i, entry in enumerate(training_data):
        if i % 100 == 0:
            print(f"  Processing entry {i+1}/{len(training_data)}")
        
        clean_entry = {
            'episode_step': entry.get('simulation_time', i + 1),
            'action': int(entry.get('action', 0)),
            'reward': float(entry.get('reward', 0)),
            'q_value': float(entry.get('q_value', 0)),
            'done': bool(entry.get('done', False))
        }
        
        # Add requested per-lane columns if present in the entry
        for col in ['lane_id', 'edge_id', 'c', 'm', 'v', 'g', 'queue_lane', 'queue_route', 'flow_route', 'wait_lane']:
            if col in entry:
                clean_entry[col] = entry[col]
        
        # Add state data
        state_cols = clean_state_data(entry.get('state', []))
        for col, val in state_cols.items():
            clean_entry[f'state_{col}'] = float(val)
        
        # Add next_state data
        next_state_cols = clean_state_data(entry.get('next_state', []))
        for col, val in next_state_cols.items():
            clean_entry[f'next_state_{col}'] = float(val)
        
        clean_data.append(clean_entry)
    
    # Create DataFrame
    df = pd.DataFrame(clean_data)
    
    # Reorder columns
    column_order = [
        'episode_step', 'lane_id', 'edge_id', 'c', 'action', 'reward', 'q_value', 'done',
        'm', 'v', 'g', 'queue_lane', 'queue_route', 'flow_route', 'wait_lane',
        'state_time_norm', 'state_queue_norm', 'state_wait_norm', 
        'state_density_norm', 'state_speed_norm', 'state_flow_norm',
        'next_state_time_norm', 'next_state_queue_norm', 'next_state_wait_norm',
        'next_state_density_norm', 'next_state_speed_norm', 'next_state_flow_norm'
    ]
    
    available_columns = [col for col in column_order if col in df.columns]
    df = df[available_columns]
    
    # Export only to Excel
    exported_files = []
    try:
        excel_file = base_filename.replace('.csv', '.xlsx')
        df.to_excel(excel_file, index=False, float_format='%.6f')
        exported_files.append(('Excel file', excel_file))
        print(f"✅ Excel file: {excel_file}")
    except Exception as e:
        print(f"❌ Excel export failed: {e}")
    
    return df, exported_files

def analyze_qtable(pkl_file='q_table.pkl'):
    """Analyze Q-table pickle file and display/export training data as xlsx only"""
    try:
        print(f"Loading {pkl_file}...")
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        
        print("✅ Successfully loaded pickle file!")
        print("="*60)
        
        print("📁 FILE STRUCTURE:")
        print(f"Keys in file: {list(data.keys())}")
        print()
        
        if 'q_table' in data:
            q_table = data['q_table']
            print("🧠 Q-TABLE ANALYSIS:")
            print(f"Number of states learned: {len(q_table)}")
            
            total_q_values = 0
            non_zero_q_values = 0
            for state, q_values in q_table.items():
                for q_val in q_values:
                    total_q_values += 1
                    if abs(q_val) > 1e-10:
                        non_zero_q_values += 1
            
            print(f"Total Q-values: {total_q_values}")
            print(f"Non-zero Q-values: {non_zero_q_values}")
            if total_q_values > 0:
                print(f"Learning progress: {non_zero_q_values/total_q_values*100:.2f}%")
            print()
        
        if 'training_data' in data:
            training_data = data['training_data']
            print("📈 TRAINING DATA ANALYSIS:")
            print(f"Total training episodes: {len(training_data)}")
            
            if training_data:
                actions = [entry.get('action', 0) for entry in training_data]
                print(f"\n🎯 ACTION DISTRIBUTION:")
                for action in sorted(set(actions)):
                    count = actions.count(action)
                    percentage = (count / len(actions)) * 100
                    print(f"  Action {action}: {count} times ({percentage:.1f}%)")
                
                rewards = [entry.get('reward', 0) for entry in training_data]
                print("\n💰 REWARD STATISTICS:")
                print(f"  Average reward: {np.mean(rewards):.6f}")
                print(f"  Max reward: {np.max(rewards):.6f}")
                print(f"  Min reward: {np.min(rewards):.6f}")
                print()
        
        if 'params' in data:
            params = data['params']
            print("⚙️ LEARNING PARAMETERS:")
            for key, value in params.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.6f}")
                else:
                    print(f"  {key}: {value}")
            print()
        
        # Export only as Excel
        if 'training_data' in data and training_data:
            export = input("💾 Export training data as Excel (.xlsx)? (y/n): ").lower().strip()
            if export == 'y':
                print(f"\n🔄 Exporting as Excel...")
                base_filename = pkl_file.replace('.pkl', '_training.csv')
                df, exported_files = export_training_data_xlsx_only(training_data, base_filename)
                
                print(f"\n📊 Export Summary:")
                print(f"Data shape: {len(df)} rows × {len(df.columns)} columns")
                print(f"Exported Excel file: {exported_files[0][1] if exported_files else 'None'}")
                print(f"\n💡 RECOMMENDATION: Open the .xlsx file in Excel or compatible software.")
        
    except FileNotFoundError:
        print(f"❌ Error: {pkl_file} not found!")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_qtable('enhanced_q_table.pkl')