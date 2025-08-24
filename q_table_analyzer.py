#!/usr/bin/env python3
"""
Q-Table Analyzer Script

This script analyzes Q-learning training data stored in pickle files.
It provides comprehensive analysis of Q-tables, training data, and parameters.

Usage:
    python q_table_analyzer.py [path_to_pickle_file]
    
If no path is provided, it will look for 'q_table.pkl' in common locations.
"""

import pickle
import os
import sys
import json
from typing import Dict, List, Any, Optional
import argparse
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
    NUMPY_AVAILABLE = True
except ImportError:
    try:
        import numpy as np
        PANDAS_AVAILABLE = False
        NUMPY_AVAILABLE = True
        print("Warning: pandas not available. CSV export will be limited.")
    except ImportError:
        PANDAS_AVAILABLE = False
        NUMPY_AVAILABLE = False
        print("Warning: pandas and numpy not available. Advanced statistics and CSV export will be limited.")


class QTableAnalyzer:
    """Analyzer for Q-learning training data and Q-tables."""
    
    def __init__(self, pickle_file_path: str):
        """Initialize analyzer with pickle file path."""
        self.pickle_file_path = pickle_file_path
        self.data = None
        self.q_table = None
        self.training_data = None
        self.params = None
        
    def load_data(self) -> bool:
        """
        Safely load pickle file data.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not os.path.exists(self.pickle_file_path):
                print(f"Error: File '{self.pickle_file_path}' not found.")
                return False
                
            with open(self.pickle_file_path, 'rb') as f:
                self.data = pickle.load(f)
                
            # Validate data structure
            if not isinstance(self.data, dict):
                print("Error: Pickle file does not contain a dictionary.")
                return False
                
            # Extract components
            self.q_table = self.data.get('q_table', {})
            self.training_data = self.data.get('training_data', [])
            self.params = self.data.get('params', {})
            
            print(f"✓ Successfully loaded data from '{self.pickle_file_path}'")
            return True
            
        except pickle.UnpicklingError:
            print("Error: Invalid pickle file format.")
            return False
        except Exception as e:
            print(f"Error loading file: {e}")
            return False
    
    def display_file_info(self):
        """Display basic file information."""
        print("\n" + "="*60)
        print("FILE INFORMATION")
        print("="*60)
        
        file_path = Path(self.pickle_file_path)
        file_size = file_path.stat().st_size
        
        print(f"File Path: {self.pickle_file_path}")
        print(f"File Size: {file_size:,} bytes ({file_size/1024:.2f} KB)")
        print(f"Data Structure Keys: {list(self.data.keys())}")
        
    def display_q_table_statistics(self):
        """Display comprehensive Q-table statistics."""
        print("\n" + "="*60)
        print("Q-TABLE STATISTICS")
        print("="*60)
        
        if not self.q_table:
            print("No Q-table data found.")
            return
            
        num_states = len(self.q_table)
        print(f"Number of States: {num_states:,}")
        
        if num_states > 0:
            # Get action size from first state
            first_state_key = next(iter(self.q_table))
            action_values = self.q_table[first_state_key]
            
            if NUMPY_AVAILABLE and isinstance(action_values, np.ndarray):
                action_size = len(action_values)
                print(f"Number of Actions per State: {action_size}")
                
                # Collect all Q-values for statistics
                all_q_values = []
                for state_actions in self.q_table.values():
                    all_q_values.extend(state_actions)
                
                q_array = np.array(all_q_values)
                print(f"Total Q-values: {len(all_q_values):,}")
                print(f"Q-value Statistics:")
                print(f"  Min: {q_array.min():.6f}")
                print(f"  Max: {q_array.max():.6f}")
                print(f"  Mean: {q_array.mean():.6f}")
                print(f"  Std: {q_array.std():.6f}")
                
                # Show non-zero Q-values
                non_zero_q = q_array[q_array != 0]
                print(f"Non-zero Q-values: {len(non_zero_q):,} ({len(non_zero_q)/len(all_q_values)*100:.1f}%)")
                
            else:
                print(f"Actions per state: {len(action_values) if hasattr(action_values, '__len__') else 'Unknown'}")
                
        # Show sample states
        print(f"\nSample States (first 5):")
        for i, (state_key, q_values) in enumerate(list(self.q_table.items())[:5]):
            print(f"  State {i+1}: {state_key}")
            if NUMPY_AVAILABLE and isinstance(q_values, np.ndarray):
                print(f"    Q-values: {q_values}")
            else:
                print(f"    Q-values: {q_values}")
            
    def display_training_data_statistics(self):
        """Display training data statistics."""
        print("\n" + "="*60)
        print("TRAINING DATA STATISTICS")
        print("="*60)
        
        if not self.training_data:
            print("No training data found.")
            return
            
        num_episodes = len(self.training_data)
        print(f"Total Training Episodes: {num_episodes:,}")
        
        if num_episodes > 0:
            # Analyze first episode structure
            sample_episode = self.training_data[0]
            print(f"Episode Structure: {list(sample_episode.keys())}")
            
            if NUMPY_AVAILABLE:
                try:
                    # Analyze rewards
                    rewards = [episode.get('reward', 0) for episode in self.training_data]
                    rewards_array = np.array(rewards)
                    
                    print(f"\nReward Statistics:")
                    print(f"  Min Reward: {rewards_array.min():.6f}")
                    print(f"  Max Reward: {rewards_array.max():.6f}")
                    print(f"  Mean Reward: {rewards_array.mean():.6f}")
                    print(f"  Total Reward: {rewards_array.sum():.6f}")
                    
                    # Analyze actions
                    actions = [episode.get('action', 0) for episode in self.training_data]
                    unique_actions, action_counts = np.unique(actions, return_counts=True)
                    
                    print(f"\nAction Distribution:")
                    for action, count in zip(unique_actions, action_counts):
                        print(f"  Action {action}: {count:,} times ({count/num_episodes*100:.1f}%)")
                        
                    # Analyze Q-values
                    q_values = [episode.get('q_value', 0) for episode in self.training_data]
                    q_values_array = np.array(q_values)
                    
                    print(f"\nTraining Q-value Statistics:")
                    print(f"  Min: {q_values_array.min():.6f}")
                    print(f"  Max: {q_values_array.max():.6f}")
                    print(f"  Mean: {q_values_array.mean():.6f}")
                    print(f"  Std: {q_values_array.std():.6f}")
                    
                except Exception as e:
                    print(f"Error analyzing training data: {e}")
            else:
                # Basic statistics without numpy
                rewards = [episode.get('reward', 0) for episode in self.training_data]
                print(f"\nReward Statistics:")
                print(f"  Min Reward: {min(rewards):.6f}")
                print(f"  Max Reward: {max(rewards):.6f}")
                print(f"  Mean Reward: {sum(rewards)/len(rewards):.6f}")
                print(f"  Total Reward: {sum(rewards):.6f}")
                
                # Count actions manually
                action_counts = {}
                for episode in self.training_data:
                    action = episode.get('action', 0)
                    action_counts[action] = action_counts.get(action, 0) + 1
                
                print(f"\nAction Distribution:")
                for action, count in sorted(action_counts.items()):
                    print(f"  Action {action}: {count:,} times ({count/num_episodes*100:.1f}%)")
                    
    def display_sample_training_data(self, num_samples: int = 5):
        """Display sample training data entries."""
        print("\n" + "="*60)
        print(f"SAMPLE TRAINING DATA (First {num_samples} entries)")
        print("="*60)
        
        if not self.training_data:
            print("No training data available.")
            return
            
        for i, episode in enumerate(self.training_data[:num_samples]):
            print(f"\nEpisode {i+1}:")
            for key, value in episode.items():
                if key in ['state', 'next_state']:
                    # Format state arrays nicely
                    if isinstance(value, (list, tuple)) or (NUMPY_AVAILABLE and isinstance(value, np.ndarray)):
                        if len(value) <= 10:  # Show full array if small
                            print(f"  {key}: {value}")
                        else:  # Show truncated for large arrays
                            print(f"  {key}: [{value[0]:.3f}, {value[1]:.3f}, ..., {value[-1]:.3f}] (length: {len(value)})")
                    else:
                        print(f"  {key}: {value}")
                else:
                    print(f"  {key}: {value}")
                    
    def display_parameters(self):
        """Display learning parameters."""
        print("\n" + "="*60)
        print("LEARNING PARAMETERS")
        print("="*60)
        
        if not self.params:
            print("No parameter information found.")
            return
            
        for key, value in self.params.items():
            print(f"{key}: {value}")
            
    def export_to_csv(self, output_dir: str = ".") -> bool:
        """
        Export training data to CSV files.
        
        Args:
            output_dir: Directory to save CSV files
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not PANDAS_AVAILABLE:
            print("Error: pandas is required for CSV export. Please install pandas.")
            return False
            
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            # Export training data
            if self.training_data:
                training_df = pd.DataFrame(self.training_data)
                
                # Convert state and next_state arrays to string representation
                if 'state' in training_df.columns:
                    training_df['state'] = training_df['state'].apply(lambda x: json.dumps(x.tolist() if hasattr(x, 'tolist') else x))
                if 'next_state' in training_df.columns:
                    training_df['next_state'] = training_df['next_state'].apply(lambda x: json.dumps(x.tolist() if hasattr(x, 'tolist') else x))
                    
                training_file = output_path / "training_data.csv"
                training_df.to_csv(training_file, index=False)
                print(f"✓ Training data exported to: {training_file}")
                
            # Export Q-table
            if self.q_table:
                q_table_data = []
                for state_key, q_values in self.q_table.items():
                    row = {'state': json.dumps(state_key)}
                    if NUMPY_AVAILABLE and isinstance(q_values, np.ndarray):
                        for i, q_val in enumerate(q_values):
                            row[f'action_{i}_q_value'] = q_val
                    else:
                        row['q_values'] = json.dumps(q_values.tolist() if hasattr(q_values, 'tolist') else q_values)
                    q_table_data.append(row)
                    
                q_table_df = pd.DataFrame(q_table_data)
                q_table_file = output_path / "q_table.csv"
                q_table_df.to_csv(q_table_file, index=False)
                print(f"✓ Q-table exported to: {q_table_file}")
                
            # Export parameters
            if self.params:
                params_df = pd.DataFrame([self.params])
                params_file = output_path / "parameters.csv"
                params_df.to_csv(params_file, index=False)
                print(f"✓ Parameters exported to: {params_file}")
                
            return True
            
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
            return False
            
    def generate_report(self, num_samples: int = 5):
        """Generate a comprehensive analysis report."""
        print("\n" + "="*80)
        print("Q-TABLE ANALYSIS REPORT")
        print("="*80)
        
        self.display_file_info()
        self.display_parameters()
        self.display_q_table_statistics()
        self.display_training_data_statistics()
        self.display_sample_training_data(num_samples)
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)


def find_q_table_file() -> Optional[str]:
    """Find q_table.pkl file in common locations."""
    common_paths = [
        "q_table.pkl",
        "New folder (2)/q_table.pkl",
        "./New folder (2)/q_table.pkl",
        "../q_table.pkl"
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            return path
            
    return None


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Analyze Q-learning training data from pickle files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python q_table_analyzer.py                          # Auto-find q_table.pkl
  python q_table_analyzer.py data/q_table.pkl         # Specify file path
  python q_table_analyzer.py --export ./output        # Export to directory
        """
    )
    
    parser.add_argument(
        'pickle_file', 
        nargs='?', 
        help='Path to the pickle file containing Q-table data'
    )
    
    parser.add_argument(
        '--export', 
        metavar='DIR',
        help='Export data to CSV files in specified directory'
    )
    
    parser.add_argument(
        '--samples', 
        type=int, 
        default=5,
        help='Number of sample training entries to display (default: 5)'
    )
    
    args = parser.parse_args()
    
    # Determine pickle file path
    if args.pickle_file:
        pickle_file_path = args.pickle_file
    else:
        pickle_file_path = find_q_table_file()
        if not pickle_file_path:
            print("Error: Could not find q_table.pkl file.")
            print("Please specify the path to your pickle file:")
            print("  python q_table_analyzer.py /path/to/q_table.pkl")
            sys.exit(1)
    
    # Create analyzer and load data
    analyzer = QTableAnalyzer(pickle_file_path)
    
    if not analyzer.load_data():
        sys.exit(1)
    
    # Generate analysis report
    analyzer.generate_report(args.samples)
    
    # Export to CSV if requested
    if args.export:
        print(f"\nExporting data to CSV files...")
        if analyzer.export_to_csv(args.export):
            print("Export completed successfully!")
        else:
            print("Export failed!")
    
    # Offer interactive CSV export
    elif PANDAS_AVAILABLE:
        print("\n" + "-"*60)
        response = input("Would you like to export data to CSV files? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            output_dir = input("Enter output directory (press Enter for current directory): ").strip()
            if not output_dir:
                output_dir = "."
            
            if analyzer.export_to_csv(output_dir):
                print("Export completed successfully!")
            else:
                print("Export failed!")


if __name__ == "__main__":
    main()