# Q-Table Analyzer

A Python script for analyzing Q-learning training data stored in pickle files. This tool provides comprehensive analysis of Q-tables, training episodes, and learning parameters.

## Features

- **Safe Loading**: Robust error handling for corrupt or invalid pickle files
- **Comprehensive Statistics**: Detailed analysis of Q-table and training data
- **Sample Data Display**: View sample training episodes with customizable count
- **Parameter Information**: Display learning parameters like epsilon, learning rate, etc.
- **CSV Export**: Export all data to CSV files for further analysis
- **Auto-Detection**: Automatically finds q_table.pkl in common locations

## Requirements

- Python 3.6+
- pandas (for CSV export and advanced statistics)
- numpy (for numerical analysis)

Install dependencies:
```bash
pip install pandas numpy
```

## Usage

### Basic Usage

```bash
# Auto-find q_table.pkl in current directory or common locations
python q_table_analyzer.py

# Specify pickle file path
python q_table_analyzer.py path/to/q_table.pkl

# Show only 2 sample training entries instead of default 5
python q_table_analyzer.py --samples 2

# Export data to CSV files
python q_table_analyzer.py --export ./output_directory
```

### Command Line Options

- `pickle_file`: Path to the pickle file (optional, auto-detects if not provided)
- `--export DIR`: Export data to CSV files in specified directory
- `--samples N`: Number of sample training entries to display (default: 5)
- `--help`: Show help message

## Expected Data Structure

The script expects a pickle file containing a dictionary with these keys:

- `'q_table'`: Dictionary of state-action Q-values
- `'training_data'`: List of training episodes with keys:
  - `state`: Current state vector
  - `action`: Action taken
  - `reward`: Reward received
  - `next_state`: Next state vector
  - `q_value`: Updated Q-value
- `'params'`: Learning parameters including:
  - `state_size`: Number of state features
  - `action_size`: Number of possible actions
  - `learning_rate`: Learning rate alpha
  - `discount_factor`: Discount factor gamma
  - `epsilon`: Exploration rate

## Output

The script provides:

1. **File Information**: File size and structure
2. **Learning Parameters**: All training parameters
3. **Q-Table Statistics**: Number of states, actions, Q-value distribution
4. **Training Data Statistics**: Episode count, reward statistics, action distribution
5. **Sample Training Data**: Detailed view of training episodes
6. **CSV Export** (optional): Separate files for training data, Q-table, and parameters

## Examples

### Basic Analysis
```bash
python q_table_analyzer.py
```

### Export to CSV
```bash
python q_table_analyzer.py --export ./analysis_results
```

### Custom Sample Count
```bash
python q_table_analyzer.py --samples 10
```

### Full Analysis with Export
```bash
python q_table_analyzer.py data/my_q_table.pkl --export ./results --samples 3
```

## Error Handling

The script handles various error conditions:
- Missing or inaccessible files
- Corrupt pickle files
- Invalid data structures
- Missing data components
- Export failures

All errors are reported with clear messages and appropriate exit codes.