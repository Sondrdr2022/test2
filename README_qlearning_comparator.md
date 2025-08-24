# Q-Learning Training Data Comparator

## Overview

This tool analyzes and compares two Excel files containing Q-learning training data to identify differences between "before" and "after" training datasets. It provides comprehensive analysis to demonstrate whether the RL agent is actually learning and improving between training sessions.

## Features

### Data Loading & Validation
- Safe loading of Excel files with comprehensive error handling
- Validation of data structure and compatibility
- Support for different file formats and structures
- Handling of missing columns and data type inconsistencies

### Comprehensive Analysis
1. **Data Structure Comparison**
   - Total number of training entries
   - Column structure validation
   - New vs. removed records

2. **Reward Analysis**
   - Mean, min, max reward statistics
   - Statistical significance testing (t-test)
   - Reward distribution changes
   - Performance improvement indicators

3. **Q-Value Analysis**
   - Q-value progression and improvements
   - Convergence indicators (variance reduction)
   - Learning stability metrics

4. **Action Distribution Analysis**
   - Policy changes and refinements
   - Action frequency distributions
   - Policy diversity metrics (entropy)

5. **State Space Analysis**
   - State feature variations
   - Coverage analysis
   - Next-state progression

6. **Detailed Difference Detection**
   - Record-level comparison
   - Column-specific change detection
   - Identical vs. modified data identification

### Outputs

1. **Comprehensive Report** (`qlearning_comparison_report.txt`)
   - Executive summary
   - Detailed analysis results
   - Learning indicators assessment
   - Recommendations for improvement

2. **Visualizations**
   - Reward and Q-value distribution comparisons
   - Action distribution charts
   - Learning curves and progression plots
   - State feature correlation heatmaps
   - Comprehensive comparison dashboard

3. **Learning Assessment**
   - Automatic detection of learning improvements
   - Evidence-based conclusions
   - Actionable recommendations

## Usage

### Basic Usage
```bash
python qlearning_data_comparator.py --before q_table_before_training.xlsx --after q_table_training.xlsx
```

### Advanced Usage
```bash
python qlearning_data_comparator.py \
    --before path/to/before_training.xlsx \
    --after path/to/after_training.xlsx \
    --output custom_results_directory
```

### Command Line Arguments

- `--before` (required): Path to Excel file with data before training
- `--after` (required): Path to Excel file with data after training  
- `--output` (optional): Output directory for results (default: `comparison_results`)

## Expected Data Format

The Excel files should contain Q-learning training data with the following columns:

### Required Columns
- `episode_step`: Training step/episode number
- `action`: Action taken by the agent
- `reward`: Reward received
- `q_value`: Q-value for the state-action pair

### Optional Columns
- `done`: Episode termination flag
- `state_*`: State feature columns (e.g., `state_time_norm`, `state_queue_norm`)
- `next_state_*`: Next state feature columns

## Example Output

### Console Output
```
============================================================
ANALYZING REWARDS AND PERFORMANCE
============================================================
Reward Statistics Comparison:
Metric       Before       After        Change       % Change    
------------------------------------------------------------
mean         12.1005      12.6456      0.5451       4.50%
std          5.8075       5.7651       -0.0424      -0.73%
min          -1.4676      -1.0608      0.4069       -27.72%
max          20.8478      25.3244      4.4766       21.47%

Statistical Significance Test (t-test):
  - t-statistic: -1.6141
  - p-value: 0.1068
  - Significant improvement: No (α = 0.05)
```

### Learning Assessment
```
Learning Indicators:
  ✓ Mean reward increased
  ✓ Q-values show convergence  
  ✓ New training data was added

Overall Assessment: 3/3 positive indicators
CONCLUSION: Strong evidence of learning and improvement
```

## Installation Requirements

```bash
pip install pandas openpyxl matplotlib seaborn scipy numpy
```

## File Structure

```
qlearning_data_comparator.py    # Main analysis script
README_qlearning_comparator.md  # This documentation
comparison_results/             # Default output directory
├── qlearning_comparison_report.txt           # Detailed text report
├── qlearning_comparison_visualization.png    # Main visualization
└── learning_curves.png                       # Learning progression plots
```

## Interpretation Guide

### Positive Learning Indicators
- **Mean reward increased**: Agent is performing better
- **Q-values show convergence**: Learning is stabilizing (variance reduction)
- **New training data added**: More experience collected
- **Policy refinement**: Action distribution becomes more focused on better actions

### Warning Signs
- **No new training data**: May indicate stalled training
- **Decreasing rewards**: Performance regression
- **Increasing Q-value variance**: Unstable learning
- **No policy changes**: Agent may not be exploring or learning

### Recommendations
The tool provides specific recommendations based on the analysis:
- Hyperparameter adjustments
- Training strategy modifications
- Monitoring suggestions
- Next steps for improvement

## Integration with Traffic Light RL System

This tool is designed to work with the Q-learning traffic light control system in this repository. It can analyze training data generated by:

- `QLearningAgent` class in `Lane.py`
- `DataLogger` for CSV-based training logs
- Excel exports of Q-table and training data

## Troubleshooting

### Common Issues

1. **File not found**: Ensure file paths are correct and files exist
2. **Column mismatch**: Verify both files have compatible column structure
3. **Memory issues**: For very large datasets, consider sampling
4. **No differences detected**: Files may be identical - check if training actually occurred

### Error Messages

- `"Before training file not found"`: Check the file path
- `"Column structure mismatch"`: Ensure both files have same columns
- `"No new training data detected"`: Both files have same number of records

## Contributing

To extend the analysis capabilities:

1. Add new analysis methods to the `QTableComparator` class
2. Extend visualization functions for new metrics
3. Update the reporting system with new findings
4. Add support for additional file formats

## License

This tool is part of the traffic light RL system project and follows the same license terms.