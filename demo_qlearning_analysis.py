#!/usr/bin/env python3
"""
Demonstration script for Q-Learning Training Data Comparator

This script shows how to use the qlearning_data_comparator.py tool
to analyze Q-learning training data and detect learning improvements.
"""

import os
import sys
from qlearning_data_comparator import QTableComparator

def run_analysis_demo():
    """Run a demonstration of the Q-learning data analysis."""
    
    print("="*80)
    print("Q-LEARNING TRAINING DATA ANALYSIS DEMONSTRATION")
    print("="*80)
    
    # Define file paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    before_file = os.path.join(current_dir, "New folder (2)", "q_table_before_training.xlsx")
    after_file = os.path.join(current_dir, "New folder (2)", "q_table_training.xlsx")
    demo_after_file = os.path.join(current_dir, "New folder (2)", "q_table_after_training_demo.xlsx")
    
    print(f"Demonstration files:")
    print(f"  Original before: {before_file}")
    print(f"  Original after:  {after_file}")
    print(f"  Demo improved:   {demo_after_file}")
    
    # Check if files exist
    files_exist = all(os.path.exists(f) for f in [before_file, after_file])
    demo_exists = os.path.exists(demo_after_file)
    
    if not files_exist:
        print("\n❌ Original training data files not found!")
        print("Please ensure the Excel files are available in 'New folder (2)/'")
        return False
    
    print(f"\n✓ Original files found")
    print(f"✓ Demo file available: {demo_exists}")
    
    # Analysis 1: Compare identical files (original case)
    print("\n" + "="*60)
    print("ANALYSIS 1: COMPARING IDENTICAL DATASETS")
    print("="*60)
    print("This demonstrates what happens when no learning occurred...")
    
    comparator1 = QTableComparator(before_file, after_file, "results_identical")
    success1 = comparator1.run_complete_analysis()
    
    if success1:
        print("\n✓ Analysis 1 completed - shows identical datasets")
    else:
        print("\n❌ Analysis 1 failed")
        return False
    
    # Analysis 2: Compare with improved data (if demo file exists)
    if demo_exists:
        print("\n" + "="*60)
        print("ANALYSIS 2: COMPARING WITH LEARNING IMPROVEMENTS")
        print("="*60)
        print("This demonstrates detection of actual learning progress...")
        
        comparator2 = QTableComparator(before_file, demo_after_file, "results_improved")
        success2 = comparator2.run_complete_analysis()
        
        if success2:
            print("\n✓ Analysis 2 completed - shows learning improvements")
        else:
            print("\n❌ Analysis 2 failed")
            return False
    else:
        print("\n⚠️  Demo improved file not found - skipping Analysis 2")
        print("Run the data generation part of qlearning_data_comparator.py to create it")
    
    # Summary
    print("\n" + "="*80)
    print("DEMONSTRATION SUMMARY")
    print("="*80)
    
    print("\n📁 Generated Output Files:")
    
    output_dirs = ["results_identical"]
    if demo_exists:
        output_dirs.append("results_improved")
    
    for output_dir in output_dirs:
        if os.path.exists(output_dir):
            print(f"\n  {output_dir}/")
            for file in os.listdir(output_dir):
                print(f"    ├── {file}")
    
    print("\n📊 Key Findings:")
    print("  • Analysis 1: Demonstrates detection of identical datasets")
    print("  • Analysis 1: Shows recommendations when no learning is detected")
    
    if demo_exists:
        print("  • Analysis 2: Demonstrates detection of learning improvements")
        print("  • Analysis 2: Shows positive learning indicators")
        print("  • Analysis 2: Quantifies performance improvements")
    
    print("\n🎯 Use Cases:")
    print("  • Monitor Q-learning training progress")
    print("  • Validate that learning is actually occurring")
    print("  • Quantify performance improvements")
    print("  • Generate reports for research or development")
    print("  • Debug training issues when no improvement is seen")
    
    print("\n📖 Next Steps:")
    print("  1. Review the generated reports and visualizations")
    print("  2. Use the tool with your own Q-learning training data")
    print("  3. Integrate into your training pipeline for automatic monitoring")
    print("  4. Customize the analysis for your specific use case")
    
    return True

def show_usage_examples():
    """Show usage examples for the comparator tool."""
    
    print("\n" + "="*60)
    print("USAGE EXAMPLES")
    print("="*60)
    
    examples = [
        {
            "title": "Basic Usage",
            "command": "python qlearning_data_comparator.py --before before.xlsx --after after.xlsx",
            "description": "Compare two Excel files with default output directory"
        },
        {
            "title": "Custom Output Directory", 
            "command": "python qlearning_data_comparator.py --before data/before.xlsx --after data/after.xlsx --output my_results/",
            "description": "Specify custom output directory for results"
        },
        {
            "title": "Programmatic Usage",
            "command": """from qlearning_data_comparator import QTableComparator
comparator = QTableComparator('before.xlsx', 'after.xlsx', 'output/')
success = comparator.run_complete_analysis()""",
            "description": "Use as a Python module in your own scripts"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['title']}")
        print(f"   {example['description']}")
        print(f"   Command: {example['command']}")

if __name__ == "__main__":
    print("Q-Learning Training Data Analysis Demonstration")
    print("This script demonstrates the capabilities of the Q-learning data comparator tool.\n")
    
    # Run the main demonstration
    success = run_analysis_demo()
    
    if success:
        # Show usage examples
        show_usage_examples()
        
        print(f"\n🎉 Demonstration completed successfully!")
        print(f"Check the generated output directories for detailed results.")
    else:
        print(f"\n❌ Demonstration failed!")
        sys.exit(1)