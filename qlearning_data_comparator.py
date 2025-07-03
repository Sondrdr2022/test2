#!/usr/bin/env python3
"""
Q-Learning Training Data Comparator

This script analyzes and compares two Excel files containing Q-learning training data
to identify differences between "before" and "after" training datasets and quantify
learning improvements.

Author: AI Assistant
Date: 2024
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
from typing import Dict, Tuple, List, Optional
import argparse
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class QTableComparator:
    """
    Comprehensive Q-learning training data comparator and analyzer.
    
    This class loads, compares, and analyzes Q-learning training data from Excel files
    to identify learning improvements and generate detailed comparison reports.
    """
    
    def __init__(self, before_file: str, after_file: str, output_dir: str = "comparison_results"):
        """
        Initialize the comparator with file paths.
        
        Args:
            before_file: Path to Excel file with data before training
            after_file: Path to Excel file with data after training
            output_dir: Directory to save output files and visualizations
        """
        self.before_file = before_file
        self.after_file = after_file
        self.output_dir = output_dir
        self.before_data = None
        self.after_data = None
        self.comparison_results = {}
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def load_data(self) -> bool:
        """
        Load both Excel files with comprehensive error handling.
        
        Returns:
            bool: True if both files loaded successfully, False otherwise
        """
        print("Loading Q-learning training data files...")
        
        try:
            # Load before training data
            print(f"Loading before training data from: {self.before_file}")
            if not os.path.exists(self.before_file):
                raise FileNotFoundError(f"Before training file not found: {self.before_file}")
            
            self.before_data = pd.read_excel(self.before_file)
            print(f"✓ Before training data loaded: {self.before_data.shape[0]} records, {self.before_data.shape[1]} columns")
            
            # Load after training data
            print(f"Loading after training data from: {self.after_file}")
            if not os.path.exists(self.after_file):
                raise FileNotFoundError(f"After training file not found: {self.after_file}")
            
            self.after_data = pd.read_excel(self.after_file)
            print(f"✓ After training data loaded: {self.after_data.shape[0]} records, {self.after_data.shape[1]} columns")
            
            # Validate data structure
            self._validate_data_structure()
            
            return True
            
        except Exception as e:
            print(f"✗ Error loading data files: {e}")
            return False
    
    def _validate_data_structure(self):
        """Validate that both datasets have compatible structure."""
        print("\nValidating data structure...")
        
        # Check if columns match
        before_cols = set(self.before_data.columns)
        after_cols = set(self.after_data.columns)
        
        if before_cols != after_cols:
            missing_before = after_cols - before_cols
            missing_after = before_cols - after_cols
            
            if missing_before:
                print(f"⚠ Columns missing in before data: {missing_before}")
            if missing_after:
                print(f"⚠ Columns missing in after data: {missing_after}")
        else:
            print("✓ Column structure matches between datasets")
        
        # Check for required columns
        required_cols = ['episode_step', 'action', 'reward', 'q_value']
        for col in required_cols:
            if col not in self.before_data.columns:
                print(f"⚠ Required column '{col}' missing in before data")
            if col not in self.after_data.columns:
                print(f"⚠ Required column '{col}' missing in after data")
        
        # Check data types
        print(f"✓ Data validation complete")
    
    def compare_dimensions(self) -> Dict:
        """
        Compare data dimensions and structure between datasets.
        
        Returns:
            Dict: Dimension comparison results
        """
        print("\n" + "="*60)
        print("COMPARING DATA DIMENSIONS AND STRUCTURE")
        print("="*60)
        
        results = {
            'before_shape': self.before_data.shape,
            'after_shape': self.after_data.shape,
            'records_added': self.after_data.shape[0] - self.before_data.shape[0],
            'columns_before': list(self.before_data.columns),
            'columns_after': list(self.after_data.columns),
        }
        
        print(f"Before training dataset:")
        print(f"  - Records: {results['before_shape'][0]:,}")
        print(f"  - Columns: {results['before_shape'][1]}")
        
        print(f"\nAfter training dataset:")
        print(f"  - Records: {results['after_shape'][0]:,}")
        print(f"  - Columns: {results['after_shape'][1]}")
        
        print(f"\nDifference:")
        print(f"  - New records added: {results['records_added']:,}")
        
        if results['records_added'] == 0:
            print("  ⚠ No new training records detected - datasets may be identical")
        elif results['records_added'] > 0:
            print(f"  ✓ Training expanded dataset by {results['records_added']} records")
        else:
            print(f"  ⚠ Dataset size decreased by {abs(results['records_added'])} records")
        
        self.comparison_results['dimensions'] = results
        return results
    
    def analyze_rewards(self) -> Dict:
        """
        Analyze reward distributions and improvements.
        
        Returns:
            Dict: Reward analysis results
        """
        print("\n" + "="*60)
        print("ANALYZING REWARDS AND PERFORMANCE")
        print("="*60)
        
        # Calculate reward statistics
        before_rewards = self.before_data['reward'].describe()
        after_rewards = self.after_data['reward'].describe()
        
        results = {
            'before_stats': before_rewards.to_dict(),
            'after_stats': after_rewards.to_dict(),
            'improvement': {
                'mean_change': after_rewards['mean'] - before_rewards['mean'],
                'max_change': after_rewards['max'] - before_rewards['max'],
                'min_change': after_rewards['min'] - before_rewards['min'],
                'std_change': after_rewards['std'] - before_rewards['std'],
            }
        }
        
        print("Reward Statistics Comparison:")
        print(f"{'Metric':<12} {'Before':<12} {'After':<12} {'Change':<12} {'% Change':<12}")
        print("-" * 60)
        
        for metric in ['mean', 'std', 'min', 'max']:
            before_val = before_rewards[metric]
            after_val = after_rewards[metric]
            change = after_val - before_val
            pct_change = (change / before_val) * 100 if before_val != 0 else 0
            
            print(f"{metric:<12} {before_val:<12.4f} {after_val:<12.4f} {change:<12.4f} {pct_change:<12.2f}%")
        
        # Statistical significance test
        if len(self.before_data) > 1 and len(self.after_data) > 1:
            t_stat, p_value = stats.ttest_ind(self.before_data['reward'], self.after_data['reward'])
            results['statistical_test'] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
            
            print(f"\nStatistical Significance Test (t-test):")
            print(f"  - t-statistic: {t_stat:.4f}")
            print(f"  - p-value: {p_value:.4f}")
            print(f"  - Significant improvement: {'Yes' if p_value < 0.05 else 'No'} (α = 0.05)")
        
        self.comparison_results['rewards'] = results
        return results
    
    def analyze_qvalues(self) -> Dict:
        """
        Analyze Q-value distributions and learning progression.
        
        Returns:
            Dict: Q-value analysis results
        """
        print("\n" + "="*60)
        print("ANALYZING Q-VALUES AND LEARNING PROGRESSION")
        print("="*60)
        
        # Q-value statistics
        before_qvals = self.before_data['q_value'].describe()
        after_qvals = self.after_data['q_value'].describe()
        
        results = {
            'before_stats': before_qvals.to_dict(),
            'after_stats': after_qvals.to_dict(),
            'improvement': {
                'mean_change': after_qvals['mean'] - before_qvals['mean'],
                'max_change': after_qvals['max'] - before_qvals['max'],
                'min_change': after_qvals['min'] - before_qvals['min'],
                'std_change': after_qvals['std'] - before_qvals['std'],
            }
        }
        
        print("Q-Value Statistics Comparison:")
        print(f"{'Metric':<12} {'Before':<12} {'After':<12} {'Change':<12} {'% Change':<12}")
        print("-" * 60)
        
        for metric in ['mean', 'std', 'min', 'max']:
            before_val = before_qvals[metric]
            after_val = after_qvals[metric]
            change = after_val - before_val
            pct_change = (change / before_val) * 100 if before_val != 0 else 0
            
            print(f"{metric:<12} {before_val:<12.4f} {after_val:<12.4f} {change:<12.4f} {pct_change:<12.2f}%")
        
        # Analyze Q-value convergence indicators
        before_variance = self.before_data['q_value'].var()
        after_variance = self.after_data['q_value'].var()
        
        results['convergence'] = {
            'before_variance': before_variance,
            'after_variance': after_variance,
            'variance_reduction': before_variance - after_variance,
            'convergence_indicator': (before_variance - after_variance) / before_variance if before_variance > 0 else 0
        }
        
        print(f"\nQ-Value Convergence Analysis:")
        print(f"  - Before variance: {before_variance:.6f}")
        print(f"  - After variance: {after_variance:.6f}")
        print(f"  - Variance reduction: {before_variance - after_variance:.6f}")
        print(f"  - Convergence indicator: {results['convergence']['convergence_indicator']:.2%}")
        
        self.comparison_results['qvalues'] = results
        return results
    
    def analyze_actions(self) -> Dict:
        """
        Analyze action distributions and policy changes.
        
        Returns:
            Dict: Action analysis results
        """
        print("\n" + "="*60)
        print("ANALYZING ACTION DISTRIBUTIONS AND POLICY")
        print("="*60)
        
        # Action distribution analysis
        before_actions = self.before_data['action'].value_counts().sort_index()
        after_actions = self.after_data['action'].value_counts().sort_index()
        
        # Ensure both have same actions (fill missing with 0)
        all_actions = sorted(set(before_actions.index) | set(after_actions.index))
        before_dist = pd.Series([before_actions.get(a, 0) for a in all_actions], index=all_actions)
        after_dist = pd.Series([after_actions.get(a, 0) for a in all_actions], index=all_actions)
        
        # Convert to percentages
        before_pct = (before_dist / before_dist.sum()) * 100
        after_pct = (after_dist / after_dist.sum()) * 100
        
        results = {
            'before_distribution': before_dist.to_dict(),
            'after_distribution': after_dist.to_dict(),
            'before_percentages': before_pct.to_dict(),
            'after_percentages': after_pct.to_dict(),
            'changes': (after_pct - before_pct).to_dict()
        }
        
        print("Action Distribution Comparison:")
        print(f"{'Action':<8} {'Before Count':<12} {'Before %':<10} {'After Count':<12} {'After %':<10} {'Change %':<10}")
        print("-" * 70)
        
        for action in all_actions:
            before_count = before_dist[action]
            after_count = after_dist[action]
            before_p = before_pct[action]
            after_p = after_pct[action]
            change = after_p - before_p
            
            print(f"{action:<8} {before_count:<12} {before_p:<10.1f} {after_count:<12} {after_p:<10.1f} {change:<10.1f}")
        
        # Calculate policy diversity (entropy)
        def calculate_entropy(probabilities):
            probabilities = probabilities / probabilities.sum()  # Normalize
            probabilities = probabilities[probabilities > 0]  # Remove zeros
            return -np.sum(probabilities * np.log2(probabilities))
        
        before_entropy = calculate_entropy(before_dist)
        after_entropy = calculate_entropy(after_dist)
        
        results['policy_diversity'] = {
            'before_entropy': before_entropy,
            'after_entropy': after_entropy,
            'entropy_change': after_entropy - before_entropy
        }
        
        print(f"\nPolicy Diversity Analysis (Entropy):")
        print(f"  - Before entropy: {before_entropy:.3f}")
        print(f"  - After entropy: {after_entropy:.3f}")
        print(f"  - Change: {after_entropy - before_entropy:.3f}")
        print(f"  - Policy became: {'more diverse' if after_entropy > before_entropy else 'more focused'}")
        
        self.comparison_results['actions'] = results
        return results
    
    def analyze_states(self) -> Dict:
        """
        Analyze state space coverage and variations.
        
        Returns:
            Dict: State analysis results
        """
        print("\n" + "="*60)
        print("ANALYZING STATE SPACE COVERAGE")
        print("="*60)
        
        # Identify state columns
        state_cols = [col for col in self.before_data.columns if col.startswith('state_')]
        next_state_cols = [col for col in self.before_data.columns if col.startswith('next_state_')]
        
        results = {
            'state_columns': state_cols,
            'next_state_columns': next_state_cols,
            'state_statistics': {},
            'next_state_statistics': {}
        }
        
        if state_cols:
            print("Current State Analysis:")
            print(f"{'State Feature':<25} {'Before Mean':<12} {'After Mean':<12} {'Change':<12}")
            print("-" * 61)
            
            for col in state_cols:
                before_mean = self.before_data[col].mean()
                after_mean = self.after_data[col].mean()
                change = after_mean - before_mean
                
                results['state_statistics'][col] = {
                    'before_mean': before_mean,
                    'after_mean': after_mean,
                    'change': change
                }
                
                print(f"{col:<25} {before_mean:<12.4f} {after_mean:<12.4f} {change:<12.4f}")
        
        if next_state_cols:
            print(f"\nNext State Analysis:")
            print(f"{'Next State Feature':<25} {'Before Mean':<12} {'After Mean':<12} {'Change':<12}")
            print("-" * 61)
            
            for col in next_state_cols:
                before_mean = self.before_data[col].mean()
                after_mean = self.after_data[col].mean()
                change = after_mean - before_mean
                
                results['next_state_statistics'][col] = {
                    'before_mean': before_mean,
                    'after_mean': after_mean,
                    'change': change
                }
                
                print(f"{col:<25} {before_mean:<12.4f} {after_mean:<12.4f} {change:<12.4f}")
        
        self.comparison_results['states'] = results
        return results
    
    def detect_data_differences(self) -> Dict:
        """
        Detect specific differences between datasets at the record level.
        
        Returns:
            Dict: Detailed difference analysis
        """
        print("\n" + "="*60)
        print("DETECTING DETAILED DATA DIFFERENCES")
        print("="*60)
        
        results = {
            'identical_records': 0,
            'different_records': 0,
            'differences_by_column': {},
            'new_records': 0,
            'removed_records': 0
        }
        
        # If datasets have different sizes, analyze differences
        if len(self.before_data) != len(self.after_data):
            results['new_records'] = max(0, len(self.after_data) - len(self.before_data))
            results['removed_records'] = max(0, len(self.before_data) - len(self.after_data))
            print(f"Dataset size changed:")
            print(f"  - New records: {results['new_records']}")
            print(f"  - Removed records: {results['removed_records']}")
        
        # Compare overlapping records
        min_len = min(len(self.before_data), len(self.after_data))
        
        if min_len > 0:
            # Compare record by record for overlapping portion
            for col in self.before_data.columns:
                if col in self.after_data.columns:
                    before_vals = self.before_data[col].iloc[:min_len]
                    after_vals = self.after_data[col].iloc[:min_len]
                    
                    # For numeric columns, use tolerance for floating point comparison
                    if pd.api.types.is_numeric_dtype(before_vals):
                        differences = ~np.isclose(before_vals, after_vals, rtol=1e-9, atol=1e-9, equal_nan=True)
                    else:
                        differences = before_vals != after_vals
                    
                    diff_count = differences.sum()
                    results['differences_by_column'][col] = diff_count
                    
                    if diff_count > 0:
                        print(f"Column '{col}': {diff_count} differences out of {min_len} records ({diff_count/min_len:.1%})")
            
            # Count identical vs different records
            total_diffs = sum(results['differences_by_column'].values())
            if total_diffs == 0:
                results['identical_records'] = min_len
                print(f"\n✓ All {min_len} overlapping records are identical")
            else:
                # This is a simplified count - a more sophisticated approach would check row-wise
                results['different_records'] = len([col for col, count in results['differences_by_column'].items() if count > 0])
                print(f"\n⚠ Found differences in {results['different_records']} columns")
        
        self.comparison_results['differences'] = results
        return results
    
    def generate_visualizations(self):
        """Generate comprehensive visualizations of the comparison."""
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        
        # Set up the figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Reward distribution comparison
        plt.subplot(3, 3, 1)
        plt.hist(self.before_data['reward'], alpha=0.7, label='Before Training', bins=30, density=True)
        plt.hist(self.after_data['reward'], alpha=0.7, label='After Training', bins=30, density=True)
        plt.xlabel('Reward Value')
        plt.ylabel('Density')
        plt.title('Reward Distribution Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Q-value distribution comparison
        plt.subplot(3, 3, 2)
        plt.hist(self.before_data['q_value'], alpha=0.7, label='Before Training', bins=30, density=True)
        plt.hist(self.after_data['q_value'], alpha=0.7, label='After Training', bins=30, density=True)
        plt.xlabel('Q-Value')
        plt.ylabel('Density')
        plt.title('Q-Value Distribution Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Action distribution comparison
        plt.subplot(3, 3, 3)
        actions_before = self.before_data['action'].value_counts().sort_index()
        actions_after = self.after_data['action'].value_counts().sort_index()
        
        x_pos = np.arange(len(actions_before))
        width = 0.35
        
        plt.bar(x_pos - width/2, actions_before.values, width, label='Before Training', alpha=0.8)
        plt.bar(x_pos + width/2, actions_after.values, width, label='After Training', alpha=0.8)
        plt.xlabel('Action')
        plt.ylabel('Frequency')
        plt.title('Action Distribution Comparison')
        plt.xticks(x_pos, actions_before.index)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Reward progression over episodes
        plt.subplot(3, 3, 4)
        if 'episode_step' in self.before_data.columns:
            # Calculate moving averages for smoother visualization
            window_size = max(1, len(self.before_data) // 50)
            before_rewards_smooth = self.before_data['reward'].rolling(window=window_size, min_periods=1).mean()
            after_rewards_smooth = self.after_data['reward'].rolling(window=window_size, min_periods=1).mean()
            
            plt.plot(self.before_data['episode_step'], before_rewards_smooth, label='Before Training', alpha=0.8)
            plt.plot(self.after_data['episode_step'], after_rewards_smooth, label='After Training', alpha=0.8)
            plt.xlabel('Episode Step')
            plt.ylabel('Reward (Moving Average)')
            plt.title('Reward Progression Over Time')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 5. Q-value progression over episodes
        plt.subplot(3, 3, 5)
        if 'episode_step' in self.before_data.columns:
            window_size = max(1, len(self.before_data) // 50)
            before_qvals_smooth = self.before_data['q_value'].rolling(window=window_size, min_periods=1).mean()
            after_qvals_smooth = self.after_data['q_value'].rolling(window=window_size, min_periods=1).mean()
            
            plt.plot(self.before_data['episode_step'], before_qvals_smooth, label='Before Training', alpha=0.8)
            plt.plot(self.after_data['episode_step'], after_qvals_smooth, label='After Training', alpha=0.8)
            plt.xlabel('Episode Step')
            plt.ylabel('Q-Value (Moving Average)')
            plt.title('Q-Value Progression Over Time')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 6. Correlation heatmap for state features (before)
        plt.subplot(3, 3, 6)
        state_cols = [col for col in self.before_data.columns if 'state_' in col][:6]  # Limit to first 6
        if state_cols:
            corr_before = self.before_data[state_cols].corr()
            sns.heatmap(corr_before, annot=True, cmap='coolwarm', center=0, square=True,
                       fmt='.2f', cbar_kws={'shrink': 0.8})
            plt.title('State Features Correlation (Before)')
        
        # 7. Correlation heatmap for state features (after)
        plt.subplot(3, 3, 7)
        if state_cols:
            corr_after = self.after_data[state_cols].corr()
            sns.heatmap(corr_after, annot=True, cmap='coolwarm', center=0, square=True,
                       fmt='.2f', cbar_kws={'shrink': 0.8})
            plt.title('State Features Correlation (After)')
        
        # 8. Reward vs Q-value scatter plot
        plt.subplot(3, 3, 8)
        sample_size = min(1000, len(self.before_data))  # Sample for performance
        before_sample = self.before_data.sample(n=sample_size, random_state=42)
        after_sample = self.after_data.sample(n=sample_size, random_state=42)
        
        plt.scatter(before_sample['reward'], before_sample['q_value'], 
                   alpha=0.6, label='Before Training', s=20)
        plt.scatter(after_sample['reward'], after_sample['q_value'], 
                   alpha=0.6, label='After Training', s=20)
        plt.xlabel('Reward')
        plt.ylabel('Q-Value')
        plt.title('Reward vs Q-Value Relationship')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 9. Summary statistics comparison
        plt.subplot(3, 3, 9)
        metrics = ['reward', 'q_value']
        before_means = [self.before_data[m].mean() for m in metrics]
        after_means = [self.after_data[m].mean() for m in metrics]
        
        x_pos = np.arange(len(metrics))
        width = 0.35
        
        plt.bar(x_pos - width/2, before_means, width, label='Before Training', alpha=0.8)
        plt.bar(x_pos + width/2, after_means, width, label='After Training', alpha=0.8)
        plt.xlabel('Metrics')
        plt.ylabel('Mean Value')
        plt.title('Mean Statistics Comparison')
        plt.xticks(x_pos, metrics)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the comprehensive visualization
        viz_path = os.path.join(self.output_dir, 'qlearning_comparison_visualization.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comprehensive visualization saved to: {viz_path}")
        
        plt.close()
        
        # Generate additional focused plots if there are significant differences
        self._generate_detailed_plots()
    
    def _generate_detailed_plots(self):
        """Generate additional detailed plots for specific analyses."""
        
        # Learning curve comparison (if we have episode data)
        if 'episode_step' in self.before_data.columns:
            plt.figure(figsize=(12, 8))
            
            # Create bins for episode steps to show learning progression
            max_steps = max(self.before_data['episode_step'].max(), self.after_data['episode_step'].max())
            bins = np.linspace(0, max_steps, 20)
            
            # Calculate mean rewards per bin
            before_binned = pd.cut(self.before_data['episode_step'], bins)
            after_binned = pd.cut(self.after_data['episode_step'], bins)
            
            before_means = self.before_data.groupby(before_binned)['reward'].mean()
            after_means = self.after_data.groupby(after_binned)['reward'].mean()
            
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            plt.subplot(2, 1, 1)
            plt.plot(bin_centers, before_means, marker='o', label='Before Training', linewidth=2)
            plt.plot(bin_centers, after_means, marker='s', label='After Training', linewidth=2)
            plt.xlabel('Episode Step')
            plt.ylabel('Mean Reward')
            plt.title('Learning Curve: Reward Progression')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Q-value progression
            before_qmeans = self.before_data.groupby(before_binned)['q_value'].mean()
            after_qmeans = self.after_data.groupby(after_binned)['q_value'].mean()
            
            plt.subplot(2, 1, 2)
            plt.plot(bin_centers, before_qmeans, marker='o', label='Before Training', linewidth=2)
            plt.plot(bin_centers, after_qmeans, marker='s', label='After Training', linewidth=2)
            plt.xlabel('Episode Step')
            plt.ylabel('Mean Q-Value')
            plt.title('Learning Curve: Q-Value Progression')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            learning_curve_path = os.path.join(self.output_dir, 'learning_curves.png')
            plt.savefig(learning_curve_path, dpi=300, bbox_inches='tight')
            print(f"✓ Learning curves saved to: {learning_curve_path}")
            plt.close()
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive text report of the comparison.
        
        Returns:
            str: Path to the generated report file
        """
        print("\n" + "="*60)
        print("GENERATING COMPREHENSIVE REPORT")
        print("="*60)
        
        report_path = os.path.join(self.output_dir, 'qlearning_comparison_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("Q-LEARNING TRAINING DATA COMPARISON REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Before training file: {self.before_file}\n")
            f.write(f"After training file: {self.after_file}\n\n")
            
            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 20 + "\n")
            
            if 'dimensions' in self.comparison_results:
                dims = self.comparison_results['dimensions']
                f.write(f"• Dataset size: {dims['before_shape'][0]:,} → {dims['after_shape'][0]:,} records\n")
                f.write(f"• New training data: {dims['records_added']:,} records\n")
            
            if 'rewards' in self.comparison_results:
                rewards = self.comparison_results['rewards']
                mean_change = rewards['improvement']['mean_change']
                f.write(f"• Mean reward change: {mean_change:+.4f}\n")
                
                if 'statistical_test' in rewards:
                    significance = "Yes" if rewards['statistical_test']['significant'] else "No"
                    f.write(f"• Statistically significant improvement: {significance}\n")
            
            if 'qvalues' in self.comparison_results:
                qvals = self.comparison_results['qvalues']
                convergence = qvals['convergence']['convergence_indicator']
                f.write(f"• Q-value convergence indicator: {convergence:.2%}\n")
            
            # Detailed Analysis
            f.write("\nDETAILED ANALYSIS\n")
            f.write("-" * 20 + "\n\n")
            
            # Write each analysis section
            for section_name, section_data in self.comparison_results.items():
                f.write(f"{section_name.upper()} ANALYSIS\n")
                f.write("." * 30 + "\n")
                
                if section_name == 'rewards' and 'before_stats' in section_data:
                    f.write("Reward Statistics:\n")
                    for metric in ['mean', 'std', 'min', 'max']:
                        before_val = section_data['before_stats'][metric]
                        after_val = section_data['after_stats'][metric]
                        change = section_data['improvement'][f'{metric}_change']
                        f.write(f"  {metric}: {before_val:.4f} → {after_val:.4f} (Δ{change:+.4f})\n")
                
                elif section_name == 'qvalues' and 'before_stats' in section_data:
                    f.write("Q-Value Statistics:\n")
                    for metric in ['mean', 'std', 'min', 'max']:
                        before_val = section_data['before_stats'][metric]
                        after_val = section_data['after_stats'][metric]
                        change = section_data['improvement'][f'{metric}_change']
                        f.write(f"  {metric}: {before_val:.4f} → {after_val:.4f} (Δ{change:+.4f})\n")
                    
                    if 'convergence' in section_data:
                        conv = section_data['convergence']
                        f.write(f"  Variance reduction: {conv['variance_reduction']:.6f}\n")
                        f.write(f"  Convergence indicator: {conv['convergence_indicator']:.2%}\n")
                
                elif section_name == 'actions' and 'before_distribution' in section_data:
                    f.write("Action Distribution Changes:\n")
                    for action, change in section_data['changes'].items():
                        f.write(f"  Action {action}: {change:+.1f}%\n")
                    
                    if 'policy_diversity' in section_data:
                        entropy_change = section_data['policy_diversity']['entropy_change']
                        f.write(f"  Policy diversity change: {entropy_change:+.3f}\n")
                
                f.write("\n")
            
            # Conclusions and Recommendations
            f.write("CONCLUSIONS AND RECOMMENDATIONS\n")
            f.write("-" * 35 + "\n")
            
            # Analyze if learning actually occurred
            learning_indicators = []
            
            if 'rewards' in self.comparison_results:
                mean_reward_change = self.comparison_results['rewards']['improvement']['mean_change']
                if mean_reward_change > 0:
                    learning_indicators.append("✓ Mean reward increased")
                else:
                    learning_indicators.append("✗ Mean reward decreased or unchanged")
            
            if 'qvalues' in self.comparison_results:
                convergence = self.comparison_results['qvalues']['convergence']['convergence_indicator']
                if convergence > 0:
                    learning_indicators.append("✓ Q-values show convergence")
                else:
                    learning_indicators.append("✗ Q-values show divergence")
            
            if 'dimensions' in self.comparison_results:
                new_records = self.comparison_results['dimensions']['records_added']
                if new_records > 0:
                    learning_indicators.append("✓ New training data was added")
                else:
                    learning_indicators.append("✗ No new training data detected")
            
            f.write("Learning Indicators:\n")
            for indicator in learning_indicators:
                f.write(f"  {indicator}\n")
            
            positive_indicators = sum(1 for ind in learning_indicators if ind.startswith("✓"))
            total_indicators = len(learning_indicators)
            
            f.write(f"\nOverall Assessment: {positive_indicators}/{total_indicators} positive indicators\n")
            
            if positive_indicators >= total_indicators * 0.7:
                f.write("CONCLUSION: Strong evidence of learning and improvement\n")
            elif positive_indicators >= total_indicators * 0.5:
                f.write("CONCLUSION: Moderate evidence of learning\n")
            else:
                f.write("CONCLUSION: Limited or no evidence of learning improvement\n")
            
            # Recommendations
            f.write("\nRecommendations:\n")
            if new_records <= 0:
                f.write("  • Consider running more training episodes to generate new data\n")
            if mean_reward_change <= 0:
                f.write("  • Review reward function and hyperparameters\n")
                f.write("  • Consider adjusting learning rate or exploration strategy\n")
            if convergence <= 0:
                f.write("  • Monitor Q-value stability and consider early stopping criteria\n")
            
            f.write("  • Continue monitoring learning progress with additional training\n")
            f.write("  • Consider implementing learning curves visualization for real-time monitoring\n")
        
        print(f"✓ Comprehensive report saved to: {report_path}")
        return report_path
    
    def run_complete_analysis(self) -> bool:
        """
        Run the complete comparison analysis pipeline.
        
        Returns:
            bool: True if analysis completed successfully
        """
        print("="*80)
        print("Q-LEARNING TRAINING DATA ANALYSIS")
        print("="*80)
        print(f"Comparing training datasets:")
        print(f"  Before: {self.before_file}")
        print(f"  After:  {self.after_file}")
        print(f"  Output: {self.output_dir}")
        
        try:
            # Load data
            if not self.load_data():
                return False
            
            # Run all analyses
            self.compare_dimensions()
            self.analyze_rewards()
            self.analyze_qvalues()
            self.analyze_actions()
            self.analyze_states()
            self.detect_data_differences()
            
            # Generate outputs
            self.generate_visualizations()
            report_path = self.generate_report()
            
            print("\n" + "="*80)
            print("ANALYSIS COMPLETE")
            print("="*80)
            print(f"✓ All analyses completed successfully")
            print(f"✓ Results saved to: {self.output_dir}")
            print(f"✓ Report available at: {report_path}")
            print(f"✓ Visualizations generated")
            
            return True
            
        except Exception as e:
            print(f"✗ Error during analysis: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main function to run the Q-learning data comparison."""
    parser = argparse.ArgumentParser(
        description="Analyze and compare Q-learning training data from Excel files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python qlearning_data_comparator.py --before q_table_before_training.xlsx --after q_table_training.xlsx
  python qlearning_data_comparator.py --before data/before.xlsx --after data/after.xlsx --output results/
        """
    )
    
    parser.add_argument('--before', required=True, 
                       help='Path to Excel file with data before training')
    parser.add_argument('--after', required=True,
                       help='Path to Excel file with data after training')
    parser.add_argument('--output', default='comparison_results',
                       help='Output directory for results (default: comparison_results)')
    
    args = parser.parse_args()
    
    # Create and run the comparator
    comparator = QTableComparator(args.before, args.after, args.output)
    success = comparator.run_complete_analysis()
    
    if success:
        print("\n🎉 Analysis completed successfully!")
        return 0
    else:
        print("\n❌ Analysis failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())