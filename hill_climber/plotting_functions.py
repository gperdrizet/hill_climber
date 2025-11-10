"""Plotting functions for hill climbing optimization."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_input_data(data):
    """Plot input data distribution.
    
    Args:
        data: numpy array (Nx2) or pandas DataFrame with 2 columns
    """
    # Extract columns
    if isinstance(data, pd.DataFrame):
        cols = data.columns.tolist()
        x, y = data[cols[0]], data[cols[1]]
        x_label, y_label = cols[0], cols[1]
    else:
        x, y = data[:, 0], data[:, 1]
        x_label, y_label = 'x', 'y'
    
    plt.figure(figsize=(6, 6))
    plt.title('Input distributions')
    plt.scatter(x, y, s=5, color='black')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def plot_results(results):
    """Visualize hill climbing results with progress and snapshots.
    
    Creates a comprehensive visualization showing:
    - Progress plot with metrics and objective value over time
    - Snapshot plots at 25%, 50%, 75%, and 100% completion
    
    Args:
        results: List of (best_data, steps_df) tuples from climb_parallel()
    """
    n_replicates = len(results)
    fig = plt.figure(constrained_layout=True, figsize=(15, 2.5*n_replicates))
    spec = fig.add_gridspec(nrows=n_replicates, ncols=5, width_ratios=[1.1, 1, 1, 1, 1])
    fig.suptitle('Hill climb results', fontsize=16)

    for i in range(n_replicates):
        best_data, steps_df = results[i]
        
        # Get metric columns
        metric_columns = [col for col in steps_df.columns 
                          if col not in ['Step', 'Objective value', 'Best_data']]
        
        # Progress plot
        ax = fig.add_subplot(spec[i, 0])
        ax.set_title(f'Replicate {i+1}: Progress', fontsize=10)
        
        lines = []
        for metric_name in metric_columns:
            lines.extend(ax.plot(steps_df['Step'] / 100000, steps_df[metric_name], label=metric_name))
        
        ax.set_xlabel('Step (x 100000)')
        ax.set_ylabel('Metrics', color='black')
        
        ax2 = ax.twinx()
        lines.extend(ax2.plot(steps_df['Step'] / 100000, steps_df['Objective value'], 
                              label='Objective', color='black'))
        ax2.set_ylabel('Objective value', color='black')
        ax2.legend(lines, [l.get_label() for l in lines], loc='best', fontsize=7)
        
        # Snapshot plots at 25%, 50%, 75%, 100%
        for j, (pct, label) in enumerate(zip([0.25, 0.50, 0.75, 1.0], ['25%', '50%', '75%', '100%'])):
            ax = fig.add_subplot(spec[i, j+1])
            
            step_idx = max(0, min(int(len(steps_df) * pct) - 1, len(steps_df) - 1))
            snapshot_data = steps_df['Best_data'].iloc[step_idx]
            
            # Extract x and y
            if isinstance(snapshot_data, pd.DataFrame):
                snap_x, snap_y = snapshot_data.iloc[:, 0], snapshot_data.iloc[:, 1]
            else:
                snap_x, snap_y = snapshot_data[:, 0], snapshot_data[:, 1]
            
            ax.set_title(f'Climb {label} complete', fontsize=10)
            ax.scatter(snap_x, snap_y, color='black', s=1)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            
            # Build stats text
            obj_val = steps_df['Objective value'].iloc[step_idx]
            stats_text = f'Obj={obj_val:.4f}\n'
            for metric_name in metric_columns:
                abbrev = ''.join([word[0] for word in metric_name.split()])
                stats_text += f'{abbrev}={steps_df[metric_name].iloc[step_idx]:.3f}\n'
            
            ax.text(0.04, 0.95, stats_text.strip(), transform=ax.transAxes,
                    fontsize=7, verticalalignment='top',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

    plt.show()


def plot_solution(data, title='Solution'):
    """Plot solution data with correlation statistics and distributions.
    
    Args:
        data: numpy array (Nx2) or pandas DataFrame with 2 columns
        title: String for the plot title (default: 'Solution')
    """
    # Extract columns
    if isinstance(data, pd.DataFrame):
        cols = data.columns.tolist()
        x, y = data[cols[0]], data[cols[1]]
        x_label, y_label = cols[0], cols[1]
    else:
        x, y = data[:, 0], data[:, 1]
        x_label, y_label = 'x', 'y'
    
    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    
    fig.suptitle(title, fontsize=14)
    
    # Calculate correlation coefficients
    pearson_corr = pd.Series(x).corr(pd.Series(y), method='pearson')
    spearman_corr = pd.Series(x).corr(pd.Series(y), method='spearman')
    corr_diff = abs(spearman_corr - pearson_corr)
    
    # Calculate linear regression
    coeffs = np.polyfit(x, y, 1)
    poly = np.poly1d(coeffs)
    x_line = np.linspace(x.min() if hasattr(x, 'min') else np.min(x), 
                         x.max() if hasattr(x, 'max') else np.max(x), 100)
    y_line = poly(x_line)
    
    # Left plot: Scatter plot with regression line and statistics
    axs[0].set_title('Vectors')
    axs[0].scatter(x, y, color='black', s=1)
    axs[0].plot(x_line, y_line, 'r-', linewidth=2, label='Linear Regression')
    axs[0].set_xlabel(x_label)
    axs[0].set_ylabel(y_label)
    
    # Add correlation statistics
    stats_text = f'Pearson: {pearson_corr:.4f}\n'
    stats_text += f'Spearman: {spearman_corr:.4f}\n'
    stats_text += f'Difference: {corr_diff:.4f}'
    
    axs[0].text(0.02, 0.98, stats_text, transform=axs[0].transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(facecolor='lightgrey', alpha=0.8))
    
    # Middle plot: x distribution
    axs[1].set_title(f'{x_label} distribution')
    axs[1].hist(x, bins=30, color='gray', edgecolor='black')
    axs[1].set_xlabel(x_label)
    axs[1].set_ylabel('Count')
    
    # Right plot: y distribution
    axs[2].set_title(f'{y_label} distribution')
    axs[2].hist(y, bins=30, color='gray', edgecolor='black')
    axs[2].set_xlabel(y_label)
    axs[2].set_ylabel('Count')
    
    plt.tight_layout()
    plt.show()
