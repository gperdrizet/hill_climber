"""Plotting functions for hill climbing optimization."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


def plot_input_data(data, plot_type='scatter'):
    """Plot input data distribution.
    
    Args:
        data: numpy array (Nx2) or pandas DataFrame with 2 columns
        plot_type: Type of plot - 'scatter' or 'kde' (default: 'scatter')
    
    Raises:
        ValueError: If plot_type is not 'scatter' or 'kde'
    """
    if plot_type not in ['scatter', 'kde']:
        raise ValueError(f"plot_type must be 'scatter' or 'kde', got '{plot_type}'")
    
    # Extract columns
    if isinstance(data, pd.DataFrame):
        cols = data.columns.tolist()
        x, y = data[cols[0]], data[cols[1]]
        x_label, y_label = cols[0], cols[1]

    else:
        x, y = data[:, 0], data[:, 1]
        x_label, y_label = 'x', 'y'
    
    if plot_type == 'scatter':
        plt.figure(figsize=(5, 5))
        plt.title('Input distributions')
        plt.scatter(x, y, s=5, color='black')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()

    else:  # kde
        plt.figure(figsize=(6, 4))
        plt.title('Input distributions (KDE)', fontsize=14)
        
        # Convert to numpy arrays
        x_data = np.array(x)
        y_data = np.array(y)
        
        try:
            # Create KDE for both distributions
            kde_x = gaussian_kde(x_data)
            kde_y = gaussian_kde(y_data)
            
            # Create evaluation points
            x_min = min(x_data.min(), y_data.min())
            x_max = max(x_data.max(), y_data.max())
            x_eval = np.linspace(x_min, x_max, 200)
            
            # Evaluate KDEs
            density_x = kde_x(x_eval)
            density_y = kde_y(x_eval)
            
            # Plot overlapping KDEs with standard matplotlib colors
            plt.plot(x_eval, density_x, label=x_label, linewidth=2, alpha=0.8)
            plt.fill_between(x_eval, density_x, alpha=0.3)
            plt.plot(x_eval, density_y, label=y_label, linewidth=2, alpha=0.8)
            plt.fill_between(x_eval, density_y, alpha=0.3)
            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
        except (np.linalg.LinAlgError, ValueError) as e:
            # Fall back to histograms if KDE fails
            plt.hist(x_data, bins=20, alpha=0.6, label=x_label, edgecolor='black')
            plt.hist(y_data, bins=20, alpha=0.6, label=y_label, edgecolor='black')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.title('Input distributions (histogram)')
            plt.legend()
            
        plt.tight_layout()
        plt.show()



def plot_results(results, plot_type='scatter', metrics=None):
    """Visualize hill climbing results with progress and snapshots.
    
    Creates a comprehensive visualization showing:
    - Progress plot with metrics and objective value over time
    - Snapshot plots at 25%, 50%, 75%, and 100% completion
    
    Args:
        results: List of (best_data, steps_df) tuples from climb_parallel()
        plot_type: Type of snapshot plots - 'scatter' or 'histogram' (default: 'scatter')
                   Note: 'histogram' uses KDE (Kernel Density Estimation) plots
        metrics: List of metric names to display in progress plots and snapshots.
                 If None (default), all available metrics are shown.
                 Example: ['Pearson', 'Spearman'] or ['Mean X', 'Std X']
    
    Raises:
        ValueError: If plot_type is not 'scatter' or 'histogram'
        ValueError: If any specified metric is not found in the results
    """
    if plot_type not in ['scatter', 'histogram']:
        raise ValueError(f"plot_type must be 'scatter' or 'histogram', got '{plot_type}'")
    
    # Validate metrics if provided
    if metrics is not None:
        _, steps_df = results[0]
        available_metrics = [col for col in steps_df.columns 
                            if col not in ['Step', 'Objective value', 'Best_data']]
        invalid_metrics = [m for m in metrics if m not in available_metrics]
        if invalid_metrics:
            raise ValueError(f"Metrics not found in results: {invalid_metrics}. "
                           f"Available metrics: {available_metrics}")
    
    if plot_type == 'scatter':
        _plot_results_scatter(results, metrics)
    else:
        _plot_results_histogram(results, metrics)


def _plot_results_scatter(results, metrics=None):
    """Internal function: Visualize results with scatter plots.
    
    Args:
        results: List of (best_data, steps_df) tuples from climb_parallel()
        metrics: List of metric names to display, or None for all metrics
    """
    n_replicates = len(results)
    fig = plt.figure(constrained_layout=True, figsize=(15, 2.5*n_replicates))
    spec = fig.add_gridspec(nrows=n_replicates, ncols=5, width_ratios=[1.1, 1, 1, 1, 1])
    fig.suptitle('Hill climb results (Scatter plots)', fontsize=16)

    for i in range(n_replicates):
        best_data, steps_df = results[i]
        
        # Get metric columns
        all_metric_columns = [col for col in steps_df.columns 
                              if col not in ['Step', 'Objective value', 'Best_data']]
        
        # Use specified metrics or all available metrics
        metric_columns = metrics if metrics is not None else all_metric_columns
        
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


def _plot_results_histogram(results, metrics=None):
    """Internal function: Visualize results with KDE plots.
    
    Args:
        results: List of (best_data, steps_df) tuples from climb_parallel()
        metrics: List of metric names to display, or None for all metrics
    """
    n_replicates = len(results)
    fig = plt.figure(constrained_layout=True, figsize=(15, 2.5*n_replicates))
    spec = fig.add_gridspec(nrows=n_replicates, ncols=5, width_ratios=[1.1, 1, 1, 1, 1])
    fig.suptitle('Hill climb results (KDE plots)', fontsize=16)

    for i in range(n_replicates):
        best_data, steps_df = results[i]
        
        # Get metric columns
        all_metric_columns = [col for col in steps_df.columns 
                              if col not in ['Step', 'Objective value', 'Best_data']]
        
        # Use specified metrics or all available metrics
        metric_columns = metrics if metrics is not None else all_metric_columns
        
        # Progress plot (same as scatter version)
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
        
        # Snapshot histograms at 25%, 50%, 75%, 100%
        for j, (pct, label) in enumerate(zip([0.25, 0.50, 0.75, 1.0], ['25%', '50%', '75%', '100%'])):
            ax = fig.add_subplot(spec[i, j+1])
            
            step_idx = max(0, min(int(len(steps_df) * pct) - 1, len(steps_df) - 1))
            snapshot_data = steps_df['Best_data'].iloc[step_idx]
            
            # Extract x and y
            if isinstance(snapshot_data, pd.DataFrame):
                snap_x, snap_y = snapshot_data.iloc[:, 0], snapshot_data.iloc[:, 1]
                x_label = snapshot_data.columns[0]
                y_label = snapshot_data.columns[1]
            else:
                snap_x, snap_y = snapshot_data[:, 0], snapshot_data[:, 1]
                x_label = 'x'
                y_label = 'y'
            
            ax.set_title(f'Climb {label} complete', fontsize=10)
            
            # Create KDE plots
            # Convert to numpy arrays if needed
            x_data = np.array(snap_x)
            y_data = np.array(snap_y)
            
            # Create KDE for both distributions
            try:
                kde_x = gaussian_kde(x_data)
                kde_y = gaussian_kde(y_data)
                
                # Create evaluation points
                x_min = min(x_data.min(), y_data.min())
                x_max = max(x_data.max(), y_data.max())
                x_eval = np.linspace(x_min, x_max, 200)
                
                # Evaluate KDEs
                density_x = kde_x(x_eval)
                density_y = kde_y(x_eval)
                
                # Plot KDE curves
                ax.plot(x_eval, density_x, label=x_label, color='blue', linewidth=2, alpha=0.8)
                ax.fill_between(x_eval, density_x, alpha=0.3, color='blue')
                ax.plot(x_eval, density_y, label=y_label, color='red', linewidth=2, alpha=0.8)
                ax.fill_between(x_eval, density_y, alpha=0.3, color='red')
                
                ax.set_xlabel('Value')
                ax.set_ylabel('Density')
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)
                
            except (np.linalg.LinAlgError, ValueError) as e:
                # If KDE fails (e.g., all values identical), fall back to histogram
                ax.hist(x_data, bins=20, alpha=0.6, label=x_label, color='blue', edgecolor='black')
                ax.hist(y_data, bins=20, alpha=0.6, label=y_label, color='red', edgecolor='black')
                ax.set_xlabel('Value')
                ax.set_ylabel('Frequency')
                ax.legend(fontsize=7)
            
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
