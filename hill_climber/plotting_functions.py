"""Plotting functions for hill climbing optimization."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


def plot_input_data(data, plot_type='scatter'):
    """Plot input data distribution.
    
    Args:
        data: numpy array (Nx2) or pandas DataFrame with 2 columns
        plot_type: Type of plot, either ``'scatter'`` or ``'kde'``
                   (default is ``'scatter'``)
    
    Raises:
        ValueError: If plot_type is not ``'scatter'`` or ``'kde'``
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

    else:  # kde

        plt.figure(figsize=(6, 4))
        plt.title('Input distributions (KDE)', fontsize=14)
        
        x_data, y_data = np.array(x), np.array(y)
        
        try:
            # Create KDE
            kde_x, kde_y = gaussian_kde(x_data), gaussian_kde(y_data)
            
            # Create evaluation range
            x_min, x_max = min(x_data.min(), y_data.min()), max(x_data.max(), y_data.max())
            x_eval = np.linspace(x_min, x_max, 200)
            
            # Plot KDEs
            plt.plot(x_eval, kde_x(x_eval), label=x_label, linewidth=2, alpha=0.8)
            plt.fill_between(x_eval, kde_x(x_eval), alpha=0.3)
            plt.plot(x_eval, kde_y(x_eval), label=y_label, linewidth=2, alpha=0.8)
            plt.fill_between(x_eval, kde_y(x_eval), alpha=0.3)
            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.legend()

        except (np.linalg.LinAlgError, ValueError):
    
            # Fall back to histograms if KDE fails
            plt.hist(x_data, bins=20, alpha=0.6, label=x_label, edgecolor='black')
            plt.hist(y_data, bins=20, alpha=0.6, label=y_label, edgecolor='black')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.title('Input distributions (histogram)')
            plt.legend()
        
        plt.tight_layout()
    
    plt.show()


def plot_results(results, plot_type='scatter', metrics=None, exchange_interval=None):
    """Visualize hill climbing results with progress and snapshots.
    
    Creates a comprehensive visualization showing:
    - Progress plot with metrics and objective value over time
    - Snapshot plots at 25%, 50%, 75%, and 100% completion
    
    Args:
        results: Results from climb(). Can be:
                 - Tuple (best_data, steps_df) from single climb() call
                 - List of result tuples for multi-replica visualization
        plot_type: Type of snapshot plots, either ``'scatter'`` or ``'histogram'``
                   (default is ``'scatter'``).
                   Note: ``'histogram'`` uses KDE (Kernel Density Estimation) plots
        metrics: List of metric names to display in progress plots and snapshots.
                 If None (default), all available metrics are shown.
                 Example: ``['Pearson', 'Spearman']`` or ``['Mean X', 'Std X']``
        exchange_interval: Number of steps per batch (if provided, x-axis shows batches instead of steps)
    
    Raises:
        ValueError: If plot_type is not ``'scatter'`` or ``'histogram'``
        ValueError: If any specified metric is not found in the results
    """

    if plot_type not in ['scatter', 'histogram']:
        raise ValueError(f"plot_type must be 'scatter' or 'histogram', got '{plot_type}'")
    
    # Ensure results is a list
    if isinstance(results, tuple):
        # Single result tuple - could be (best_data, steps_df) or (temp_history, best_data, steps_df)
        results_list = [results]
    else:
        results_list = results
    
    # Get steps_df to validate metrics - handle both old and new formats
    if len(results_list[0]) == 3:
        _, _, steps_df = results_list[0]  # New format with temperature_history
    else:
        _, steps_df = results_list[0]  # Old format
    
    # Validate metrics if provided
    if metrics is not None:
        available_metrics = [col for col in steps_df.columns 
                            if col not in ['Step', 'Objective value', 'Best_data']]

        invalid_metrics = [m for m in metrics if m not in available_metrics]

        if invalid_metrics:
            raise ValueError(f"Metrics not found in results: {invalid_metrics}. "
                           f"Available metrics: {available_metrics}")
    
    if plot_type == 'scatter':
        _plot_results_scatter(results_list, metrics)

    else:
        _plot_results_histogram(results_list, metrics)


def _plot_results_scatter(results, metrics=None, exchange_interval=None):
    """Internal function: Visualize results with scatter plots.
    
    Args:
        results: List of result tuples - handles both (data, df) and (_, data, df) formats
        metrics: List of metric names to display, or None for all metrics
        exchange_interval: Number of steps per batch (if provided, x-axis shows batches instead of steps)
    """

    n_replicates = len(results)

    # Create 2-row layout: top row for progress plots, bottom row for data snapshots
    fig = plt.figure(figsize=(14, 5.5*n_replicates))
    spec = fig.add_gridspec(nrows=2*n_replicates, ncols=5, 
                           width_ratios=[1, 1, 1, 1, 1],
                           hspace=0.5, wspace=0.5, top=0.94)
    fig.suptitle('Hill climb results', fontsize=16)

    for i in range(n_replicates):
        # Handle both old and new formats
        if len(results[i]) == 3:
            temp_history, best_data, steps_df = results[i]

        else:
            best_data, steps_df = results[i]
            temp_history = []
        
        # Get metric columns
        all_metric_columns = [col for col in steps_df.columns 
                              if col not in ['Step', 'Objective value', 'Best_data']]
        
        # Use specified metrics or all available metrics
        metric_columns = metrics if metrics is not None else all_metric_columns
        
        # Calculate x-axis values (batch numbers if exchange_interval provided, else steps)
        if exchange_interval is not None:
            x_values = steps_df['Step'] / exchange_interval
            x_label = 'Batch'
        else:
            x_values = steps_df['Step']
            x_label = 'Step'
        
        # Progress plot - spans all columns in top row
        ax = fig.add_subplot(spec[2*i, :])
        ax.set_title(f'Replicate {i+1}: Progress', fontsize=10)
        
        lines = []

        for metric_name in metric_columns:

            lines.extend(
                ax.plot(
                    x_values, steps_df[metric_name], label=metric_name
                )
            )
        
        ax.set_xlabel(x_label)
        ax.set_ylabel('Metrics', color='black')
        ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
        
        ax2 = ax.twinx()
        lines.extend(ax2.plot(x_values, steps_df['Objective value'], 
                              label='Objective', color='black'))

        ax2.set_ylabel('Objective value', color='black')
        ax2.legend(lines, [l.get_label() for l in lines], loc='upper left', fontsize=7, edgecolor='black')
        ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
        
        ax2 = ax.twinx()
        lines.extend(ax2.plot(x_values, steps_df['Objective value'], 
                              label='Objective', color='black'))

        ax2.set_ylabel('Objective value', color='black')
        ax2.legend(lines, [l.get_label() for l in lines], loc='upper left', fontsize=7, edgecolor='black')
        
        # Add vertical lines for temperature exchanges
        if temp_history:
            y_min, y_max = ax.get_ylim()
            y_pos = y_max - 0.05 * (y_max - y_min)  # Position label near top
            x_min, x_max = ax.get_xlim()
            x_offset = 0.003 * (x_max - x_min)  # 0.3% offset to the right
            for step, new_temp in temp_history:
                x_pos = step / exchange_interval if exchange_interval else step
                ax.axvline(x=x_pos, color='black', linestyle=':', linewidth=1, alpha=0.7)
                ax.text(x_pos + x_offset, y_pos, f'T={new_temp:.2f}', fontsize=6, 
                       ha='left', va='top', rotation=0,
                       bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1))
        
        # Get initial data from step 0
        initial_data = steps_df['Best_data'].iloc[0]
        initial_step = steps_df['Step'].iloc[0]
        
        # Input data snapshot (bottom row, first column)
        ax = fig.add_subplot(spec[2*i+1, 0])
        if isinstance(initial_data, pd.DataFrame):
            snap_x, snap_y = initial_data.iloc[:, 0], initial_data.iloc[:, 1]
        else:
            snap_x, snap_y = initial_data[:, 0], initial_data[:, 1]
        
        if exchange_interval is not None:
            batch_num = initial_step / exchange_interval
            ax.set_title(f'Batch {batch_num:.2f}', fontsize=10)
        else:
            ax.set_title(f'Step {initial_step:.2e}', fontsize=10)
        ax.scatter(snap_x, snap_y, color='black', s=1)
        ax.set_xlabel('x', fontsize=8)
        ax.set_ylabel('y', fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.locator_params(axis='x', nbins=4)
        ax.locator_params(axis='y', nbins=4)
        
        # Add metrics annotation
        initial_metrics = steps_df.iloc[0]
        metric_text = '\n'.join([f'{col}: {initial_metrics[col]:.3f}' 
                                  for col in metric_columns])
        if metric_text:
            ax.text(0.02, 0.98, metric_text, transform=ax.transAxes,
                   fontsize=6, verticalalignment='top',
                   bbox=dict(facecolor='white', edgecolor='black', alpha=0.8, pad=2))
        
        # Snapshot plots at 25%, 50%, 75%, 100% (bottom row, columns 1-4)
        for j, pct in enumerate([0.25, 0.50, 0.75, 1.0]):
            
            ax = fig.add_subplot(spec[2*i+1, j+1])
            
            step_idx = max(0, min(int(len(steps_df) * pct) - 1, len(steps_df) - 1))
            snapshot_data = steps_df['Best_data'].iloc[step_idx]
            snapshot_step = steps_df['Step'].iloc[step_idx]
            
            # For scatter plots, we can only show 2D projections
            # Extract first two columns for visualization
            if isinstance(snapshot_data, pd.DataFrame):
                snap_x, snap_y = snapshot_data.iloc[:, 0], snapshot_data.iloc[:, 1]

            else:
                snap_x, snap_y = snapshot_data[:, 0], snapshot_data[:, 1]
            
            if exchange_interval is not None:
                batch_num = snapshot_step / exchange_interval
                ax.set_title(f'Batch {batch_num:.2f}', fontsize=10)
            else:
                ax.set_title(f'Step {snapshot_step:.2e}', fontsize=10)
            ax.scatter(snap_x, snap_y, color='black', s=1)
            ax.set_xlabel('x', fontsize=8)
            ax.set_ylabel('y', fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=7)
            ax.locator_params(axis='x', nbins=4)
            ax.locator_params(axis='y', nbins=4)
            
            # Build stats text
            obj_val = steps_df['Objective value'].iloc[step_idx]
            stats_text = f'Obj={obj_val:.4f}\n'

            for metric_name in metric_columns:

                abbrev = ''.join([word[0] for word in metric_name.split()])
                stats_text += f'{abbrev}={steps_df[metric_name].iloc[step_idx]:.3f}\n'
            
            ax.text(
                0.06, 0.94,
                stats_text.strip(),
                transform=ax.transAxes,
                fontsize=7,
                verticalalignment='top',
                bbox=dict(facecolor='white', edgecolor='black')
            )

    plt.tight_layout()
    plt.show()


def _plot_results_histogram(results, metrics=None, exchange_interval=None):
    """Internal function: Visualize results with histogram/KDE plots.
    
    Args:
        results: List of result tuples - handles both (data, df) and (_, data, df) formats
        metrics: List of metric names to display, or None for all metrics
        exchange_interval: Number of steps per batch (if provided, x-axis shows batches instead of steps)
    """

    n_replicates = len(results)

    # Create 2-row layout: top row for progress plots, bottom row for data snapshots
    fig = plt.figure(figsize=(14, 5.3*n_replicates))
    spec = fig.add_gridspec(nrows=2*n_replicates, ncols=5,
                           width_ratios=[1, 1, 1, 1, 1],
                           hspace=0.5, wspace=0.5, top=0.955)
    fig.suptitle('Hill climb results', fontsize=16)

    for i in range(n_replicates):

        # Handle both old and new formats
        if len(results[i]) == 3:
            temp_history, best_data, steps_df = results[i]

        else:
            best_data, steps_df = results[i]
            temp_history = []
        
        # Get metric columns
        all_metric_columns = [col for col in steps_df.columns 
                              if col not in ['Step', 'Objective value', 'Best_data']]
        
        # Use specified metrics or all available metrics
        metric_columns = metrics if metrics is not None else all_metric_columns
        
        # Calculate x-axis values (batch numbers if exchange_interval provided, else steps)
        if exchange_interval is not None:
            x_values = steps_df['Step'] / exchange_interval
            x_label = 'Batch'
        else:
            x_values = steps_df['Step']
            x_label = 'Step'
        
        # Progress plot - spans all columns in top row
        ax = fig.add_subplot(spec[2*i, :])
        ax.set_title(f'Replicate {i+1}')
        
        lines = []

        for metric_name in metric_columns:

            lines.extend(
                ax.plot(
                    x_values, steps_df[metric_name], label=metric_name
                )
            )
        
        ax.set_xlabel(x_label)
        ax.set_ylabel('Metrics')
        ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
        
        ax2 = ax.twinx()

        lines.extend(ax2.plot(x_values, steps_df['Objective value'], 
                              label='Objective', color='black'))

        ax2.set_ylabel('Objective value', color='black')
        ax2.legend(lines, [l.get_label() for l in lines], loc='upper left', fontsize=7, edgecolor='black')
        
        # Add vertical lines for temperature exchanges
        if temp_history:

            y_min, y_max = ax.get_ylim()
            y_pos = y_max - 0.05 * (y_max - y_min)  # Position label near top
            x_min, x_max = ax.get_xlim()
            x_offset = 0.003 * (x_max - x_min)  # 0.3% offset to the right

            for step, new_temp in temp_history:
                x_pos = step / exchange_interval if exchange_interval else step
                ax.axvline(x=x_pos, color='black', linestyle=':', linewidth=1, alpha=0.7)
                ax.text(
                    x_pos + x_offset, y_pos, 
                    f'T={new_temp:.2f}', fontsize=6, 
                    ha='left', va='top', rotation=0,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1)
                )
        
        # Get initial data from step 0
        initial_data = steps_df['Best_data'].iloc[0]
        initial_step = steps_df['Step'].iloc[0]
        
        # Input data KDE (bottom row, first column)
        ax = fig.add_subplot(spec[2*i+1, 0])
        
        # Extract all columns dynamically
        if isinstance(initial_data, pd.DataFrame):
            columns = initial_data.columns.tolist()
            column_data = {col: np.array(initial_data[col]) for col in columns}

        else:
            # For numpy arrays, generate column names
            n_cols = initial_data.shape[1] if len(initial_data.shape) > 1 else 1
            columns = [f'col_{k}' for k in range(n_cols)]
            column_data = {columns[k]: initial_data[:, k] for k in range(n_cols)}
        
        if exchange_interval is not None:
            batch_num = initial_step / exchange_interval
            ax.set_title(f'Batch {batch_num:.2f}', fontsize=10)
        else:
            ax.set_title(f'Step {initial_step:.2e}')
        
        # Use matplotlib's default color cycle
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        # Create KDE plots for all columns
        try:
            # Get global min/max across all columns for consistent x-axis
            all_data = np.concatenate([column_data[col] for col in columns])
            x_min, x_max = all_data.min(), all_data.max()
            x_eval = np.linspace(x_min, x_max, 200)
            
            # Plot KDE for each column
            for k, col in enumerate(columns):
                col_data = column_data[col]
                color = colors[k % len(colors)]
                
                kde = gaussian_kde(col_data)
                density = kde(x_eval)
                
                ax.plot(x_eval, density, color=color, linewidth=2, alpha=0.8)
                ax.fill_between(x_eval, density, alpha=0.2, color=color)
            
            ax.set_xlabel('Value', fontsize=8)
            ax.set_ylabel('Density', fontsize=8)
            # ax.tick_params(axis='both', which='major', labelsize=7)
            # ax.locator_params(axis='x', nbins=4)
            # ax.locator_params(axis='y', nbins=4)
            
        except (np.linalg.LinAlgError, ValueError):
            # If KDE fails, fall back to histogram
            for k, col in enumerate(columns):
                col_data = column_data[col]
                color = colors[k % len(colors)]
                ax.hist(col_data, bins=20, alpha=0.5, color=color, edgecolor='black')
            
            ax.set_xlabel('Value', fontsize=8)
            ax.set_ylabel('Frequency', fontsize=8)
            # ax.tick_params(axis='both', which='major', labelsize=7)
        
        # Add metrics annotation
        initial_metrics = steps_df.iloc[0]
        metric_text = '\n'.join([f'{col}: {initial_metrics[col]:.3f}' 
                                  for col in metric_columns])
        if metric_text:
            ax.text(0.02, 0.98, metric_text, transform=ax.transAxes,
                   fontsize=6, verticalalignment='top',
                   bbox=dict(facecolor='white', edgecolor='black', alpha=0.8, pad=2))
            # ax.locator_params(axis='x', nbins=4)
            # ax.locator_params(axis='y', nbins=4)
        
        # Snapshot histograms at 25%, 50%, 75%, 100% (bottom row, columns 1-4)
        for j, pct in enumerate([0.25, 0.50, 0.75, 1.0]):
            ax = fig.add_subplot(spec[2*i+1, j+1])
            
            step_idx = max(0, min(int(len(steps_df) * pct) - 1, len(steps_df) - 1))
            snapshot_data = steps_df['Best_data'].iloc[step_idx]
            snapshot_step = steps_df['Step'].iloc[step_idx]
            
            # Extract all columns dynamically
            if isinstance(snapshot_data, pd.DataFrame):
                columns = snapshot_data.columns.tolist()
                column_data = {col: np.array(snapshot_data[col]) for col in columns}

            else:
                # For numpy arrays, generate column names
                n_cols = snapshot_data.shape[1] if len(snapshot_data.shape) > 1 else 1
                columns = [f'col_{k}' for k in range(n_cols)]
                column_data = {columns[k]: snapshot_data[:, k] for k in range(n_cols)}
            
            if exchange_interval is not None:
                batch_num = snapshot_step / exchange_interval
                ax.set_title(f'Batch {batch_num:.2f}', fontsize=10)
            else:
                ax.set_title(f'Step {snapshot_step:.2e}')
            
            # Use matplotlib's default color cycle
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            
            # Create KDE plots for all columns
            try:

                # Get global min/max across all columns for consistent x-axis
                all_data = np.concatenate([column_data[col] for col in columns])
                x_min, x_max = all_data.min(), all_data.max()
                x_eval = np.linspace(x_min, x_max, 200)
                
                # Plot KDE for each column
                for k, col in enumerate(columns):

                    col_data = column_data[col]
                    color = colors[k % len(colors)]
                    
                    kde = gaussian_kde(col_data)
                    density = kde(x_eval)
                    
                    ax.plot(
                        x_eval,
                        density,
                        label=col,
                        color=color,
                        linewidth=2,
                        alpha=0.8
                    )

                    ax.fill_between(x_eval, density, alpha=0.2, color=color)
                
                ax.set_xlabel('Value', fontsize=8)
                ax.set_ylabel('Density', fontsize=8)
                # ax.tick_params(axis='both', which='major', labelsize=7)
                # ax.locator_params(axis='x', nbins=4)
                # ax.locator_params(axis='y', nbins=4)
                
            except (np.linalg.LinAlgError, ValueError) as e:

                # If KDE fails (e.g., all values identical), fall back to histogram
                for k, col in enumerate(columns):

                    col_data = column_data[col]
                    color = colors[k % len(colors)]
                    ax.hist(
                        col_data,
                        bins=20,
                        alpha=0.5,
                        label=col,
                        color=color,
                        edgecolor='black'
                    )
                
                ax.set_xlabel('Value', fontsize=8)
                ax.set_ylabel('Frequency', fontsize=8)
                # ax.tick_params(axis='both', which='major', labelsize=7)
                # ax.locator_params(axis='x', nbins=4)
                # ax.locator_params(axis='y', nbins=4)
            
            # Build stats text
            obj_val = steps_df['Objective value'].iloc[step_idx]
            stats_text = f'Obj={obj_val:.4f}\n'

            for metric_name in metric_columns:

                abbrev = ''.join([word[0] for word in metric_name.split()])
                stats_text += f'{abbrev}={steps_df[metric_name].iloc[step_idx]:.3f}\n'
            
            ax.text(
                0.05, 0.94,
                stats_text.strip(),
                transform=ax.transAxes,
                fontsize=7,
                verticalalignment='top',
                bbox=dict(facecolor='white', edgecolor='black')
            )

    plt.show()

