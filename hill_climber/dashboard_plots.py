"""Plot generation functions for the hill climber dashboard.

This module creates all Plotly charts used in the dashboard,
separating visualization logic from data and UI concerns.
"""

from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_temperature_ladder_plot(
    temp_ladder_history_df: pd.DataFrame,
    temp_ladder_df: pd.DataFrame
) -> go.Figure:
    """Create a plot showing temperature ladder evolution over time.
    
    Each trace represents a fixed ladder position (rank), showing how the
    temperature at that position decreases due to cooling over time.
    
    Args:
        temp_ladder_history_df (pd.DataFrame): Temperature ladder history with columns:
            batch_num, ladder_position, temperature.
        temp_ladder_df (pd.DataFrame): Initial temperature ladder (unused, kept for compatibility).
        
    Returns:
        go.Figure: Plotly Figure showing temperature evolution for each ladder step.
    """
    fig = go.Figure()
    
    if temp_ladder_history_df.empty:
        # Return empty figure with message
        fig.add_annotation(
            text="No temperature ladder data available yet",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            height=400,
            title_text="Temperature ladder",
            title_font_size=20,
            margin=dict(l=40, r=20, t=90, b=40)
        )
        return fig
    
    # Determine number of ladder positions
    n_replicas = temp_ladder_history_df['ladder_position'].nunique()
    
    # Generate red color gradient from dark to light
    red_colors = []
    for i in range(n_replicas):
        intensity = 1.0 - (0.7 * i / max(1, n_replicas - 1))  # 1.0 to 0.3
        red = int(255 * intensity)
        green = int(50 * (1 - intensity))  # Slight warmth
        blue = int(50 * (1 - intensity))
        red_colors.append(f'rgb({red},{green},{blue})')
    
    # Plot each ladder position
    for position in range(n_replicas):
        position_data = temp_ladder_history_df[
            temp_ladder_history_df['ladder_position'] == position
        ].sort_values('batch_num')
        
        if not position_data.empty:
            batches = position_data['batch_num']
            temps = position_data['temperature']
            
            fig.add_trace(
                go.Scatter(
                    x=batches,
                    y=temps,
                    mode='lines',
                    line=dict(color=red_colors[position], width=2),
                    hovertemplate=f'Ladder position {position}<br>Batch: %{{x:.0f}}<br>Temperature: %{{y:.2e}}<extra></extra>',
                    showlegend=False
                )
            )
    
    # Layout
    fig.update_layout(
        title_text="Temperature ladder",
        title_font_size=20,
        xaxis_title="Batch",
        yaxis=dict(
            title="Temperature",
            type="log",
            exponentformat="power"
        ),
        height=400,
        margin=dict(l=60, r=20, t=90, b=40),
        showlegend=False
    )
    
    return fig


def create_batch_statistics_plot(batch_stats_df: pd.DataFrame) -> go.Figure:
    """Create a plot showing step spread and acceptance rates over time.
    
    Shows two traces: step spread and mean acceptance rate with min-max range.
    
    Args:
        batch_stats_df (pd.DataFrame): Batch statistics with columns:
            batch_num, step_spread, mean_acceptance_rate, min_acceptance_rate, max_acceptance_rate.
        
    Returns:
        go.Figure: Plotly Figure with dual y-axes showing step spread and acceptance rates.
    """
    fig = go.Figure()
    
    if batch_stats_df.empty:
        # Return empty figure with message
        fig.add_annotation(
            text="No batch statistics available yet",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            height=400,
            title_text="Step performance",
            title_font_size=20,
            margin=dict(l=40, r=40, t=90, b=40)
        )
        return fig
    
    # Clean data - replace NaN/inf with 0
    batch_stats_df = batch_stats_df.fillna(0)
    batch_stats_df = batch_stats_df.replace([np.inf, -np.inf], 0)
    
    # Convert to Python native types for JSON serialization
    batches = [int(x) for x in batch_stats_df['batch_num']]
    step_spread = [float(x) for x in batch_stats_df['step_spread']]
    mean_accept = [float(x) for x in batch_stats_df['mean_acceptance_rate']]
    min_accept = [float(x) for x in batch_stats_df['min_acceptance_rate']]
    max_accept = [float(x) for x in batch_stats_df['max_acceptance_rate']]
    
    # Add step spread trace (left y-axis)
    fig.add_trace(
        go.Scatter(
            x=batches,
            y=step_spread,
            mode='lines',
            name='Step spread',
            line=dict(color='blue', width=2),
            hovertemplate='Batch: %{x:.0f}<br>Step spread: %{y:.1%}<extra></extra>',
            yaxis='y'
        )
    )
    
    # Add acceptance rate traces (right y-axis)
    # Mean line
    fig.add_trace(
        go.Scatter(
            x=batches,
            y=mean_accept,
            mode='lines',
            name='Mean acceptance',
            line=dict(color='green', width=2),
            hovertemplate='Batch: %{x:.0f}<br>Acceptance: %{y:.1%}<extra></extra>',
            yaxis='y2'
        )
    )
    
    # Calculate bounds with clipping to valid range [0, 1]
    upper_bound = [float(np.clip(x, 0, 1)) for x in max_accept]
    lower_bound = [float(np.clip(x, 0, 1)) for x in min_accept]
    
    # Upper bound (max)
    fig.add_trace(
        go.Scatter(
            x=batches,
            y=upper_bound,
            mode='lines',
            name='Min-max range',
            line=dict(color='lightgreen', width=0),
            showlegend=False,
            hoverinfo='skip',
            yaxis='y2'
        )
    )
    
    # Lower bound (min) with fill
    fig.add_trace(
        go.Scatter(
            x=batches,
            y=lower_bound,
            mode='lines',
            name='Min-max range',
            line=dict(color='lightgreen', width=0),
            fill='tonexty',
            fillcolor='rgba(144, 238, 144, 0.2)',
            showlegend=True,
            hoverinfo='skip',
            yaxis='y2'
        )
    )
    
    # Update layout with dual y-axes
    fig.update_layout(
        xaxis=dict(title="Batch"),
        yaxis=dict(
            title="Step spread",
            side='left',
            tickformat='.1%'
        ),
        yaxis2=dict(
            title="Acceptance rate",
            side='right',
            overlaying='y',
            tickformat='.1%'
        )
    )
    
    fig.update_layout(
        title_text="Step performance",
        title_font_size=20,
        height=400,
        margin=dict(l=60, r=60, t=90, b=40),
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def create_replica_plot(
    metrics_df: pd.DataFrame,
    replica_id: int,
    objective_metric: str,
    additional_metrics: List[str],
    exchange_interval: int,
    replica_temps: Dict[int, float],
    exchanges_df: pd.DataFrame,
    normalize_metrics: bool = False,
    show_exchanges: bool = False
) -> go.Figure:
    """Create a plot for a single replica showing objective and additional metrics.
    
    Args:
        metrics_df (pd.DataFrame): DataFrame with columns: replica_id, perturbation_num, metric_name, value.
        replica_id (int): ID of the replica to plot.
        objective_metric (str): Name of the objective metric to plot on primary y-axis.
        additional_metrics (List[str]): List of additional metric names to plot.
        exchange_interval (int): Steps between exchange attempts (for x-axis scaling).
        replica_temps (Dict[int, float]): Dictionary mapping replica_id to current temperature.
        exchanges_df (pd.DataFrame): DataFrame with temperature exchange events.
        normalize_metrics (bool): If True, normalize all metrics to [0, 1]. Default is False.
        show_exchanges (bool): If True, draw vertical lines at exchange events. Default is False.
        
    Returns:
        go.Figure: Plotly Figure object with replica metrics plot.
    """
    # Calculate total number of legend entries
    total_metrics = 1 + len(additional_metrics)  # objective + additional
    if show_exchanges:
        total_metrics += 1  # add exchange marker
    
    # Base height + additional height per metric (beyond 1)
    base_height = 350
    height_per_metric = 35
    plot_height = base_height + (total_metrics - 1) * height_per_metric
    
    # Calculate top margin based on number of metrics
    base_top_margin = 90
    margin_per_metric = 20
    top_margin = base_top_margin + (total_metrics - 1) * margin_per_metric
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Color palette
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', 
              '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Plot objective metric
    obj_data = metrics_df[
        (metrics_df['replica_id'] == replica_id) & 
        (metrics_df['metric_name'] == objective_metric)
    ]
    
    if not obj_data.empty:
        batch_numbers = obj_data['perturbation_num'] / exchange_interval
        obj_values = obj_data['value'].values
        
        # Normalize if requested
        if normalize_metrics:
            obj_min, obj_max = obj_values.min(), obj_values.max()
            y_values = (obj_values - obj_min) / (obj_max - obj_min) if obj_max > obj_min else obj_values * 0
            hover_template = f'Batch: %{{x:.2f}}<br>{objective_metric} (norm): %{{y:.4f}}<br>Actual: %{{customdata:.4f}}<extra></extra>'
            customdata = obj_values
        else:
            y_values = obj_values
            hover_template = f'Batch: %{{x:.2f}}<br>{objective_metric}: %{{y:.4f}}<extra></extra>'
            customdata = None
        
        fig.add_trace(
            go.Scatter(
                x=batch_numbers, y=y_values, mode='lines', 
                name='Objective', line=dict(color='#2E86AB', width=3),
                hovertemplate=hover_template, customdata=customdata
            ),
            secondary_y=False
        )
    
    # Plot additional metrics
    for i, metric_name in enumerate(additional_metrics):
        add_data = metrics_df[
            (metrics_df['replica_id'] == replica_id) & 
            (metrics_df['metric_name'] == metric_name)
        ]
        
        if not add_data.empty:
            batch_numbers_add = add_data['perturbation_num'] / exchange_interval
            add_values = add_data['value'].values
            color = colors[i % len(colors)]
            
            # Normalize if requested
            if normalize_metrics:
                add_min, add_max = add_values.min(), add_values.max()
                y_values_add = (add_values - add_min) / (add_max - add_min) if add_max > add_min else add_values * 0
                hover_template_add = f'Batch: %{{x:.2f}}<br>{metric_name} (norm): %{{y:.4f}}<br>Actual: %{{customdata:.4f}}<extra></extra>'
                customdata_add = add_values
            else:
                y_values_add = add_values
                hover_template_add = f'Batch: %{{x:.2f}}<br>{metric_name}: %{{y:.4f}}<extra></extra>'
                customdata_add = None
            
            fig.add_trace(
                go.Scatter(
                    x=batch_numbers_add, y=y_values_add, mode='lines',
                    name=metric_name, line=dict(color=color, width=2, dash='dot'),
                    hovertemplate=hover_template_add, customdata=customdata_add
                ),
                secondary_y=not normalize_metrics
            )
    
    # Add temperature exchange markers if enabled
    if show_exchanges and not exchanges_df.empty:
        replica_exchanges = exchanges_df[exchanges_df['replica_id'] == replica_id]
        # Draw vertical lines for each exchange
        for _, exchange in replica_exchanges.iterrows():
            exchange_batch = exchange['perturbation_num'] / exchange_interval
            fig.add_vline(
                x=exchange_batch, 
                line_dash="dash", 
                line_color="#555", 
                line_width=1
            )
        
        # Add a dummy trace for the legend entry (vlines don't show in legend)
        if len(replica_exchanges) > 0:
            fig.add_trace(
                go.Scatter(
                    x=[None], y=[None],
                    mode='lines',
                    name='Exchange',
                    line=dict(color='#555', width=1, dash='dash'),
                    showlegend=True
                ),
                secondary_y=False
            )
    
    # Configure axes
    fig.update_xaxes(title_text="Batch")
    fig.update_yaxes(title_text="Objective", secondary_y=False)
    if additional_metrics and not normalize_metrics:
        fig.update_yaxes(title_text="Metric", secondary_y=True)
    
    # Layout
    replica_temp = replica_temps.get(replica_id)
    title = f"Replica {int(replica_id)}"
    if replica_temp is not None:
        title += f" (T={replica_temp:.1e})"
    
    fig.update_layout(
        title_text=title,
        title_font_size=20,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        margin=dict(l=40, r=20, t=top_margin, b=40),
        height=plot_height
    )
    
    return fig
