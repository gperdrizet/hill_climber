"""Plot generation functions for the hill climber dashboard.

This module creates all Plotly charts used in the dashboard,
separating visualization logic from data and UI concerns.
"""

from typing import List, Dict, Optional
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
        metrics_df: DataFrame with columns: replica_id, step, metric_name, value
        replica_id: ID of the replica to plot
        objective_metric: Name of the objective metric to plot
        additional_metrics: List of additional metric names to plot
        exchange_interval: Steps between exchange attempts
        replica_temps: Dictionary mapping replica_id to current temperature
        exchanges_df: DataFrame with temperature exchange events
        normalize_metrics: If True, normalize all metrics to [0, 1]
        show_exchanges: If True, draw vertical lines at exchange events
        
    Returns:
        Plotly Figure object
    """
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
        batch_numbers = obj_data['step'] / exchange_interval
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
                name=objective_metric, line=dict(color='#2E86AB', width=3),
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
            batch_numbers_add = add_data['step'] / exchange_interval
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
        for _, exchange in replica_exchanges.iterrows():
            exchange_batch = exchange['step'] / exchange_interval
            fig.add_vline(x=exchange_batch, line_dash="dash", line_color="#555", line_width=1)
    
    # Configure axes
    fig.update_xaxes(title_text="Batch")
    fig.update_yaxes(title_text="Objective", secondary_y=False)
    if additional_metrics and not normalize_metrics:
        fig.update_yaxes(title_text="Metric", secondary_y=True)
    
    # Layout
    replica_temp = replica_temps.get(replica_id)
    title = f"Replica {int(replica_id)}"
    if replica_temp is not None:
        title += f" (T={replica_temp:.4f})"
    
    fig.update_layout(
        title_text=title,
        title_font_size=20,
        legend=dict(orientation='h', yanchor='bottom', y=-0.2),
        margin=dict(l=40, r=20, t=40, b=40),
        height=400
    )
    
    return fig
