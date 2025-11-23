"""Streamlit dashboard for monitoring hill climber optimization progress in real-time."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from pathlib import Path
import sqlite3
import json
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Hill Climber Progress Monitor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("Hill Climber Progress Monitor")

# Sidebar configuration
st.sidebar.header("Configuration")

# Database file selection
default_db = "data/hill_climber_progress.db"
db_path = st.sidebar.text_input("Database Path", value=default_db)

# Auto-refresh settings
auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", min_value=1, max_value=30, value=15)

# Manual refresh button
if st.sidebar.button("Refresh Now"):
    st.rerun()

# Check if database exists
if not Path(db_path).exists():
    st.error(f"Database file not found: {db_path}")
    st.info("Make sure your HillClimber instance has `db_enabled=True` and is currently running.")
    st.stop()

# Database connection
@st.cache_resource
def get_connection(db_path_str):
    """Create a read-only database connection."""
    return sqlite3.connect(f"file:{db_path_str}?mode=ro", uri=True, check_same_thread=False)

try:
    conn = get_connection(db_path)
except Exception as e:
    st.error(f"Failed to connect to database: {e}")
    st.stop()

# Load run metadata
@st.cache_data(ttl=60)
def load_run_metadata(_conn):
    """Load run metadata from database."""
    query = "SELECT * FROM run_metadata WHERE run_id = 1"
    cursor = _conn.cursor()
    cursor.execute(query)
    row = cursor.fetchone()
    
    if row:
        return {
            'run_id': row[0],
            'start_time': row[1],
            'n_replicas': row[2],
            'exchange_interval': row[3],
            'db_step_interval': row[4],
            'db_buffer_size': row[5],
            'hyperparameters': json.loads(row[6])
        }
    return None

# Load replica status
def load_replica_status(conn):
    """Load current replica status."""
    query = """
        SELECT replica_id, step, temperature, best_objective, 
               current_objective, timestamp
        FROM replica_status
        ORDER BY replica_id
    """
    return pd.read_sql_query(query, conn)

# Load metrics history
def load_metrics_history(conn, replica_id=None, metric_names=None):
    """Load metrics history."""
    query = "SELECT replica_id, step, metric_name, value FROM metrics_history WHERE 1=1"
    params = []
    
    if replica_id is not None:
        query += " AND replica_id = ?"
        params.append(replica_id)
    
    if metric_names:
        placeholders = ','.join(['?' for _ in metric_names])
        query += f" AND metric_name IN ({placeholders})"
        params.extend(metric_names)
    
    query += " ORDER BY replica_id, step, metric_name"
    
    return pd.read_sql_query(query, conn, params=params)

# Load temperature exchanges
def load_temperature_exchanges(conn):
    """Load temperature exchange history."""
    query = """
        SELECT step, replica_id, new_temperature, timestamp
        FROM temperature_exchanges
        ORDER BY step
    """
    return pd.read_sql_query(query, conn)

# Get available metrics
def get_available_metrics(conn):
    """Get list of all metric names in database."""
    query = "SELECT DISTINCT metric_name FROM metrics_history ORDER BY metric_name"
    cursor = conn.cursor()
    cursor.execute(query)
    return [row[0] for row in cursor.fetchall()]

# Load data
try:
    metadata = load_run_metadata(conn)
    
    if metadata is None:
        st.warning("No run metadata found. Waiting for optimization to start...")
        time.sleep(refresh_interval if auto_refresh else 10)
        if auto_refresh:
            st.rerun()
        st.stop()
    
    # Get available metrics
    available_metrics = get_available_metrics(conn)
    
    # Metric selection with multiselect in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Metrics to Display")
    st.sidebar.caption("Objective value is always shown")
    
    # Filter out 'Objective value' from additional metrics
    other_metrics = [m for m in available_metrics if m != 'Objective value']
    
    if other_metrics:
        additional_metrics = st.sidebar.multiselect(
            "Additional Metrics",
            options=other_metrics,
            default=[other_metrics[0]] if len(other_metrics) > 0 else []
        )
    else:
        additional_metrics = []
    
    # Build metrics list (always include Objective value)
    metrics_to_plot = ['Objective value'] + additional_metrics
    
    # Normalization option
    normalize_metrics = st.sidebar.checkbox(
        "Normalize metrics to [0, 1] range",
        value=False,
        help="Scale all metrics to their min-max range for easier comparison on same scale"
    )
    
    # Display run information
    st.sidebar.markdown("---")
    st.sidebar.subheader("Run Information")
    st.sidebar.text(f"Started: {datetime.fromtimestamp(metadata['start_time']).strftime('%Y-%m-%d %H:%M:%S')}")
    st.sidebar.text(f"Replicas: {metadata['n_replicas']}")
    st.sidebar.text(f"Exchange Interval: {metadata['exchange_interval']}")
    st.sidebar.text(f"DB Step Interval: {metadata['db_step_interval']}")
    st.sidebar.text(f"DB Buffer Size: {metadata['db_buffer_size']}")
    
    # Load current status
    status_df = load_replica_status(conn)
    
    if status_df.empty:
        st.info("Waiting for first batch to complete...")
        time.sleep(refresh_interval if auto_refresh else 10)
        if auto_refresh:
            st.rerun()
        st.stop()
    
    # Display replica status cards
    st.header("Replica Status")
    
    cols = st.columns(metadata['n_replicas'])
    for idx, row in status_df.iterrows():
        with cols[idx]:
            elapsed = time.time() - row['timestamp']
            freshness = "🟢" if elapsed < 30 else "🟡" if elapsed < 120 else "🔴"
            
            st.metric(
                label=f"Replica {row['replica_id']} {freshness}",
                value=f"Step {int(row['step']):,}",
                delta=f"T={row['temperature']:.2f}"
            )
            st.caption(f"Best: {row['best_objective']:.4f}")
            st.caption(f"Current: {row['current_objective']:.4f}")
    
    # Load metrics history
    metrics_df = load_metrics_history(conn, metric_names=metrics_to_plot)
    exchanges_df = load_temperature_exchanges(conn)
    
    if not metrics_df.empty:
        # Create progress plots - one subplot per replica
        st.header("Optimization Progress")
        
        # Get list of replicas
        replica_ids = sorted(metrics_df['replica_id'].unique())
        n_replicas = len(replica_ids)
        
        # Create a 2-column layout for plots
        n_cols = 2
        
        for idx, replica_id in enumerate(replica_ids):
            # Create new row every 2 replicas
            if idx % n_cols == 0:
                cols = st.columns(n_cols)
            
            with cols[idx % n_cols]:
                # Create individual figure with secondary y-axis
                from plotly.subplots import make_subplots
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Get objective value data
                obj_data = metrics_df[
                    (metrics_df['replica_id'] == replica_id) & 
                    (metrics_df['metric_name'] == 'Objective value')
                ]
                
                has_objective = not obj_data.empty
                
                if has_objective:
                    # Convert step to batch number
                    batch_numbers = obj_data['step'] / metadata['exchange_interval']
                    
                    # Normalize if requested
                    obj_values = obj_data['value'].values
                    if normalize_metrics:
                        obj_min, obj_max = obj_values.min(), obj_values.max()
                        if obj_max > obj_min:
                            obj_values_norm = (obj_values - obj_min) / (obj_max - obj_min)
                        else:
                            obj_values_norm = obj_values * 0  # All same value -> 0
                        y_values = obj_values_norm
                        hover_template = 'Batch: %{x:.2f}<br>Objective (norm): %{y:.4f}<br>Actual: %{customdata:.4f}<extra></extra>'
                        customdata = obj_values
                    else:
                        y_values = obj_values
                        hover_template = 'Batch: %{x:.2f}<br>Objective: %{y:.4f}<extra></extra>'
                        customdata = None
                    
                    # Add objective value trace
                    fig.add_trace(go.Scatter(
                        x=batch_numbers,
                        y=y_values,
                        mode='lines',
                        name='Objective',
                        line=dict(color='#2E86AB', width=3),  # Darker blue, thicker line
                        hovertemplate=hover_template,
                        customdata=customdata
                    ), secondary_y=False)
                else:
                    st.warning(f"No Objective value data for Replica {replica_id}")
                
                # Add additional metrics if selected
                colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                for i, additional_metric in enumerate(additional_metrics):
                    add_data = metrics_df[
                        (metrics_df['replica_id'] == replica_id) & 
                        (metrics_df['metric_name'] == additional_metric)
                    ]
                    
                    if not add_data.empty:
                        batch_numbers_add = add_data['step'] / metadata['exchange_interval']
                        color = colors[i % len(colors)]
                        
                        # Normalize if requested
                        add_values = add_data['value'].values
                        if normalize_metrics:
                            add_min, add_max = add_values.min(), add_values.max()
                            if add_max > add_min:
                                add_values_norm = (add_values - add_min) / (add_max - add_min)
                            else:
                                add_values_norm = add_values * 0  # All same value -> 0
                            y_values_add = add_values_norm
                            hover_template_add = f'Batch: %{{x:.2f}}<br>{additional_metric} (norm): %{{y:.4f}}<br>Actual: %{{customdata:.4f}}<extra></extra>'
                            customdata_add = add_values
                        else:
                            y_values_add = add_values
                            hover_template_add = f'Batch: %{{x:.2f}}<br>{additional_metric}: %{{y:.4f}}<extra></extra>'
                            customdata_add = None
                        
                        fig.add_trace(go.Scatter(
                            x=batch_numbers_add,
                            y=y_values_add,
                            mode='lines',
                            name=additional_metric,
                            line=dict(color=color, width=2, dash='dot'),
                            hovertemplate=hover_template_add,
                            customdata=customdata_add
                        ), secondary_y=not normalize_metrics)
                
                # Add temperature exchange markers
                if not exchanges_df.empty:
                    replica_exchanges = exchanges_df[exchanges_df['replica_id'] == replica_id]
                    
                    for _, exchange in replica_exchanges.iterrows():
                        exchange_batch = exchange['step'] / metadata['exchange_interval']
                        
                        # Add vertical line at exchange point
                        fig.add_vline(
                            x=exchange_batch,
                            line_dash="dash",
                            line_color="gray",
                            line_width=1,
                            opacity=0.5
                        )
                        
                        # Add annotation for new temperature
                        fig.add_annotation(
                            x=exchange_batch,
                            y=1.0,
                            yref='y domain',
                            text=f'T={exchange["new_temperature"]:.1f}',
                            showarrow=False,
                            textangle=0,
                            font=dict(size=10, color='gray'),
                            xanchor='left',
                            yanchor='top'
                        )
                
                # Update layout
                fig.update_layout(
                    title_text=f'Replica {replica_id}',
                    height=350,
                    hovermode='closest',
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                # Update axes
                fig.update_xaxes(title_text="Batch")
                if normalize_metrics:
                    fig.update_yaxes(title_text="Normalized Value [0, 1]", secondary_y=False)
                else:
                    fig.update_yaxes(title_text="Objective value", secondary_y=False)
                    if additional_metrics:
                        # Use first metric name or generic label for multiple metrics
                        secondary_label = additional_metrics[0] if len(additional_metrics) == 1 else "Additional Metrics"
                        fig.update_yaxes(title_text=secondary_label, secondary_y=True)
                
                st.plotly_chart(fig, width='stretch')
    else:
        st.info("No metrics data available yet. Waiting for workers to collect data...")
    
    # Temperature exchange timeline
    st.header("Temperature Exchange Timeline")
    
    if not exchanges_df.empty:
        fig = go.Figure()
        
        for replica_id in sorted(exchanges_df['replica_id'].unique()):
            replica_exchanges = exchanges_df[exchanges_df['replica_id'] == replica_id]
            
            # Convert step to batch number
            batch_numbers = replica_exchanges['step'] / metadata['exchange_interval']
            
            fig.add_trace(go.Scatter(
                x=batch_numbers,
                y=replica_exchanges['new_temperature'],
                mode='markers+lines',
                name=f'Replica {replica_id}',
                hovertemplate=f'<b>Replica {replica_id}</b><br>' +
                             'Batch: %{x:.2f}<br>' +
                             'Temperature: %{y:.2f}<extra></extra>'
            ))
        
        fig.update_layout(
            title='Temperature Exchanges Over Time',
            xaxis_title='Batch',
            yaxis_title='Temperature',
            yaxis_type='log',
            hovermode='closest',
            height=400
        )
        
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("No temperature exchanges recorded yet.")
    
    # Statistics
    st.header("Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        best_replica = status_df.loc[status_df['best_objective'].idxmax()]
        st.metric(
            "Best Replica",
            f"Replica {best_replica['replica_id']}",
            f"{best_replica['best_objective']:.4f}"
        )
    
    with col2:
        total_exchanges = len(exchanges_df) if not exchanges_df.empty else 0
        st.metric("Total Temperature Exchanges", f"{total_exchanges:,}")
    
    with col3:
        max_step = status_df['step'].max()
        st.metric("Max Steps", f"{int(max_step):,}")
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

except Exception as e:
    st.error(f"Error loading data: {e}")
    import traceback
    st.code(traceback.format_exc())
    
    if auto_refresh:
        st.info(f"Retrying in {refresh_interval} seconds...")
        time.sleep(refresh_interval)
        st.rerun()
