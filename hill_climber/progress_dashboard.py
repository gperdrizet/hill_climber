"""Streamlit dashboard for monitoring hill climber optimization progress in real-time.

This module can be launched via the console script `hill-climber-dashboard`
once the package is installed, or directly in development using:

    python -m hill_climber.progress_dashboard

It requires `streamlit`, `plotly`, and `pandas` to be installed.
"""

import sys
import os
import time
from pathlib import Path


def _init_session_state(st):
    """Initialize session state variables."""
    if 'db_user_selected' not in st.session_state:
        st.session_state.db_user_selected = False
    
    if 'db_path' not in st.session_state:
        # Try common default locations
        default_candidates = [
            "data/hill_climber_progress.db",
            "../data/hill_climber_progress.db",
            "hill_climber_progress.db"
        ]
        for candidate in default_candidates:
            if Path(candidate).exists():
                st.session_state.db_path = candidate
                return
        st.session_state.db_path = "data/hill_climber_progress.db"


def render():
    """Render the Streamlit dashboard."""
    # Import modular dashboard components (use absolute imports for Streamlit compatibility)
    from hill_climber.dashboard_data import (
        get_connection,
        load_run_metadata,
        load_metrics_history,
        load_temperature_exchanges,
        get_available_metrics,
        get_available_directories,
        load_leaderboard,
        load_replica_temperatures,
        load_temperature_ladder,
        load_progress_stats
    )
    from hill_climber.dashboard_ui import (
        apply_custom_css,
        render_sidebar_title,
        render_database_selector,
        render_auto_refresh_controls,
        render_plot_options,
        render_run_information,
        render_hyperparameters,
        render_temperature_ladder,
        render_leaderboard,
        render_progress_stats
    )
    from hill_climber.dashboard_plots import create_replica_plot
    
    try:
        import streamlit as st
        import pandas as pd
    except ImportError as e:
        print(f"Missing dependency: {e}")
        sys.exit(1)

    # Page config
    st.set_page_config(
        page_title="Hill climber progress monitor",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Apply custom styling
    apply_custom_css()
    render_sidebar_title()

    # Initialize session state
    _init_session_state(st)
    
    # Sidebar: Database selection
    dirs = get_available_directories()
    db_path = render_database_selector(st.session_state, dirs)
    
    # Sidebar: Auto-refresh controls
    auto_refresh, refresh_interval = render_auto_refresh_controls()

    # Check database and connect
    if not Path(db_path).exists():
        st.info("Select a database in the sidebar to view progress.")
        return

    try:
        conn = get_connection(db_path)
    except Exception as e:
        st.error(f"Failed to connect to database: {e}")
        st.stop()

    metadata = load_run_metadata(conn)
    if metadata is None:
        st.warning("No run metadata found. Waiting for optimization to start...")
        time.sleep(refresh_interval if auto_refresh else 10)
        if auto_refresh:
            st.rerun()
        st.stop()

    available_metrics = get_available_metrics(conn)

    # Sidebar: Plot options
    plot_config = render_plot_options(available_metrics)
    
    # Sidebar: Run information
    render_run_information(metadata)
    render_hyperparameters(metadata)
    
    # Sidebar: Temperature ladder
    temp_ladder_df = load_temperature_ladder(conn)
    render_temperature_ladder(temp_ladder_df)

    # Load data
    metrics_df = load_metrics_history(
        conn,
        metric_names=[plot_config['objective_metric']] + plot_config['additional_metrics'],
        max_points_per_replica=plot_config['max_points']
    )
    exchanges_df = load_temperature_exchanges(conn)

    if metrics_df.empty:
        st.info("No metrics found yet. Waiting for data...")
        return

    # Verify objective metric exists
    loaded_metrics = metrics_df['metric_name'].unique().tolist()
    objective_metric = plot_config['objective_metric']
    
    if objective_metric not in loaded_metrics:
        st.warning(f"'{objective_metric}' not found. Available: {', '.join(loaded_metrics)}")
        # Fallback to available objective metric
        for fallback in ['Best Objective', 'Objective value']:
            if fallback in loaded_metrics:
                objective_metric = fallback
                st.info(f"Falling back to '{fallback}'")
                break
        else:
            st.error("No objective metric found in database.")
            st.stop()

    # Main content: Leaderboard
    leaderboard_df = load_leaderboard(conn, limit=3)
    render_leaderboard(leaderboard_df)

    # Main content: Progress stats
    stats = load_progress_stats(conn)
    render_progress_stats(stats, metadata)
    
    # Main content: Progress plots
    replica_ids = sorted(metrics_df['replica_id'].unique())
    replica_temps = load_replica_temperatures(conn)
    
    n_cols = plot_config['n_cols']
    for idx, replica_id in enumerate(replica_ids):
        # Create column layout
        if idx % n_cols == 0:
            cols = st.columns(n_cols)
        
        with cols[idx % n_cols]:
            fig = create_replica_plot(
                metrics_df=metrics_df,
                replica_id=replica_id,
                objective_metric=objective_metric,
                additional_metrics=plot_config['additional_metrics'],
                exchange_interval=metadata['exchange_interval'],
                replica_temps=replica_temps,
                exchanges_df=exchanges_df,
                normalize_metrics=plot_config['normalize_metrics'],
                show_exchanges=plot_config['show_exchanges']
            )
            st.plotly_chart(fig, width='stretch')
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


    # End render


def main():
    """Launch the Streamlit dashboard via `streamlit run` for CLI use.

    This replaces the current process with `streamlit run` pointing at this
    module file, ensuring proper Streamlit runtime initialization.
    """
    module_path = Path(__file__).resolve()
    os.execvp('streamlit', ['streamlit', 'run', str(module_path)])


if __name__ == "__main__":
    # If launched directly (streamlit run will set __name__ == "__main__"), render the app.
    # When imported via console script, only main() is invoked and render() is not executed,
    # avoiding bare-mode warnings.
    render()
