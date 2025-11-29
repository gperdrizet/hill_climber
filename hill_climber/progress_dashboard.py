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
    
    # Initialize plot refresh counter for forcing clean re-renders
    if 'plot_refresh_key' not in st.session_state:
        st.session_state.plot_refresh_key = 0


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

    # Sidebar: Plot options (renders widgets and updates session state)
    plot_config = render_plot_options(available_metrics)
    
    # Sidebar: Run information
    render_run_information(metadata)
    render_hyperparameters(metadata)
    
    # Sidebar: Temperature ladder
    temp_ladder_df = load_temperature_ladder(conn)
    render_temperature_ladder(temp_ladder_df)

    # Load data based on plot configuration
    metrics_df = load_metrics_history(
        conn,
        metric_names=[plot_config['objective_metric']] + plot_config['additional_metrics'],
        max_points_per_replica=plot_config['max_points']
    )
    exchanges_df = load_temperature_exchanges(conn)

    if metrics_df.empty:
        st.info("No metrics found yet. Waiting for data...")
        return

    # Verify objective metric exists in loaded data
    loaded_metrics = metrics_df['metric_name'].unique().tolist()
    
    if plot_config['objective_metric'] not in loaded_metrics:
        st.warning(f"'{plot_config['objective_metric']}' not found. Available: {', '.join(loaded_metrics)}")
        # Fallback to available objective metric
        for fallback in ['Best Objective', 'Objective value']:
            if fallback in loaded_metrics:
                st.info(f"Falling back to '{fallback}'")
                plot_config['objective_metric'] = fallback
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
    
    # Detect layout changes and increment refresh key to force clean re-render
    current_n_cols = plot_config['n_cols']
    if 'previous_layout' not in st.session_state:
        st.session_state.previous_layout = current_n_cols
    
    layout_changed = st.session_state.previous_layout != current_n_cols
    if layout_changed:
        st.session_state.previous_layout = current_n_cols
        st.session_state.plot_refresh_key += 1
    
    # Use a keyed container to ensure complete re-rendering on refresh
    # The key changes on layout changes, forcing Streamlit to completely
    # rebuild the container and eliminate ghost plots
    plot_container_key = f"plot_container_{st.session_state.plot_refresh_key}_{current_n_cols}"
    
    # Render all plots inside a keyed container
    with st.container(key=plot_container_key):
        # Create all plots
        for idx, replica_id in enumerate(replica_ids):
            # Create column layout at start of each row
            if idx % current_n_cols == 0:
                cols = st.columns(current_n_cols)
            
            # Use the appropriate column
            col_idx = idx % current_n_cols
            with cols[col_idx]:
                fig = create_replica_plot(
                    metrics_df=metrics_df,
                    replica_id=replica_id,
                    objective_metric=plot_config['objective_metric'],
                    additional_metrics=plot_config['additional_metrics'],
                    exchange_interval=metadata['exchange_interval'],
                    replica_temps=replica_temps,
                    exchanges_df=exchanges_df,
                    normalize_metrics=plot_config['normalize_metrics'],
                    show_exchanges=plot_config['show_exchanges']
                )
                st.plotly_chart(fig, key=f"plot_{replica_id}_{plot_container_key}", width='stretch')
    
    # Auto-refresh logic
    if auto_refresh:
        # Increment refresh key to force clean plot re-rendering
        st.session_state.plot_refresh_key = st.session_state.get('plot_refresh_key', 0) + 1
        # Save current plot options before auto-refresh
        st.session_state.saved_history_type = st.session_state.get('history_type', 'Best')
        st.session_state.saved_additional_metrics = st.session_state.get('additional_metrics', [])
        st.session_state.saved_normalize_metrics = st.session_state.get('normalize_metrics', False)
        st.session_state.saved_show_exchanges = st.session_state.get('show_exchanges', False)
        st.session_state.saved_max_points = st.session_state.get('max_points', 1000)
        st.session_state.saved_plot_columns = st.session_state.get('plot_columns', 'Two columns')
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
