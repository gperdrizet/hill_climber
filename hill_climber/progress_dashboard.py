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
from typing import Any


def _init_session_state(st: Any) -> None:
    """Initialize session state variables.
    
    Args:
        st (Any): Streamlit module.
    """
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


def render() -> None:
    """Render the Streamlit dashboard.
    
    Main dashboard rendering function that orchestrates all UI components,
    data loading, and plot generation.
    """
    # Import modular dashboard components (use absolute imports for Streamlit compatibility)
    from hill_climber.dashboard_data import (
        get_connection,
        load_run_metadata,
        load_metrics_history,
        load_temperature_exchanges,
        load_temperature_ladder_history,
        load_batch_statistics,
        get_available_metrics,
        find_all_databases,
        get_project_root,
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
    from hill_climber.dashboard_plots import (
        create_replica_plot, 
        create_temperature_ladder_plot,
        create_batch_statistics_plot
    )
    
    try:
        import streamlit as st
        import pandas as pd
    except ImportError as e:
        print(f"Missing dependency: {e}")
        sys.exit(1)

    # Page config
    import os
    icon_path = os.path.join(os.path.dirname(__file__), 'assets', 'favicon.svg')
    st.set_page_config(
        page_title="Dashboard",
        page_icon=icon_path if os.path.exists(icon_path) else None,
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Apply custom styling
    apply_custom_css()
    render_sidebar_title()

    # Initialize session state
    _init_session_state(st)
    
    # Sidebar: Database selection
    db_files = find_all_databases(get_project_root())
    db_path = render_database_selector(st.session_state, db_files, get_project_root())
    
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

    # Get all available metrics (from improvements table - superset of all metrics)
    # Note: perturbations won't have detailed metrics, but UI will show them in selector
    available_metrics = get_available_metrics(conn, history_type='improvements')
    
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
        history_type=plot_config['history_type'],
        max_points_per_replica=plot_config['max_points']
    )
    exchanges_df = load_temperature_exchanges(conn)
    temp_ladder_history_df = load_temperature_ladder_history(conn)
    batch_stats_df = load_batch_statistics(conn)

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
    st.markdown("---")
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
    
    # Generate unique key for this render that includes layout
    plot_container_key = f"{st.session_state.plot_refresh_key}_{current_n_cols}"
    
    # Create temperature ladder plot as first plot in grid
    if 0 % current_n_cols == 0:
        cols = st.columns(current_n_cols)
    
    with cols[0]:
        temp_ladder_fig = create_temperature_ladder_plot(
            temp_ladder_history_df=temp_ladder_history_df,
            temp_ladder_df=temp_ladder_df
        )
        st.plotly_chart(temp_ladder_fig, key=f"plot_temp_ladder_{plot_container_key}", use_container_width=True)
    
    # Create batch statistics plot as second plot in grid
    plot_idx = 1
    if plot_idx % current_n_cols == 0:
        cols = st.columns(current_n_cols)
    
    col_idx = plot_idx % current_n_cols
    with cols[col_idx]:
        batch_stats_fig = create_batch_statistics_plot(batch_stats_df)
        st.plotly_chart(batch_stats_fig, key=f"plot_batch_stats_{plot_container_key}", use_container_width=True)
    
    # Create all replica plots (offset by 2 for temp ladder and batch stats)
    for idx, replica_id in enumerate(replica_ids):
        # Offset by 2 to account for temperature ladder and batch stats plots
        plot_idx = idx + 2
        
        # Create column layout at start of each row
        if plot_idx % current_n_cols == 0:
            cols = st.columns(current_n_cols)
        
        # Use the appropriate column
        col_idx = plot_idx % current_n_cols
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
            st.plotly_chart(fig, key=f"plot_{replica_id}_{plot_container_key}", use_container_width=True)
    
    # Auto-refresh logic
    if auto_refresh:
        # Increment refresh key to force clean plot re-rendering
        st.session_state.plot_refresh_key = st.session_state.get('plot_refresh_key', 0) + 1
        # Save current plot options before auto-refresh
        st.session_state.saved_history_type = st.session_state.get('history_type', 'Best')
        st.session_state.saved_additional_base_metrics = st.session_state.get('additional_base_metrics', [])
        st.session_state.saved_normalize_metrics = st.session_state.get('normalize_metrics', False)
        st.session_state.saved_show_exchanges = st.session_state.get('show_exchanges', False)
        st.session_state.saved_max_points = st.session_state.get('max_points', 1000)
        st.session_state.saved_plot_columns = st.session_state.get('plot_columns', 'Two columns')
        time.sleep(refresh_interval)
        st.rerun()


    # End render


def main() -> None:
    """Launch the Streamlit dashboard via streamlit run for CLI use.

    This replaces the current process with streamlit run pointing at this
    module file, ensuring proper Streamlit runtime initialization.
    """
    module_path = Path(__file__).resolve()
    os.execvp('streamlit', [
        'streamlit', 'run',
        '--server.headless=true',
        '--server.showEmailPrompt=false',
        '--browser.gatherUsageStats=false',
        str(module_path)
    ])


if __name__ == "__main__":
    # If launched directly (streamlit run will set __name__ == "__main__"), render the app.
    # When imported via console script, only main() is invoked and render() is not executed,
    # avoiding bare-mode warnings.
    render()
