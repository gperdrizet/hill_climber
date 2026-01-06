"""Streamlit dashboard for monitoring hill climber optimization progress in real-time.

This module can be launched via the console script `hill-climber-dashboard`
once the package is installed, or directly in development using:

    python -m hill_climber.progress_dashboard

It requires `streamlit`, `plotly`, and `pandas` to be installed.
"""

import sys
import os
import logging
from datetime import timedelta
from pathlib import Path
from typing import Any

# Configure logging to terminal
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Use absolute import to work when run via streamlit run
try:
    from hill_climber.dashboard_imports import st, HAS_STREAMLIT
except ImportError:
    # Fallback for relative import when imported as module
    from .dashboard_imports import st, HAS_STREAMLIT


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


def render() -> None:
    """Render the Streamlit dashboard.
    
    Main dashboard rendering function that orchestrates all UI components,
    data loading, and plot generation.
    """
    logger.info("=" * 60)
    logger.info("render() START")
    
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
        load_progress_stats,
        clear_data_cache
    )
    from hill_climber.dashboard_ui import (
        apply_custom_css,
        render_sidebar_title,
        render_database_selector,
        render_auto_refresh_controls,
        render_plot_options,
        render_run_information,
        render_hyperparameters,
        render_leaderboard,
        render_progress_stats
    )
    from hill_climber.dashboard_plots import (
        create_replica_plot, 
        create_temperature_ladder_plot,
        create_batch_statistics_plot
    )
    
    if not HAS_STREAMLIT:
        print("Error: streamlit is required for the dashboard. Install with: pip install streamlit plotly")
        sys.exit(1)
    
    import pandas as pd

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
    logger.info("Session state initialized")
    
    # Sidebar: Database selection
    db_files = find_all_databases(get_project_root())
    db_path = render_database_selector(st.session_state, db_files, get_project_root())
    logger.info(f"Database path: {db_path}")
    
    # Sidebar: Auto-refresh controls (returns refresh_interval in seconds)
    auto_refresh, refresh_interval_seconds = render_auto_refresh_controls(clear_data_cache)
    logger.info(f"Auto-refresh: {auto_refresh}, interval: {refresh_interval_seconds}s")

    # Check database and connect
    if not Path(db_path).exists():
        st.info("Select a database in the sidebar to view progress.")
        logger.info("render() END - no database")
        return

    try:
        conn = get_connection(db_path)
    except Exception as e:
        st.error(f"Failed to connect to database: {e}")
        st.stop()

    metadata = load_run_metadata(conn)
    if metadata is None:
        st.warning("No run metadata found. Waiting for optimization to start...")
        st.stop()

    # Get all available metrics (from improvements table - superset of all metrics)
    # Note: perturbations won't have detailed metrics, but UI will show them in selector
    available_metrics = get_available_metrics(conn, history_type='improvements')
    
    # Sidebar: Plot options (renders widgets and updates session state)
    logger.info("About to call render_plot_options()")
    plot_config = render_plot_options(available_metrics)
    logger.info(f"plot_config returned: n_cols={plot_config['n_cols']}, history_type={plot_config['history_type']}, additional_metrics={plot_config['additional_metrics']}")
    
    # Sidebar: Run information
    render_run_information(metadata)
    render_hyperparameters(metadata)
    
    # Define the auto-refreshing content fragment
    # Use dynamic run_every based on user setting
    refresh_interval = timedelta(seconds=refresh_interval_seconds) if auto_refresh else None
    
    @st.fragment(run_every=refresh_interval)
    def refreshable_content():
        """Fragment that contains all refreshable dashboard content."""
        logger.info("refreshable_content() called")
        
        # Clear cache on each fragment run to get fresh data
        clear_data_cache()

        # Load temperature ladder (needed for plot, not sidebar)
        temp_ladder_df = load_temperature_ladder(db_path)

        # Load data based on plot configuration (uses cached functions)
        metrics_df = load_metrics_history(
            db_path,
            metric_names=[plot_config['objective_metric']] + plot_config['additional_metrics'],
            history_type=plot_config['history_type'],
            max_points_per_replica=plot_config['max_points']
        )
        
        # Only load temperature exchanges if user wants to see them (performance optimization)
        if plot_config['show_exchanges']:
            exchanges_df = load_temperature_exchanges(db_path)
        else:
            exchanges_df = pd.DataFrame()  # Empty DataFrame to skip loading
        
        temp_ladder_history_df = load_temperature_ladder_history(db_path)
        batch_stats_df = load_batch_statistics(db_path)

        if metrics_df.empty:
            st.info("No metrics found yet. Waiting for data...")
            return

        # Verify objective metric exists in loaded data
        loaded_metrics = metrics_df['metric_name'].unique().tolist()
        current_objective = plot_config['objective_metric']
        
        if current_objective not in loaded_metrics:
            st.warning(f"'{current_objective}' not found. Available: {', '.join(loaded_metrics)}")
            # Fallback to available objective metric
            for fallback in ['Best Objective', 'Objective value']:
                if fallback in loaded_metrics:
                    st.info(f"Falling back to '{fallback}'")
                    current_objective = fallback
                    break
            else:
                st.error("No objective metric found in database.")
                return

        # Main content: Leaderboard
        leaderboard_df = load_leaderboard(db_path, limit=3)
        render_leaderboard(leaderboard_df)

        # Main content: Progress stats
        stats = load_progress_stats(db_path)
        render_progress_stats(stats, metadata)
        
        # Main content: Progress plots
        # Load additional data needed for plots
        replica_temps = load_replica_temperatures(db_path)
        
        replica_ids = sorted(metrics_df['replica_id'].unique())
        current_n_cols = plot_config['n_cols']
        
        st.markdown("---")
        
        # List of all figures to plot in order
        figures = []
        
        # 1. Temperature Ladder
        temp_ladder_fig = create_temperature_ladder_plot(
            temp_ladder_history_df=temp_ladder_history_df,
            temp_ladder_df=temp_ladder_df
        )
        figures.append(("temp_ladder", temp_ladder_fig))
        
        # 2. Batch Statistics
        batch_stats_fig = create_batch_statistics_plot(batch_stats_df)
        figures.append(("batch_stats", batch_stats_fig))
        
        # 3. Replica Plots
        for replica_id in replica_ids:
            fig = create_replica_plot(
                metrics_df=metrics_df,
                replica_id=replica_id,
                objective_metric=current_objective,
                additional_metrics=plot_config['additional_metrics'],
                exchange_interval=metadata['exchange_interval'],
                replica_temps=replica_temps,
                exchanges_df=exchanges_df,
                normalize_metrics=plot_config['normalize_metrics'],
                show_exchanges=plot_config['show_exchanges']
            )
            figures.append((f"replica_{replica_id}", fig))
        
        logger.info(f"Created {len(figures)} figures, rendering in {current_n_cols} columns")
        
        # Render plots in grid
        for i in range(0, len(figures), current_n_cols):
            cols = st.columns(current_n_cols)
            for j in range(current_n_cols):
                if i + j < len(figures):
                    name, fig = figures[i + j]
                    with cols[j]:
                        st.plotly_chart(fig, width="stretch")
        
        logger.info("refreshable_content() complete")
    
    # Run the refreshable content fragment
    refreshable_content()
    
    logger.info("render() END - complete")


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
