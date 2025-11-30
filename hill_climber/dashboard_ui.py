"""Streamlit UI components for the hill climber dashboard.

This module contains all Streamlit-specific UI rendering logic,
keeping UI concerns separate from data and plotting.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Set, Tuple
import pandas as pd

try:
    import streamlit as st
except ImportError:
    st = None


def apply_custom_css() -> None:
    """Apply custom CSS styling to the dashboard.
    
    Adjusts padding, font sizes, and text wrapping for optimal dashboard appearance.
    """
    if st is None:
        return
        
    st.markdown("""
        <style>
        .main { padding-top: 0 !important; }
        .main .block-container { padding-top: 0.5rem !important; }
        .main h2 { font-size: 1.5rem !important; }
        .main h3 { font-size: 1.1rem !important; }
        .main h2:first-of-type { margin-top: 0 !important; padding-top: 0 !important; }
        
        /* Prevent text wrapping in sidebar - use ellipsis instead */
        [data-testid="stSidebar"] p {
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
        }
        </style>
    """, unsafe_allow_html=True)


def render_sidebar_title() -> None:
    """Render the sidebar title.
    
    Displays 'Hill climber' as a multi-line title in the sidebar.
    """
    if st is None:
        return
        
    st.sidebar.markdown(
        "<h1 style='margin-top: -2rem; padding-top: 0.25rem; font-size: 2.8rem; line-height: 1.2;'>"
        "Hill<br>climber</h1>",
        unsafe_allow_html=True
    )


def render_database_selector(session_state: Any, dirs: List[Path]) -> Optional[str]:
    """Render database selection UI in sidebar.
    
    Args:
        session_state (Any): Streamlit session state object.
        dirs (List[Path]): List of available directories to search for databases.
        
    Returns:
        str: Selected database path, or None if no database selected.
    """
    if st is None:
        return None
        
    st.sidebar.header("Configuration")
    
    db_path = session_state.db_path
    expander_open = not (session_state.db_user_selected and Path(db_path).exists())
    
    with st.sidebar.expander("Database", expanded=expander_open):
        st.write("Select the directory and database file.")

        # Directory dropdown
        cwd = Path.cwd()
        dir_labels = ["None"]
        for d in dirs:
            try:
                rel = d.relative_to(cwd)
                label = "." if rel == Path('.') else str(rel)
            except ValueError:
                label = str(d)
            dir_labels.append(label)

        selected_dir_idx = st.selectbox(
            "Directory",
            options=list(range(len(dir_labels))),
            format_func=lambda i: dir_labels[i]
        )
        
        selected_dir = dirs[selected_dir_idx - 1] if selected_dir_idx > 0 else None

        # File dropdown
        file_labels = ["None"]
        file_candidates = []
        
        if selected_dir is not None:
            try:
                file_candidates = [p for p in selected_dir.iterdir() if p.is_file() and p.suffix == ".db"]
                if file_candidates:
                    file_labels.extend([p.name for p in file_candidates])
                else:
                    st.info("No .db files in selected directory")
            except PermissionError:
                st.warning("Permission denied reading directory")

        selected_file_idx = st.selectbox(
            "Database file",
            options=list(range(len(file_labels))),
            format_func=lambda i: file_labels[i]
        )

        # Apply selection
        if st.sidebar.button("Use selected file", key="db_use_selected"):
            if selected_dir_idx > 0 and selected_file_idx > 0 and file_candidates:
                chosen_file = file_candidates[selected_file_idx - 1]
                session_state.db_path = str(chosen_file)
                session_state.db_user_selected = True
                st.toast("Database selected")
                st.rerun()
            else:
                st.warning("Please select both a directory and a database file")
    
    # Warn if path doesn't exist
    if session_state.db_user_selected and not Path(db_path).exists():
        st.sidebar.warning(f"⚠ Not found: `{db_path}`")
    
    return db_path


def render_auto_refresh_controls() -> Tuple[bool, float]:
    """Render auto-refresh controls in sidebar.
    
    Returns:
        Tuple[bool, float]: Tuple of (auto_refresh_enabled, refresh_interval_seconds).
    """
    if st is None:
        return False, 60.0
        
    st.sidebar.markdown("---")
    auto_refresh = st.sidebar.checkbox("Auto-refresh", key="auto_refresh")
    refresh_interval_minutes = st.sidebar.slider(
        "Refresh interval (minutes)",
        min_value=0.5, max_value=5.0, step=0.5,
        key="refresh_interval"
    )
    if st.sidebar.button("Refresh Now", key="refresh_now"):
        # Increment refresh key to force clean plot re-rendering
        st.session_state.plot_refresh_key = st.session_state.get('plot_refresh_key', 0) + 1
        # Save current plot options before refresh
        st.session_state.saved_history_type = st.session_state.get('history_type', 'Best')
        st.session_state.saved_additional_metrics = st.session_state.get('additional_metrics', [])
        st.session_state.saved_normalize_metrics = st.session_state.get('normalize_metrics', False)
        st.session_state.saved_show_exchanges = st.session_state.get('show_exchanges', False)
        st.session_state.saved_max_points = st.session_state.get('max_points', 1000)
        st.session_state.saved_plot_columns = st.session_state.get('plot_columns', 'Two columns')
        st.rerun()
    
    return auto_refresh, refresh_interval_minutes * 60


def render_plot_options(available_metrics: List[str]) -> Dict[str, Any]:
    """Render plot configuration options in sidebar.
    
    Args:
        available_metrics (List[str]): List of available metric names from database.
        
    Returns:
        Dict[str, Any]: Dictionary with plot configuration options including history_type,
            objective_metric, additional_metrics, normalize_metrics, show_exchanges,
            max_points, and n_cols.
    """
    if st is None:
        return {}
        
    st.sidebar.markdown("---")
    st.sidebar.subheader("Plot options")
    
    # Restore saved values if they exist (from manual refresh)
    if 'saved_history_type' in st.session_state and 'history_type' not in st.session_state:
        st.session_state.history_type = st.session_state.saved_history_type
    if 'saved_additional_metrics' in st.session_state and 'additional_metrics' not in st.session_state:
        st.session_state.additional_metrics = st.session_state.saved_additional_metrics
    if 'saved_normalize_metrics' in st.session_state and 'normalize_metrics' not in st.session_state:
        st.session_state.normalize_metrics = st.session_state.saved_normalize_metrics
    if 'saved_show_exchanges' in st.session_state and 'show_exchanges' not in st.session_state:
        st.session_state.show_exchanges = st.session_state.saved_show_exchanges
    if 'saved_max_points' in st.session_state and 'max_points' not in st.session_state:
        st.session_state.max_points = st.session_state.saved_max_points
    if 'saved_plot_columns' in st.session_state and 'plot_columns' not in st.session_state:
        st.session_state.plot_columns = st.session_state.saved_plot_columns
    
    # Set initial defaults for widgets if not in session state
    if 'max_points' not in st.session_state:
        st.session_state.max_points = 1000
    if 'plot_columns' not in st.session_state:
        st.session_state.plot_columns = 'Two columns'
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = True
    if 'refresh_interval' not in st.session_state:
        st.session_state.refresh_interval = 1.0
    
    # History type - widget value automatically preserved via key
    history_type = st.sidebar.radio(
        "History type",
        options=["Best", "Current"],
        key="history_type",
        help="Best: Monotonically improving (best so far)\nCurrent: Includes exploration"
    )
    
    # Extract base metrics
    base_metrics: Set[str] = set()
    for m in available_metrics:
        if m.startswith("Best "):
            base_metrics.add(m[5:])
        elif m.startswith("Current "):
            base_metrics.add(m[8:])
        elif m not in ["Objective value", "Best Objective", "Current Objective"]:
            base_metrics.add(m)
    
    base_metrics.discard("Objective")
    
    # Additional metrics - widget value automatically preserved via key
    additional_base_metrics = st.sidebar.multiselect(
        "Additional metrics",
        options=sorted(base_metrics),
        key="additional_metrics"
    )
    
    # Build full metric names from current widget values
    objective_metric = f"{history_type} Objective"
    additional_metrics = [f"{history_type} {m}" for m in additional_base_metrics]
    
    # Other options - widget values automatically preserved via keys
    normalize_metrics = st.sidebar.checkbox("Normalize", key="normalize_metrics")
    show_exchanges = st.sidebar.checkbox(
        "Show exchange markers",
        key="show_exchanges",
        help="Draw vertical markers at replica exchange events"
    )
    max_points = st.sidebar.slider(
        "Max points per replica",
        min_value=100, max_value=2000, step=100,
        key="max_points"
    )
    
    # Layout - widget value automatically preserved via key
    st.sidebar.markdown("**Plot layout:**")
    plot_columns = st.sidebar.radio(
        "Plot layout",
        options=["One column", "Two columns"],
        key="plot_columns",
        help="Switch between two-column or single-column plot layout",
        label_visibility="collapsed"
    )
    n_cols = 2 if plot_columns == "Two columns" else 1
    
    return {
        'history_type': history_type,
        'objective_metric': objective_metric,
        'additional_metrics': additional_metrics,
        'normalize_metrics': normalize_metrics,
        'show_exchanges': show_exchanges,
        'max_points': max_points,
        'n_cols': n_cols
    }


def render_run_information(metadata: Dict[str, Any]) -> None:
    """Render run information section in sidebar.
    
    Args:
        metadata (Dict[str, Any]): Dictionary with run metadata including start_time,
            hyperparameters, n_replicas, checkpoint_file, objective_function_name,
            and dataset_size.
    """
    if st is None:
        return
        
    st.sidebar.markdown("---")
    st.sidebar.subheader("Run information")
    
    hyperparams = metadata['hyperparameters']
    
    # Format display values
    max_time_seconds = hyperparams.get('max_time', 0)
    if max_time_seconds:
        if max_time_seconds > 3600:  # More than 60 minutes
            max_time_display = f"{max_time_seconds / 3600:.2f} hr"
        elif max_time_seconds > 60:  # More than 60 seconds
            max_time_display = f"{int(max_time_seconds / 60)} min"
        else:
            max_time_display = f"{max_time_seconds} sec"
    else:
        max_time_display = 'N/A'
    
    checkpoint_file = metadata.get('checkpoint_file')
    checkpoint_display = Path(checkpoint_file).name if checkpoint_file else 'None'
    
    objective_func_name = metadata.get('objective_function_name', 'N/A') or 'N/A'
    
    dataset_size = metadata.get('dataset_size')
    dataset_size_display = f"{dataset_size:,}" if dataset_size else 'N/A'
    
    run_info = f"""**Started:** {datetime.fromtimestamp(metadata['start_time']).strftime('%Y-%m-%d %H:%M')}  
**Max time:** {max_time_display}  
**Dataset size:** {dataset_size_display}  
**Replicas:** {metadata['n_replicas']}  
**Objective:** {objective_func_name}  
**Checkpoint:** {checkpoint_display}"""
    
    st.sidebar.markdown(run_info)


def render_hyperparameters(metadata: Dict[str, Any]) -> None:
    """Render hyperparameters section in sidebar.
    
    Args:
        metadata (Dict[str, Any]): Dictionary with run metadata containing hyperparameters
            and exchange_interval.
    """
    if st is None:
        return
        
    st.sidebar.markdown("---")
    st.sidebar.subheader("Hyperparameters")
    
    hyperparams = metadata['hyperparameters']
    
    # Format temperatures in scientific notation
    t_min = hyperparams.get('T_min', 'N/A')
    t_max = hyperparams.get('T_max', 'N/A')
    t_min_str = f"{t_min:.1e}" if isinstance(t_min, (int, float)) else t_min
    t_max_str = f"{t_max:.1e}" if isinstance(t_max, (int, float)) else t_max
    
    hyperparams_text = f"""**Mode:** {hyperparams.get('mode', 'N/A')}  
**Step spread:** {hyperparams.get('step_spread', 'N/A')}  
**Perturb fraction:** {hyperparams.get('perturb_fraction', 'N/A')}  
**Cooling rate:** {hyperparams.get('cooling_rate', 'N/A')}  
**Exchange interval:** {metadata['exchange_interval']}  
**Exchange strategy:** {hyperparams.get('exchange_strategy', 'N/A')}  
**T_min:** {t_min_str}  
**T_max:** {t_max_str}"""
    
    st.sidebar.markdown(hyperparams_text)


def render_temperature_ladder(temp_ladder_df: pd.DataFrame) -> None:
    """Render temperature ladder section in sidebar.
    
    Args:
        temp_ladder_df (pd.DataFrame): DataFrame with replica_id and temperature columns.
    """
    if st is None:
        return
        
    st.sidebar.markdown("---")
    st.sidebar.subheader("Temperature ladder")
    
    if not temp_ladder_df.empty:
        temp_lines = []
        for _, row in temp_ladder_df.iterrows():
            temp_lines.append(f"**Replica {int(row['replica_id'])}:** {row['temperature']:.1e}")
        st.sidebar.markdown("  \n".join(temp_lines))


def render_leaderboard(leaderboard_df: pd.DataFrame) -> None:
    """Render replica leaderboard in main content area.
    
    Args:
        leaderboard_df (pd.DataFrame): DataFrame with replica stats including replica_id,
            best_objective, step, and temperature.
    """
    if st is None:
        return
        
    st.header("Replica leaderboard")
    
    if not leaderboard_df.empty:
        cols = st.columns(3)
        medals = ['1<sup>st</sup>:', '2<sup>nd</sup>:', '3<sup>rd</sup>:']
        for idx, (_, row) in enumerate(leaderboard_df.iterrows()):
            with cols[idx]:
                st.markdown(f"### {medals[idx]} Replica {int(row['replica_id'])}", unsafe_allow_html=True)
                st.markdown(
                    f"**Objective:** {row['best_objective']:.4f}  \n"
                    f"**Accepted steps:** {int(row['step'])}  \n"
                    f"**Temperature:** {row['temperature']:.1e}"
                )
    else:
        st.info("No replica data available yet")


def render_progress_stats(stats: Dict[str, Any], metadata: Dict[str, Any]) -> None:
    """Render progress statistics in main content area.
    
    Args:
        stats (Dict[str, Any]): Dictionary with total_iterations and total_accepted.
        metadata (Dict[str, Any]): Dictionary with run metadata including start_time.
    """
    if st is None:
        return
    
    total_iterations = stats.get('total_iterations', 0)
    total_accepted = stats.get('total_accepted', 0)
    elapsed_time = time.time() - metadata['start_time']
    
    # Format elapsed time
    if elapsed_time > 3600:  # More than 60 minutes
        elapsed_display = f"{elapsed_time / 3600:.2f} hr"
    elif elapsed_time > 60:  # More than 60 seconds
        elapsed_display = f"{int(elapsed_time / 60)} min"
    else:
        elapsed_display = f"{int(elapsed_time)} sec"
    
    st.header(f"Optimization progress ({elapsed_display})")
    
    if elapsed_time > 0 and total_iterations:
        exploration_rate = total_iterations / elapsed_time
        progress_rate = total_accepted / elapsed_time
        acceptance_rate = (total_accepted / total_iterations * 100) if total_iterations > 0 else 0
        
        st.markdown(
            f"**Exploration rate:** {exploration_rate:,.1f} steps/sec | "
            f"**Progress rate:** {progress_rate:,.1f} moves/sec | "
            f"**Acceptance rate:** {acceptance_rate:.1f}% ({total_accepted:,} / {total_iterations:,})"
        )
    else:
        st.markdown("**Exploration rate:** N/A")
