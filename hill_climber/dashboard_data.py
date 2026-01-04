"""Data loading and processing functions for the hill climber dashboard.

This module handles all database queries and data transformations,
providing a clean separation from UI logic.
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd

from .dashboard_imports import st, HAS_STREAMLIT

# Set up logging for dashboard data operations
logger = logging.getLogger(__name__)


def get_connection(db_path_str: str) -> sqlite3.Connection:
    """Create a cached read-only SQLite connection with performance optimizations.
    
    Connection is cached by Streamlit when available. Multiple PRAGMAs are set
    to optimize for read-heavy dashboard workloads.
    
    Args:
        db_path_str (str): Path to the SQLite database file.
        
    Returns:
        sqlite3.Connection: Optimized read-only SQLite connection object.
    """
    # Use Streamlit caching if available
    if HAS_STREAMLIT:
        return _get_connection_cached(db_path_str)
    else:
        return _create_connection(db_path_str)


if HAS_STREAMLIT:
    @st.cache_resource
    def _get_connection_cached(db_path_str: str) -> sqlite3.Connection:
        """Streamlit-cached connection creation."""
        return _create_connection(db_path_str)


def _create_connection(db_path_str: str) -> sqlite3.Connection:
    """Create an optimized read-only SQLite connection.
    
    Args:
        db_path_str (str): Path to the SQLite database file.
        
    Returns:
        sqlite3.Connection: Optimized connection.
    """
    conn = sqlite3.connect(
        f"file:{db_path_str}?mode=ro", 
        uri=True, 
        check_same_thread=False
    )
    # Performance optimizations for read-only access
    conn.execute("PRAGMA query_only = ON")
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("PRAGMA cache_size = -128000")  # 128MB cache (increased for better performance)
    conn.execute("PRAGMA mmap_size = 536870912")  # 512MB memory-mapped I/O
    return conn


def load_run_metadata(conn: sqlite3.Connection) -> Optional[Dict[str, Any]]:
    """Load run metadata from database.
    
    Args:
        conn (sqlite3.Connection): SQLite connection.
        
    Returns:
        Dict[str, Any]: Dictionary with run metadata, or None if not found.
    """
    try:
        query = "SELECT * FROM run_metadata WHERE run_id = 1"
        cursor = conn.cursor()
        cursor.execute(query)
        row = cursor.fetchone()
        
        if row:
            return {
                'run_id': row[0],
                'start_time': row[1],
                'end_time': row[2] if len(row) > 2 else None,
                'n_replicas': row[3],
                'exchange_interval': row[4],
                'db_step_interval': row[5],
                'hyperparameters': json.loads(row[6]) if row[6] else {},
                'checkpoint_file': row[7] if len(row) > 7 else None,
                'objective_function_name': row[8] if len(row) > 8 else None,
                'dataset_size': row[9] if len(row) > 9 else None
            }
        return None
    except sqlite3.OperationalError:
        # Table doesn't exist - likely an old database schema
        return None


def load_metrics_history(
    conn: sqlite3.Connection,
    metric_names: Optional[List[str]] = None,
    history_type: str = 'improvements',
    max_points_per_replica: int = 500
) -> pd.DataFrame:
    """Load metrics history from JSON-denormalized tables.
    
    Loads data from different tables based on history_type:
    - 'improvements': Only new best values (monotonically improving)
    - 'accepted': All accepted steps (includes exploration)
    - 'perturbations': All sampled perturbations (sampled at db_step_interval)
    
    Args:
        conn (sqlite3.Connection): SQLite connection.
        metric_names (List[str], optional): List of metric names to load. Can include
            'Objective value' to load objectives. Default is None.
        history_type (str): Type of history to load - 'improvements', 'accepted', or 
            'perturbations'. Default is 'improvements'.
        max_points_per_replica (int): Unused, kept for compatibility. Default is 500.
        
    Returns:
        pd.DataFrame: DataFrame with columns: replica_id, perturbation_num, metric_name, value.
            Returns empty DataFrame if no data found.
    """
    if not metric_names:
        return pd.DataFrame()

    try:
        # Determine which table to query based on history_type
        if history_type == 'improvements':
            table = 'improvements'
            obj_column = 'best_objective'
        elif history_type == 'accepted':
            table = 'accepted_steps'
            obj_column = 'objective'
        elif history_type == 'perturbations':
            table = 'perturbations'
            obj_column = 'objective'
        else:
            raise ValueError(f"Invalid history_type: {history_type}. Must be 'improvements', 'accepted', or 'perturbations'")
        
        # Load data with metrics JSON (skip perturbations as they don't have metrics)
        if history_type == 'perturbations':
            query = f"""
                SELECT replica_id, perturbation_num, {obj_column} as objective
                FROM {table}
                ORDER BY replica_id, perturbation_num
            """
        else:
            query = f"""
                SELECT replica_id, perturbation_num, {obj_column} as objective, metrics
                FROM {table}
                ORDER BY replica_id, perturbation_num
            """
        
        df = pd.read_sql_query(query, conn)
        
        if df.empty:
            return pd.DataFrame()
        
        # Build result with requested metrics
        result_dfs = []
        
        # Add objective if requested
        if 'Objective value' in metric_names:
            obj_df = df[['replica_id', 'perturbation_num', 'objective']].copy()
            obj_df['metric_name'] = 'Objective value'
            obj_df.rename(columns={'objective': 'value'}, inplace=True)
            result_dfs.append(obj_df)
        
        # Parse JSON and extract user metrics if requested and available
        user_metrics = [m for m in metric_names if m != 'Objective value']
        if user_metrics and history_type != 'perturbations' and 'metrics' in df.columns:
            # Parse JSON metrics
            metrics_list = []
            for _, row in df.iterrows():
                if pd.notna(row['metrics']) and row['metrics']:
                    try:
                        metrics_dict = json.loads(row['metrics'])
                        for metric_name in user_metrics:
                            if metric_name in metrics_dict:
                                metrics_list.append({
                                    'replica_id': row['replica_id'],
                                    'perturbation_num': row['perturbation_num'],
                                    'metric_name': metric_name,
                                    'value': metrics_dict[metric_name]
                                })
                    except json.JSONDecodeError:
                        pass  # Skip invalid JSON
            
            if metrics_list:
                metrics_df = pd.DataFrame(metrics_list)
                result_dfs.append(metrics_df)
        
        return pd.concat(result_dfs, ignore_index=True) if result_dfs else pd.DataFrame()
    except Exception as e:
        logger.warning(f"Error loading metrics history: {e}")
        return pd.DataFrame()


def load_temperature_exchanges(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load temperature exchange events.
    
    Args:
        conn (sqlite3.Connection): SQLite connection.
        
    Returns:
        pd.DataFrame: DataFrame with columns: perturbation_num, replica_id, new_temperature, timestamp.
            Returns empty DataFrame if no data found.
    """
    query = """
        SELECT perturbation_num, replica_id, new_temperature, timestamp
        FROM temperature_exchanges
        ORDER BY perturbation_num
    """
    try:
        return pd.read_sql_query(query, conn)
    except Exception as e:
        logger.warning(f"Error loading temperature exchanges: {e}")
        return pd.DataFrame()


def get_available_metrics(conn: sqlite3.Connection, history_type: str = 'improvements') -> List[str]:
    """Get list of all metric names in the database.
    
    Includes 'Objective value' plus all user-defined metrics extracted from JSON
    in the appropriate table based on history_type.
    
    Args:
        conn (sqlite3.Connection): SQLite connection.
        history_type (str): Type of history - 'improvements', 'accepted', or 'perturbations'.
            Default is 'improvements'.
        
    Returns:
        List[str]: Sorted list of unique metric names including 'Objective value'.
    """
    metrics = ['Objective value']  # Always include objective
    
    # Determine which table to query
    if history_type == 'improvements':
        table = 'improvements'
    elif history_type == 'accepted':
        table = 'accepted_steps'
    elif history_type == 'perturbations':
        # Perturbations don't have metrics
        return metrics
    else:
        table = 'improvements'  # Default fallback
    
    # Extract metric names from first JSON entry
    query = f"SELECT metrics FROM {table} WHERE metrics IS NOT NULL LIMIT 1"
    cursor = conn.cursor()
    try:
        cursor.execute(query)
        row = cursor.fetchone()
        if row and row[0]:
            try:
                metrics_dict = json.loads(row[0])
                metrics.extend(sorted(metrics_dict.keys()))
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON in metrics column: {e}")
                pass  # Skip invalid JSON
    except Exception as e:
        logger.warning(f"Error getting available metrics: {e}")
        pass  # Table might not exist or have data yet
    
    return metrics


def get_project_root() -> Path:
    """Find the project root by looking for pyproject.toml or .git.
    
    Returns:
        Path: Project root directory.
    """
    cwd = Path.cwd()
    for parent in [cwd] + list(cwd.parents):
        if (parent / 'pyproject.toml').exists() or (parent / '.git').exists():
            return parent
    return cwd


def find_all_databases(base_path: Optional[Path] = None) -> List[Path]:
    """Find all .db files recursively within project.
    
    Searches from base_path (or project root) and returns all .db files,
    excluding hidden directories and common build/cache folders.
    
    Args:
        base_path (Path, optional): Base directory to search. Defaults to project root.
    
    Returns:
        List[Path]: Sorted list of database file paths.
    """
    if base_path is None:
        base_path = get_project_root()
    
    db_files = []
    
    # Directories to exclude from search
    exclude_dirs = {'.git', '__pycache__', '.pytest_cache', 'node_modules', 
                    '.venv', 'venv', 'env', '.tox', 'build', 'dist', '.eggs'}
    
    try:
        for item in base_path.rglob('*.db'):
            # Skip if any parent directory is in exclude list
            if any(part.startswith('.') or part in exclude_dirs for part in item.parts):
                continue
            if item.is_file():
                db_files.append(item)
    except PermissionError:
        pass
    
    # Sort by path for consistent ordering
    return sorted(db_files)


def load_leaderboard(conn: sqlite3.Connection, limit: int = 3) -> pd.DataFrame:
    """Load replica leaderboard data.
    
    Args:
        conn (sqlite3.Connection): SQLite connection.
        limit (int): Maximum number of replicas to return. Default is 3.
        
    Returns:
        pd.DataFrame: DataFrame with replica_id, best_objective, current_perturbation_num, temperature.
            Returns empty DataFrame if no data found.
    """
    query = """
        SELECT replica_id, best_objective, current_perturbation_num, temperature
        FROM replica_status
        ORDER BY best_objective DESC
        LIMIT ?
    """
    try:
        return pd.read_sql_query(query, conn, params=(limit,))
    except Exception as e:
        logger.warning(f"Error loading leaderboard: {e}")
        return pd.DataFrame()


def load_replica_temperatures(conn: sqlite3.Connection) -> Dict[int, float]:
    """Load current temperatures for all replicas.
    
    Args:
        conn (sqlite3.Connection): SQLite connection.
        
    Returns:
        Dict[int, float]: Dictionary mapping replica_id to temperature.
    """
    query = "SELECT replica_id, temperature FROM replica_status"
    try:
        temp_df = pd.read_sql_query(query, conn)
        return dict(zip(temp_df['replica_id'], temp_df['temperature']))
    except Exception as e:
        logger.warning(f"Error loading replica temperatures: {e}")
        return {}


def load_temperature_ladder(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load initial temperatures from replica_status table.
    
    Args:
        conn (sqlite3.Connection): SQLite connection.
        
    Returns:
        pd.DataFrame: DataFrame with replica_id and temperature columns.
    """
    query = "SELECT replica_id, temperature FROM replica_status ORDER BY replica_id"
    try:
        return pd.read_sql_query(query, conn)
    except Exception as e:
        logger.warning(f"Error loading temperature ladder: {e}")
        return pd.DataFrame()


def load_temperature_ladder_history(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load temperature ladder history tracking temperature at each ladder position over time.
    
    Args:
        conn (sqlite3.Connection): SQLite connection.
        
    Returns:
        pd.DataFrame: DataFrame with columns: batch_num, ladder_position, temperature.
    """
    query = """
        SELECT batch_num, ladder_position, temperature
        FROM temperature_ladder_history
        ORDER BY batch_num, ladder_position
    """
    try:
        return pd.read_sql_query(query, conn)
    except Exception as e:
        logger.debug(f"Temperature ladder history not available: {e}")
        # Table doesn't exist yet - return empty DataFrame
        return pd.DataFrame(columns=['batch_num', 'ladder_position', 'temperature'])


def load_batch_statistics(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load batch statistics including step spread and acceptance rates.
    
    Args:
        conn (sqlite3.Connection): SQLite connection.
        
    Returns:
        pd.DataFrame: DataFrame with columns: batch_num, step_spread, 
            mean_acceptance_rate, min_acceptance_rate, max_acceptance_rate.
    """
    query = """
        SELECT batch_num, step_spread, mean_acceptance_rate, min_acceptance_rate, max_acceptance_rate
        FROM batch_statistics
        ORDER BY batch_num
    """
    try:
        return pd.read_sql_query(query, conn)
    except Exception as e:
        logger.debug(f"Batch statistics not available: {e}")
        # Table doesn't exist yet - return empty DataFrame
        return pd.DataFrame(columns=['batch_num', 'step_spread', 'mean_acceptance_rate', 'min_acceptance_rate', 'max_acceptance_rate'])


def load_progress_stats(conn: sqlite3.Connection) -> Dict[str, Any]:
    """Load progress statistics including perturbation counts and acceptance counts.
    
    Args:
        conn (sqlite3.Connection): SQLite connection.
        
    Returns:
        Dict[str, Any]: Dictionary with total_perturbations and total_accepted.
            Returns zeros if no data found.
    """
    query = """
        SELECT 
            SUM(current_perturbation_num) as total_perturbations,
            SUM(num_accepted) as total_accepted
        FROM replica_status
    """
    try:
        result = pd.read_sql_query(query, conn)
        if not result.empty:
            return {
                'total_perturbations': result['total_perturbations'].iloc[0] or 0,
                'total_accepted': result['total_accepted'].iloc[0] or 0
            }
    except Exception as e:
        logger.warning(f"Error loading progress stats: {e}")
    
    return {'total_perturbations': 0, 'total_accepted': 0}
