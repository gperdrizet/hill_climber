"""Data loading and processing functions for the hill climber dashboard.

This module handles all database queries and data transformations,
providing a clean separation from UI logic.
"""

import json
import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd


def get_connection(db_path_str: str) -> sqlite3.Connection:
    """Create a read-only SQLite connection.
    
    Args:
        db_path_str (str): Path to the SQLite database file.
        
    Returns:
        sqlite3.Connection: Read-only SQLite connection object.
    """
    return sqlite3.connect(
        f"file:{db_path_str}?mode=ro", 
        uri=True, 
        check_same_thread=False
    )


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
    max_points_per_replica: int = 1000
) -> pd.DataFrame:
    """Load metrics history with optional downsampling for performance.
    
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
        max_points_per_replica (int): Downsample if more points exist. Default is 1000.
        
    Returns:
        pd.DataFrame: DataFrame with columns: replica_id, perturbation_num, metric_name, value.
            Returns empty DataFrame if no data found.
    """
    if not metric_names:
        return pd.DataFrame()

    try:
        all_dfs = []
        
        # Determine which tables to query based on history_type
        if history_type == 'improvements':
            obj_table = 'improvements'
            obj_column = 'best_objective'
            metrics_table = 'improvement_metrics'
        elif history_type == 'accepted':
            obj_table = 'accepted_steps'
            obj_column = 'objective'
            metrics_table = 'step_metrics'
        elif history_type == 'perturbations':
            obj_table = 'perturbations'
            obj_column = 'objective'
            # For perturbations, we can get metrics from step_metrics for accepted ones
            # (perturbations that were accepted will have corresponding entries)
            metrics_table = 'step_metrics'
        else:
            raise ValueError(f"Invalid history_type: {history_type}. Must be 'improvements', 'accepted', or 'perturbations'")
        
        # Check if objective value requested
        if 'Objective value' in metric_names:
            # Load objectives from appropriate table
            obj_query = f"""
                SELECT replica_id, perturbation_num, {obj_column} as value
                FROM {obj_table}
                ORDER BY replica_id, perturbation_num
            """
            obj_df = pd.read_sql_query(obj_query, conn)
            if not obj_df.empty:
                obj_df['metric_name'] = 'Objective value'
                all_dfs.append(obj_df)
        
        # Filter to get non-objective metrics
        user_metrics = [m for m in metric_names if m != 'Objective value']
        
        # Load user-defined metrics if metrics table exists for this history type
        if user_metrics and metrics_table:
            placeholders = ','.join(['?' for _ in user_metrics])
            metrics_query = f"SELECT replica_id, perturbation_num, metric_name, value FROM {metrics_table} WHERE metric_name IN ({placeholders}) ORDER BY replica_id, perturbation_num, metric_name"
            metrics_df = pd.read_sql_query(metrics_query, conn, params=user_metrics)
            if not metrics_df.empty:
                all_dfs.append(metrics_df)
        
        if not all_dfs:
            return pd.DataFrame()
        
        # Combine all data
        df = pd.concat(all_dfs, ignore_index=True)
        
        # Downsample per replica to ensure all replicas are represented
        # Group by replica_id and metric_name, then sample rows
        downsampled_dfs = []
        for (replica_id, metric_name), group in df.groupby(['replica_id', 'metric_name']):
            if len(group) > max_points_per_replica:
                # Sample evenly: keep every Nth row
                step_size = len(group) / max_points_per_replica
                indices = [int(i * step_size) for i in range(max_points_per_replica)]
                downsampled_dfs.append(group.iloc[indices])
            else:
                downsampled_dfs.append(group)
        
        return pd.concat(downsampled_dfs, ignore_index=True) if downsampled_dfs else pd.DataFrame()
    except Exception:
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
    except Exception:
        return pd.DataFrame()


def get_available_metrics(conn: sqlite3.Connection, history_type: str = 'improvements') -> List[str]:
    """Get list of all metric names in the database.
    
    Includes 'Objective value' plus all user-defined metrics from the appropriate table
    based on history_type.
    
    Args:
        conn (sqlite3.Connection): SQLite connection.
        history_type (str): Type of history - 'improvements', 'accepted', or 'perturbations'.
            Default is 'improvements'.
        
    Returns:
        List[str]: Sorted list of unique metric names including 'Objective value'.
    """
    metrics = ['Objective value']  # Always include objective
    
    # Determine which metrics table to query
    if history_type == 'improvements':
        metrics_table = 'improvement_metrics'
    elif history_type == 'accepted':
        metrics_table = 'step_metrics'
    elif history_type == 'perturbations':
        # For perturbations, metrics available from step_metrics (for accepted ones)
        metrics_table = 'step_metrics'
    else:
        metrics_table = 'improvement_metrics'  # Default fallback
    
    query = f"SELECT DISTINCT metric_name FROM {metrics_table} ORDER BY metric_name"
    cursor = conn.cursor()
    try:
        cursor.execute(query)
        metrics.extend([row[0] for row in cursor.fetchall()])
    except Exception:
        pass  # Table might not exist or have data yet
    
    return metrics


def get_available_directories() -> List[Path]:
    """Return list of candidate directories for database selection.
    
    Includes:
    - Current working directory
    - Parent directory
    - Immediate subdirectories (non-hidden)
    
    Returns:
        List[Path]: De-duplicated list of directories in deterministic order.
    """
    cwd = Path.cwd()
    dirs = [cwd, cwd.parent]
    
    # Add immediate subdirectories
    try:
        for item in cwd.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                dirs.append(item)
    except PermissionError:
        pass

    # De-duplicate while preserving order
    seen = set()
    unique_dirs = []
    for d in dirs:
        resolved = d.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_dirs.append(resolved)
    
    return unique_dirs


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
    except Exception:
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
    except Exception:
        return {}


def load_temperature_ladder(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load initial temperatures from temperature ladder table.
    
    Args:
        conn (sqlite3.Connection): SQLite connection.
        
    Returns:
        pd.DataFrame: DataFrame with replica_id and temperature columns.
            Falls back to replica_status if temperature_ladder doesn't exist.
    """
    query = """
        SELECT replica_id, temperature 
        FROM temperature_ladder 
        ORDER BY replica_id
    """
    try:
        return pd.read_sql_query(query, conn)
    except Exception:
        # Fallback: try replica_status if temperature_ladder doesn't exist
        try:
            alt_query = "SELECT replica_id, temperature FROM replica_status ORDER BY replica_id"
            return pd.read_sql_query(alt_query, conn)
        except Exception:
            return pd.DataFrame()


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
    except Exception:
        pass
    
    return {'total_perturbations': 0, 'total_accepted': 0}
