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
        db_path_str: Path to the SQLite database file
        
    Returns:
        SQLite connection object
    """
    return sqlite3.connect(
        f"file:{db_path_str}?mode=ro", 
        uri=True, 
        check_same_thread=False
    )


def load_run_metadata(conn: sqlite3.Connection) -> Optional[Dict[str, Any]]:
    """Load run metadata from database.
    
    Args:
        conn: SQLite connection
        
    Returns:
        Dictionary with run metadata or None if not found
    """
    query = "SELECT * FROM run_metadata WHERE run_id = 1"
    cursor = conn.cursor()
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
            'hyperparameters': json.loads(row[6]),
            'checkpoint_file': row[7] if len(row) > 7 else None,
            'objective_function_name': row[8] if len(row) > 8 else None,
            'dataset_size': row[9] if len(row) > 9 else None
        }
    return None


def load_metrics_history(
    conn: sqlite3.Connection,
    metric_names: Optional[List[str]] = None,
    max_points_per_replica: int = 1000
) -> pd.DataFrame:
    """Load metrics history with optional downsampling for performance.
    
    Args:
        conn: SQLite connection
        metric_names: List of metric names to load
        max_points_per_replica: Downsample if more points exist
        
    Returns:
        DataFrame with columns: replica_id, step, metric_name, value
    """
    if not metric_names:
        return pd.DataFrame()

    try:
        # Check total steps to determine if downsampling is needed
        count_query = "SELECT MAX(step) as max_step FROM metrics_history"
        result = pd.read_sql_query(count_query, conn)
        if result.empty or result['max_step'].iloc[0] is None:
            return pd.DataFrame()

        total_steps = int(result['max_step'].iloc[0])
        
        # Build query with optional downsampling
        if total_steps > max_points_per_replica:
            step_interval = max(1, total_steps // max_points_per_replica)
            query = f"SELECT replica_id, step, metric_name, value FROM metrics_history WHERE step % {step_interval} = 0"
        else:
            query = "SELECT replica_id, step, metric_name, value FROM metrics_history"

        # Filter by requested metrics
        placeholders = ','.join(['?' for _ in metric_names])
        query += f" AND metric_name IN ({placeholders}) ORDER BY replica_id, step, metric_name"
        
        return pd.read_sql_query(query, conn, params=metric_names)
    except Exception:
        return pd.DataFrame()


def load_temperature_exchanges(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load temperature exchange events.
    
    Args:
        conn: SQLite connection
        
    Returns:
        DataFrame with columns: step, replica_id, new_temperature, timestamp
    """
    query = """
        SELECT step, replica_id, new_temperature, timestamp
        FROM temperature_exchanges
        ORDER BY step
    """
    try:
        return pd.read_sql_query(query, conn)
    except Exception:
        return pd.DataFrame()


def get_available_metrics(conn: sqlite3.Connection) -> List[str]:
    """Get list of all metric names in the database.
    
    Args:
        conn: SQLite connection
        
    Returns:
        Sorted list of metric names
    """
    query = "SELECT DISTINCT metric_name FROM metrics_history ORDER BY metric_name"
    cursor = conn.cursor()
    cursor.execute(query)
    return [row[0] for row in cursor.fetchall()]


def get_available_directories() -> List[Path]:
    """Return list of candidate directories for database selection.
    
    Includes:
    - Current working directory
    - Parent directory
    - Immediate subdirectories (non-hidden)
    
    Returns:
        De-duplicated list in deterministic order
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
        conn: SQLite connection
        limit: Maximum number of replicas to return
        
    Returns:
        DataFrame with replica_id, best_objective, step, temperature
    """
    query = """
        SELECT replica_id, best_objective, step, temperature
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
        conn: SQLite connection
        
    Returns:
        Dictionary mapping replica_id to temperature
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
        conn: SQLite connection
        
    Returns:
        DataFrame with replica_id and temperature columns
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
    """Load progress statistics including iteration counts and acceptance rates.
    
    Args:
        conn: SQLite connection
        
    Returns:
        Dictionary with total_iterations, total_accepted, and other stats
    """
    query = """
        SELECT 
            SUM(total_iterations) as total_iterations,
            SUM(step) as total_accepted
        FROM replica_status
    """
    try:
        result = pd.read_sql_query(query, conn)
        if not result.empty:
            return {
                'total_iterations': result['total_iterations'].iloc[0] or 0,
                'total_accepted': result['total_accepted'].iloc[0] or 0
            }
    except Exception:
        pass
    
    return {'total_iterations': 0, 'total_accepted': 0}
