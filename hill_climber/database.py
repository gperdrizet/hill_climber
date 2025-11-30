"""Database module for storing optimization progress for real-time monitoring."""

import json
import sqlite3
import time
from contextlib import contextmanager
from typing import Dict, List, Optional, Any
import threading


class DatabaseWriter:
    """Thread-safe SQLite database writer for optimization progress.
    
    Uses SQLite WAL mode for concurrent read/write access without
    explicit connection pooling.
    """

    def __init__(self, db_path: str):
        """Initialize database writer.
        
        Args:
            db_path (str): Path to SQLite database file.
        """

        self.db_path = db_path
        self._lock = threading.Lock()


    @contextmanager
    def get_connection(self):
        """Context manager for database connections.
        
        Enables WAL mode for concurrent read/write access.
        
        Yields:
            sqlite3.Connection: Database connection with WAL mode enabled.
        """

        conn = sqlite3.connect(self.db_path, timeout=30.0)
        
        try:
            # Enable WAL mode for concurrent reads during writes
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")

            yield conn

            conn.commit()

        except Exception as e:
            conn.rollback()
            raise e

        finally:
            conn.close()


    def initialize_schema(self, drop_existing: bool = True):
        """Create database schema.
        
        Args:
            drop_existing (bool): If True, drop existing tables first. Default is True.
        """

        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if drop_existing:
                cursor.execute("DROP TABLE IF EXISTS temperature_exchanges")
                cursor.execute("DROP TABLE IF EXISTS metrics_history")
                cursor.execute("DROP TABLE IF EXISTS replica_status")
                cursor.execute("DROP TABLE IF EXISTS run_metadata")
            
            # Run metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS run_metadata (
                    run_id INTEGER PRIMARY KEY,
                    start_time REAL NOT NULL,
                    n_replicas INTEGER NOT NULL,
                    exchange_interval INTEGER NOT NULL,
                    db_step_interval INTEGER NOT NULL,
                    db_buffer_size INTEGER NOT NULL,
                    hyperparameters TEXT NOT NULL,
                    checkpoint_file TEXT,
                    objective_function_name TEXT,
                    dataset_size INTEGER
                )
            """)
            
            # Replica status table (current state)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS replica_status (
                    replica_id INTEGER PRIMARY KEY,
                    step INTEGER NOT NULL,
                    total_iterations INTEGER NOT NULL DEFAULT 0,
                    temperature REAL NOT NULL,
                    best_objective REAL NOT NULL,
                    current_objective REAL NOT NULL,
                    timestamp REAL NOT NULL
                )
            """)
            
            # Create index for leaderboard queries (optimized for ORDER BY best_objective DESC)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_replica_status_objective 
                ON replica_status(best_objective DESC)
            """)
            
            # Metrics history table (time series)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    replica_id INTEGER NOT NULL,
                    step INTEGER NOT NULL,
                    metric_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    UNIQUE(replica_id, step, metric_name)
                )
            """)
            
            # Create indexes for faster time series queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_replica_step 
                ON metrics_history(replica_id, step)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_name
                ON metrics_history(metric_name)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_step
                ON metrics_history(step)
            """)
            
            # Temperature exchanges table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS temperature_exchanges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    step INTEGER NOT NULL,
                    replica_id INTEGER NOT NULL,
                    new_temperature REAL NOT NULL,
                    timestamp REAL NOT NULL
                )
            """)
            
            # Create index for temperature history queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_temp_exchanges_step 
                ON temperature_exchanges(step)
            """)


    def insert_run_metadata(self, n_replicas: int, exchange_interval: int,
                           db_step_interval: int, db_buffer_size: int,
                           hyperparameters: Dict[str, Any], checkpoint_file: str = None,
                           objective_function_name: str = None, dataset_size: int = None):
        """Insert run metadata.
        
        Args:
            n_replicas (int): Number of replicas.
            exchange_interval (int): Steps between exchange attempts.
            db_step_interval (int): Steps between metric collection.
            db_buffer_size (int): Buffer size before database write.
            hyperparameters (Dict[str, Any]): Dictionary of hyperparameters.
            checkpoint_file (str, optional): Path to checkpoint file. Default is None.
            objective_function_name (str, optional): Name of objective function. 
                Default is None.
            dataset_size (int, optional): Total size of input dataset. Default is None.
        """

        with self.get_connection() as conn:

            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO run_metadata 
                (run_id, start_time, n_replicas, exchange_interval, 
                 db_step_interval, db_buffer_size, hyperparameters, checkpoint_file, objective_function_name, dataset_size)
                VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                time.time(),
                n_replicas,
                exchange_interval,
                db_step_interval,
                db_buffer_size,
                json.dumps(hyperparameters),
                checkpoint_file,
                objective_function_name,
                dataset_size
            ))


    def update_replica_status(self, replica_id: int, step: int,
                             total_iterations: int,
                             temperature: float, best_objective: float,
                             current_objective: float):
        """Update current replica status.
        
        Args:
            replica_id (int): Replica ID.
            step (int): Current step number (accepted steps).
            total_iterations (int): Total iterations attempted (all perturbations).
            temperature (float): Current temperature.
            best_objective (float): Best objective value found.
            current_objective (float): Current objective value.
        """

        with self.get_connection() as conn:

            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO replica_status
                (replica_id, step, total_iterations, temperature, best_objective, current_objective, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (replica_id, step, total_iterations, temperature, best_objective, current_objective, time.time()))


    def insert_metrics_batch(self, metrics_data: List[tuple]):
        """Insert batch of metrics.
        
        Args:
            metrics_data (List[tuple]): List of tuples with format 
                (replica_id, step, metric_name, value).
        """

        if not metrics_data:
            return
            
        with self._lock:
            with self.get_connection() as conn:

                cursor = conn.cursor()

                cursor.executemany("""
                    INSERT OR REPLACE INTO metrics_history
                    (replica_id, step, metric_name, value)
                    VALUES (?, ?, ?, ?)
                """, metrics_data)


    def insert_temperature_exchanges(self, exchanges: List[tuple]):
        """Insert temperature exchange records.
        
        Args:
            exchanges (List[tuple]): List of tuples with format 
                (step, replica_id, new_temperature).
        """

        if not exchanges:
            return
            
        with self.get_connection() as conn:

            cursor = conn.cursor()
            timestamp = time.time()

            cursor.executemany("""
                INSERT INTO temperature_exchanges
                (step, replica_id, new_temperature, timestamp)
                VALUES (?, ?, ?, ?)
            """, [(step, rid, temp, timestamp) for step, rid, temp in exchanges])


    def get_run_metadata(self) -> Optional[Dict[str, Any]]:
        """Get run metadata.
        
        Returns:
            Dict[str, Any]: Dictionary with run metadata, or None if not found.
        """

        with self.get_connection() as conn:

            cursor = conn.cursor()
            cursor.execute("SELECT * FROM run_metadata WHERE run_id = 1")
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


    def get_replica_status(self) -> List[Dict[str, Any]]:
        """Get current status of all replicas.
        
        Returns:
            List[Dict[str, Any]]: List of dictionaries with replica status, sorted by
                replica_id. Each dict contains replica_id, step, temperature, 
                best_objective, current_objective, and timestamp.
        """

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT replica_id, step, temperature, best_objective, 
                       current_objective, timestamp
                FROM replica_status
                ORDER BY replica_id
            """)
            
            return [{
                'replica_id': row[0],
                'step': row[1],
                'temperature': row[2],
                'best_objective': row[3],
                'current_objective': row[4],
                'timestamp': row[5]
            } for row in cursor.fetchall()]


    def get_metrics_history(
            self, replica_id: Optional[int] = None,
            min_step: Optional[int] = None,
            max_step: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get metrics history.
        
        Args:
            replica_id (int, optional): Filter by replica ID. Default is None (all replicas).
            min_step (int, optional): Minimum step. Default is None (no minimum).
            max_step (int, optional): Maximum step. Default is None (no maximum).
            
        Returns:
            List[Dict[str, Any]]: List of dictionaries with metrics history, sorted by
                replica_id, step, and metric_name. Each dict contains replica_id, step,
                metric_name, and value.
        """

        with self.get_connection() as conn:

            cursor = conn.cursor()
            
            query = "SELECT replica_id, step, metric_name, value FROM metrics_history WHERE 1=1"
            params = []
            
            if replica_id is not None:
                query += " AND replica_id = ?"
                params.append(replica_id)

            if min_step is not None:
                query += " AND step >= ?"
                params.append(min_step)

            if max_step is not None:
                query += " AND step <= ?"
                params.append(max_step)
            
            query += " ORDER BY replica_id, step, metric_name"
            
            cursor.execute(query, params)
            
            return [{
                'replica_id': row[0],
                'step': row[1],
                'metric_name': row[2],
                'value': row[3]
            } for row in cursor.fetchall()]


    def get_temperature_exchanges(self) -> List[Dict[str, Any]]:
        """Get all temperature exchange records.
        
        Returns:
            List[Dict[str, Any]]: List of dictionaries with temperature exchanges, sorted
                by step. Each dict contains step, replica_id, new_temperature, and timestamp.
        """

        with self.get_connection() as conn:

            cursor = conn.cursor()

            cursor.execute("""
                SELECT step, replica_id, new_temperature, timestamp
                FROM temperature_exchanges
                ORDER BY step
            """)
            
            return [{
                'step': row[0],
                'replica_id': row[1],
                'new_temperature': row[2],
                'timestamp': row[3]
            } for row in cursor.fetchall()]
