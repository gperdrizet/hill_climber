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
                # Drop old tables
                cursor.execute("DROP TABLE IF EXISTS temperature_exchanges")
                cursor.execute("DROP TABLE IF EXISTS metrics_history")
                cursor.execute("DROP TABLE IF EXISTS replica_status")
                cursor.execute("DROP TABLE IF EXISTS run_metadata")
                # Drop new tables
                cursor.execute("DROP TABLE IF EXISTS perturbations")
                cursor.execute("DROP TABLE IF EXISTS accepted_steps")
                cursor.execute("DROP TABLE IF EXISTS step_metrics")
                cursor.execute("DROP TABLE IF EXISTS improvements")
                cursor.execute("DROP TABLE IF EXISTS improvement_metrics")
            
            # Run metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS run_metadata (
                    run_id INTEGER PRIMARY KEY,
                    start_time REAL NOT NULL,
                    end_time REAL,
                    n_replicas INTEGER NOT NULL,
                    exchange_interval INTEGER NOT NULL,
                    db_step_interval INTEGER NOT NULL,
                    hyperparameters TEXT NOT NULL,
                    checkpoint_file TEXT,
                    objective_function_name TEXT,
                    dataset_size INTEGER
                )
            """)
            
            # ===== NEW SCHEMA =====
            
            # All perturbations evaluated (sampled at db_step_interval)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS perturbations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    replica_id INTEGER NOT NULL,
                    perturbation_num INTEGER NOT NULL,
                    objective REAL NOT NULL,
                    is_accepted BOOLEAN NOT NULL,
                    is_improvement BOOLEAN NOT NULL,
                    temperature REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    UNIQUE(replica_id, perturbation_num)
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_perturbations_replica_num
                ON perturbations(replica_id, perturbation_num)
            """)
            
            # Accepted perturbations (all SA-accepted moves)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS accepted_steps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    replica_id INTEGER NOT NULL,
                    perturbation_num INTEGER NOT NULL,
                    objective REAL NOT NULL,
                    temperature REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    UNIQUE(replica_id, perturbation_num)
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_accepted_replica_num
                ON accepted_steps(replica_id, perturbation_num)
            """)
            
            # Metrics for accepted steps
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS step_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    replica_id INTEGER NOT NULL,
                    perturbation_num INTEGER NOT NULL,
                    metric_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    UNIQUE(replica_id, perturbation_num, metric_name)
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_step_metrics_replica_num
                ON step_metrics(replica_id, perturbation_num)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_step_metrics_name
                ON step_metrics(metric_name)
            """)
            
            # New best solutions (all improvements)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS improvements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    replica_id INTEGER NOT NULL,
                    perturbation_num INTEGER NOT NULL,
                    best_objective REAL NOT NULL,
                    temperature REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    UNIQUE(replica_id, perturbation_num)
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_improvements_replica_num
                ON improvements(replica_id, perturbation_num)
            """)
            
            # Metrics for improvements
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS improvement_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    replica_id INTEGER NOT NULL,
                    perturbation_num INTEGER NOT NULL,
                    metric_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    UNIQUE(replica_id, perturbation_num, metric_name)
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_improvement_metrics_replica_num
                ON improvement_metrics(replica_id, perturbation_num)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_improvement_metrics_name
                ON improvement_metrics(metric_name)
            """)
            
            # Current replica state (snapshot updated after each batch)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS replica_status (
                    replica_id INTEGER PRIMARY KEY,
                    current_perturbation_num INTEGER NOT NULL,
                    num_accepted INTEGER NOT NULL,
                    num_improvements INTEGER NOT NULL,
                    best_objective REAL NOT NULL,
                    current_objective REAL NOT NULL,
                    temperature REAL NOT NULL,
                    timestamp REAL NOT NULL
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_replica_status_objective 
                ON replica_status(best_objective DESC)
            """)
            
            # Temperature exchanges table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS temperature_exchanges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    perturbation_num INTEGER NOT NULL,
                    replica_id INTEGER NOT NULL,
                    new_temperature REAL NOT NULL,
                    timestamp REAL NOT NULL
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_temp_exchanges_num
                ON temperature_exchanges(perturbation_num)
            """)


    def insert_run_metadata(self, n_replicas: int, exchange_interval: int,
                           db_step_interval: int,
                           hyperparameters: Dict[str, Any], checkpoint_file: str = None,
                           objective_function_name: str = None, dataset_size: int = None):
        """Insert run metadata.
        
        Args:
            n_replicas (int): Number of replicas.
            exchange_interval (int): Steps between exchange attempts.
            db_step_interval (int): Steps between metric collection.
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
                 db_step_interval, hyperparameters, checkpoint_file, objective_function_name, dataset_size)
                VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                time.time(),
                n_replicas,
                exchange_interval,
                db_step_interval,
                json.dumps(hyperparameters),
                checkpoint_file,
                objective_function_name,
                dataset_size
            ))


    def set_run_end_time(self):
        """Set the end time for the optimization run.
        
        Should be called when the optimization completes to mark the
        completion time for accurate elapsed time calculation.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE run_metadata
                SET end_time = ?
                WHERE run_id = 1
            """, (time.time(),))


    def update_replica_status(self, replica_id: int, current_perturbation_num: int,
                             num_accepted: int, num_improvements: int,
                             temperature: float, best_objective: float,
                             current_objective: float):
        """Update current replica status.
        
        Args:
            replica_id (int): Replica ID.
            current_perturbation_num (int): Current perturbation number.
            num_accepted (int): Number of accepted steps.
            num_improvements (int): Number of improvements found.
            temperature (float): Current temperature.
            best_objective (float): Best objective value found.
            current_objective (float): Current objective value.
        """

        with self.get_connection() as conn:

            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO replica_status
                (replica_id, current_perturbation_num, num_accepted, num_improvements,
                 temperature, best_objective, current_objective, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (replica_id, current_perturbation_num, num_accepted, num_improvements,
                  temperature, best_objective, current_objective, time.time()))
    
    
    def insert_perturbations_batch(self, perturbations_data: List[tuple]):
        """Insert batch of perturbation records.
        
        Args:
            perturbations_data (List[tuple]): List of tuples with format 
                (replica_id, perturbation_num, objective, is_accepted, is_improvement, temperature, timestamp).
        """

        if not perturbations_data:
            return
            
        with self._lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.executemany("""
                    INSERT OR REPLACE INTO perturbations
                    (replica_id, perturbation_num, objective, is_accepted, is_improvement, temperature, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, perturbations_data)
    
    
    def insert_accepted_steps_batch(self, accepted_data: List[tuple]):
        """Insert batch of accepted step records.
        
        Args:
            accepted_data (List[tuple]): List of tuples with format 
                (replica_id, perturbation_num, objective, temperature, timestamp).
        """

        if not accepted_data:
            return
            
        with self._lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.executemany("""
                    INSERT OR REPLACE INTO accepted_steps
                    (replica_id, perturbation_num, objective, temperature, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, accepted_data)
    
    
    def insert_step_metrics_batch(self, metrics_data: List[tuple]):
        """Insert batch of step metrics.
        
        Args:
            metrics_data (List[tuple]): List of tuples with format 
                (replica_id, perturbation_num, metric_name, value).
        """

        if not metrics_data:
            return
            
        with self._lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.executemany("""
                    INSERT OR REPLACE INTO step_metrics
                    (replica_id, perturbation_num, metric_name, value)
                    VALUES (?, ?, ?, ?)
                """, metrics_data)
    
    
    def insert_improvements_batch(self, improvements_data: List[tuple]):
        """Insert batch of improvement records.
        
        Args:
            improvements_data (List[tuple]): List of tuples with format 
                (replica_id, perturbation_num, best_objective, temperature, timestamp).
        """

        if not improvements_data:
            return
            
        with self._lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.executemany("""
                    INSERT OR REPLACE INTO improvements
                    (replica_id, perturbation_num, best_objective, temperature, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, improvements_data)
    
    
    def insert_improvement_metrics_batch(self, metrics_data: List[tuple]):
        """Insert batch of improvement metrics.
        
        Args:
            metrics_data (List[tuple]): List of tuples with format 
                (replica_id, perturbation_num, metric_name, value).
        """

        if not metrics_data:
            return
            
        with self._lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.executemany("""
                    INSERT OR REPLACE INTO improvement_metrics
                    (replica_id, perturbation_num, metric_name, value)
                    VALUES (?, ?, ?, ?)
                """, metrics_data)


    def insert_temperature_exchanges(self, exchanges: List[tuple]):
        """Insert temperature exchange records.
        
        Args:
            exchanges (List[tuple]): List of tuples with format 
                (perturbation_num, replica_id, new_temperature).
        """

        if not exchanges:
            return
            
        with self.get_connection() as conn:

            cursor = conn.cursor()
            timestamp = time.time()

            cursor.executemany("""
                INSERT INTO temperature_exchanges
                (perturbation_num, replica_id, new_temperature, timestamp)
                VALUES (?, ?, ?, ?)
            """, [(int(pnum), int(rid), float(temp), timestamp) for pnum, rid, temp in exchanges])


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
                replica_id. Each dict contains replica_id, current_perturbation_num, 
                temperature, best_objective, current_objective, and timestamp.
        """

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT replica_id, current_perturbation_num, temperature, best_objective, 
                       current_objective, timestamp
                FROM replica_status
                ORDER BY replica_id
            """)
            
            return [{
                'replica_id': row[0],
                'current_perturbation_num': row[1],
                'temperature': row[2],
                'best_objective': row[3],
                'current_objective': row[4],
                'timestamp': row[5]
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
