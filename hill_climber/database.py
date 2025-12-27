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
                cursor.execute("DROP TABLE IF EXISTS temperature_ladder_history")
                cursor.execute("DROP TABLE IF EXISTS batch_statistics")
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
            
            # Composite indexes for dashboard performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_improvement_metrics_composite
                ON improvement_metrics(replica_id, metric_name, perturbation_num)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_step_metrics_composite
                ON step_metrics(replica_id, metric_name, perturbation_num)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_perturbations_composite
                ON perturbations(replica_id, perturbation_num, objective)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_accepted_composite
                ON accepted_steps(replica_id, perturbation_num, objective)
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
            
            # Temperature ladder history table - tracks temperature at each ladder position over time
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS temperature_ladder_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_num INTEGER NOT NULL,
                    ladder_position INTEGER NOT NULL,
                    temperature REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    UNIQUE(batch_num, ladder_position)
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ladder_history_batch
                ON temperature_ladder_history(batch_num, ladder_position)
            """)
            
            # Batch statistics table - tracks step spread and acceptance rates
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS batch_statistics (
                    batch_num INTEGER PRIMARY KEY,
                    step_spread REAL NOT NULL,
                    mean_acceptance_rate REAL NOT NULL,
                    min_acceptance_rate REAL NOT NULL,
                    max_acceptance_rate REAL NOT NULL,
                    timestamp REAL NOT NULL
                )
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


    def initialize_temperature_ladder_history(self, ladder_temps: List[float], batch_num: int = 0):
        """Initialize temperature ladder history with starting temperatures.
        
        Args:
            ladder_temps (List[float]): List of temperatures sorted from high to low.
            batch_num (int): Batch number (default 0 for initialization).
        """
        if not ladder_temps:
            return
            
        with self.get_connection() as conn:
            cursor = conn.cursor()
            timestamp = time.time()
            
            for position, temp in enumerate(ladder_temps):
                cursor.execute("""
                    INSERT OR REPLACE INTO temperature_ladder_history
                    (batch_num, ladder_position, temperature, timestamp)
                    VALUES (?, ?, ?, ?)
                """, (batch_num, position, temp, timestamp))


    def update_temperature_ladder_history(self, batch_num: int, cooling_rate: float, exchange_interval: int):
        """Update temperature ladder history by applying cooling to previous batch temperatures.
        
        Args:
            batch_num (int): Current batch number.
            cooling_rate (float): Temperature decay rate per step.
            exchange_interval (int): Number of steps per batch.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            timestamp = time.time()
            
            # Get temperatures from previous batch
            cursor.execute("""
                SELECT ladder_position, temperature
                FROM temperature_ladder_history
                WHERE batch_num = ?
                ORDER BY ladder_position
            """, (batch_num - 1,))
            
            prev_temps = cursor.fetchall()
            
            if not prev_temps:
                return
            
            # Apply cooling and insert new batch
            cooling_factor = (1 - cooling_rate) ** exchange_interval
            for position, prev_temp in prev_temps:
                new_temp = prev_temp * cooling_factor
                cursor.execute("""
                    INSERT OR REPLACE INTO temperature_ladder_history
                    (batch_num, ladder_position, temperature, timestamp)
                    VALUES (?, ?, ?, ?)
                """, (batch_num, position, new_temp, timestamp))


    def insert_batch_statistics(self, batch_num: int, step_spread: float, 
                                mean_acceptance_rate: float, min_acceptance_rate: float, max_acceptance_rate: float):
        """Insert batch statistics for step spread and acceptance rates.
        
        Args:
            batch_num (int): Batch number.
            step_spread (float): Current step spread value.
            mean_acceptance_rate (float): Mean acceptance rate across replicas.
            min_acceptance_rate (float): Minimum acceptance rate across replicas.
            max_acceptance_rate (float): Maximum acceptance rate across replicas.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            timestamp = time.time()
            
            cursor.execute("""
                INSERT OR REPLACE INTO batch_statistics
                (batch_num, step_spread, mean_acceptance_rate, min_acceptance_rate, max_acceptance_rate, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (batch_num, step_spread, mean_acceptance_rate, min_acceptance_rate, max_acceptance_rate, timestamp))

