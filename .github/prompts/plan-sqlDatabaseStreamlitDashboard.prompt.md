# Plan: SQL Database for Real-Time Progress with Tunable Pooling

## TL;DR
Use SQLite with fully configurable pooled writes—tune step collection frequency (e.g., every Nth step), buffer size before write (e.g., 10 pooled steps), and pool sizes independently. Keep full history for current run only, reinitialize database on new run start. Accept partial batch loss on crashes—checkpoints provide recovery points.

## Steps

### 1. Create lean database schema with connection pooling
Add `hill_climber/database.py` with SQLAlchemy models: `run_metadata` table (run_id, start_time, hyperparameters JSON), `replica_status` table (replica_id, step, temperature, best_objective, current_objective, timestamp), `metrics_history` table (replica_id, step, metric_name, value), `temperature_exchanges` table (step, replica_id, new_temperature). Configure SQLite with `PRAGMA journal_mode=WAL`, `synchronous=NORMAL`. Add `DatabaseWriter` class with configurable connection pool size (default 4) using `QueuePool` from SQLAlchemy.

### 2. Add tunable pooling parameters to HillClimber
Extend `HillClimber.__init__()` with database configuration: `db_path` (optional, defaults to checkpoint_file.replace('.pkl', '.db')), `db_enabled` (default False), `db_step_interval` (collect every Nth step, default `max(1, exchange_interval // 1000)`), `db_buffer_size` (pooled steps before write, default 10), `db_connection_pool_size` (SQLAlchemy pool size, default 4). Store these in instance variables and pass to workers.

### 3. Implement pooled write buffer in workers
Modify `replica_worker.py.run_replica_steps()` to accept database config dict. Add in-memory buffer: check `if step % db_step_interval == 0` → collect metrics into list. When buffer reaches `db_buffer_size` → execute batch insert using `executemany()` and clear buffer. Return unflushed buffer contents with worker results. Make database writes optional—only execute if `db_enabled=True`.

### 4. Initialize/reinitialize database on run start
Add `_initialize_database()` method in `HillClimber` that: checks if database exists → if yes, drop all tables (clean slate), create fresh schema, insert run_metadata row with timestamp and hyperparameters JSON. Called at start of `climb()` before worker pool creation. On `load_checkpoint()`, call `_initialize_database()` then optionally repopulate from checkpoint history if `sync_database=True` parameter.

### 5. Flush worker buffers and sync after batch
In `optimizer.py._parallel_step_batch()`, after collecting worker results: extract unflushed buffers from each worker → batch insert all remaining pooled metrics to database. Then write batch summary: update `replica_status` table with current state for all replicas, insert temperature exchanges if exchange round occurred. Single-threaded writes from main process avoid worker contention.

### 6. Decouple checkpoint frequency
Add `checkpoint_interval` parameter (batches between checkpoints, default 10). Modify `_checkpoint_and_plot()` to track batch counter and `if batch_count % checkpoint_interval == 0: save_checkpoint()`. Database provides fine-grained progress (~every 1000 steps with defaults), checkpoints provide recovery every N batches (e.g., every 100k steps with defaults).

### 7. Build Streamlit dashboard with tunable refresh
Create `streamlit_app.py` with sidebar config: refresh_interval (seconds, default 3), metrics_to_plot (multi-select). Connect to SQLite read-only. Auto-refresh queries: current state from `replica_status`, time series from `metrics_history` with Plotly line charts, temperature timeline from `temperature_exchanges`. Display database config info (step_interval, buffer_size) from `run_metadata`. Add manual refresh button and auto-pause option.

## Further Considerations

### 1. Recommended pool size for different replica counts?
Default `db_step_interval = max(1, exchange_interval // 1000)` gives ~1000 pooled steps per batch. With `buffer_size=10`, that's ~100 writes per worker per batch. For 4 replicas: `pool_size=2` sufficient. For 16 replicas: `pool_size=4-8`. Could auto-scale: `pool_size = min(n_replicas // 2, 8)`.

### 2. Memory vs I/O tradeoff tuning?
Larger `buffer_size` reduces writes but increases memory and data loss on crash. Smaller `db_step_interval` gives finer dashboard resolution but more database size. Recommended defaults balance well: `step_interval=exchange_interval//1000` (0.1% sampling), `buffer_size=10` (typical flush every ~10k steps for 100k batch).

### 3. Database size estimation?
With defaults: 10k batch, 8 replicas, 20 metrics, `step_interval=10`, `buffer_size=10` → ~1000 steps collected per replica → 160k metric rows per batch. For 30-min run (~100 batches) → 16M rows → ~1-2GB database. Acceptable for modern systems. Users can increase `step_interval` if concerned (e.g., `exchange_interval//500` → 0.2% sampling → half the size).
