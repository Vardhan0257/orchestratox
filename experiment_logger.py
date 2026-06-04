# experiment_logger.py
# Captures intermediate pipeline states for mechanism verification
# and deterministic replay. One import, zero breaking changes.

import json
import sqlite3
import hashlib
import time
from pathlib import Path
from typing import Optional, Any

DB_PATH = "results/experiment_log.db"

def init_db():
    Path("results").mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trials (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            timestamp REAL,
            scenario_id TEXT,
            domain TEXT,
            framework TEXT,
            depth INTEGER,
            trial_idx INTEGER,
            temperature REAL,
            injection_payload TEXT,
            agent1_input TEXT,
            agent1_output TEXT,
            agent2_input TEXT,
            agent2_output TEXT,
            orchestrator_input TEXT,
            orchestrator_output TEXT,
            target_action TEXT,
            string_match_success INTEGER,
            mistral_judge_success INTEGER,
            task TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_trial(run_id: str, data: dict):
    """Log a complete trial with all intermediate states."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO trials (
            run_id, timestamp, scenario_id, domain, framework, depth,
            trial_idx, temperature, injection_payload,
            agent1_input, agent1_output, agent2_input, agent2_output,
            orchestrator_input, orchestrator_output,
            target_action, string_match_success, mistral_judge_success, task
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        run_id, time.time(),
        data.get("scenario_id", ""), data.get("domain", ""),
        data.get("framework", ""), data.get("depth", 0),
        data.get("trial_idx", 0), data.get("temperature", 0.1),
        data.get("injection_payload", ""),
        data.get("agent1_input", ""), data.get("agent1_output", ""),
        data.get("agent2_input", ""), data.get("agent2_output", ""),
        data.get("orchestrator_input", ""), data.get("orchestrator_output", ""),
        data.get("target_action", ""),
        int(data.get("string_match_success", False)),
        int(data.get("mistral_judge_success", False)),
        data.get("task", "")
    ))
    conn.commit()
    conn.close()

def get_trials_for_content_analysis(scenario_id: str, depth: int = 2) -> list:
    """Retrieve logged trials for content analysis post-hoc."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT agent1_input, agent1_output, agent2_input, agent2_output,
               injection_payload, orchestrator_output, target_action, task
        FROM trials
        WHERE scenario_id = ? AND depth = ? AND agent1_output != ''
        LIMIT 50
    """, (scenario_id, depth)).fetchall()
    conn.close()
    
    return [
        {
            "agent1_input": r[0], "agent1_output": r[1],
            "agent2_input": r[2], "agent2_output": r[3],
            "injection_payload": r[4], "orchestrator_output": r[5],
            "target_action": r[6], "task": r[7]
        }
        for r in rows
    ]

# Initialize on import
init_db()