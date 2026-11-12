from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from tas_copilot_contracts import ReviewLabel, ReviewLabelBatch


SCHEMA_VERSION = 1


class ReviewOutcomeSQLiteStore:
    """SQLite store for human review labels and calibration outcomes.

    This catalog is intentionally separate from both raw TAS data catalogs and
    paper corpus catalogs. It stores labels about candidate claims, not raw data
    or private source text.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        return connection

    def _initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS review_outcome_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS review_label_batches (
                    batch_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    decision_ontology_version TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    source_template_paths_json TEXT NOT NULL,
                    label_count INTEGER NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS review_labels (
                    label_id TEXT PRIMARY KEY,
                    batch_id TEXT,
                    object_type TEXT NOT NULL,
                    object_id TEXT NOT NULL,
                    object_secondary_id TEXT,
                    source_workflow TEXT NOT NULL,
                    source_run_id TEXT,
                    source_artifact_path TEXT,
                    source_artifact_hash TEXT,
                    verdict TEXT NOT NULL,
                    type_specific_status TEXT,
                    claim_state TEXT NOT NULL,
                    recommendation TEXT NOT NULL,
                    uncertainty_class TEXT NOT NULL,
                    reviewer_id TEXT,
                    reviewer_role TEXT,
                    reviewed_at TEXT,
                    workflow_version TEXT,
                    calibration_state_hash TEXT,
                    dag_run_id TEXT,
                    cache_key TEXT,
                    cache_scope TEXT,
                    payload_json TEXT NOT NULL,
                    FOREIGN KEY (batch_id) REFERENCES review_label_batches(batch_id)
                );

                CREATE INDEX IF NOT EXISTS idx_review_labels_object ON review_labels(object_type, object_id);
                CREATE INDEX IF NOT EXISTS idx_review_labels_verdict ON review_labels(verdict);
                CREATE INDEX IF NOT EXISTS idx_review_labels_workflow ON review_labels(source_workflow);
                CREATE INDEX IF NOT EXISTS idx_review_labels_reviewer ON review_labels(reviewer_id);
                CREATE INDEX IF NOT EXISTS idx_review_labels_dag ON review_labels(dag_run_id, cache_key);
                """
            )
            connection.execute(
                "INSERT OR REPLACE INTO review_outcome_meta(key, value) VALUES (?, ?)",
                ("schema_version", str(SCHEMA_VERSION)),
            )

    def upsert_batch(self, batch: ReviewLabelBatch) -> None:
        payload = _json_payload(batch)
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO review_label_batches
                (batch_id, created_at, decision_ontology_version, schema_version,
                 source_template_paths_json, label_count, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    batch.batch_id,
                    batch.created_at.isoformat(),
                    batch.decision_ontology_version,
                    batch.schema_version,
                    json.dumps(batch.source_template_paths, sort_keys=True),
                    len(batch.labels),
                    payload,
                ),
            )
            self._upsert_labels(connection, batch.labels, batch_id=batch.batch_id)

    def upsert_labels(self, labels: Iterable[ReviewLabel], batch_id: str | None = None) -> None:
        with self.connect() as connection:
            self._upsert_labels(connection, labels, batch_id=batch_id)

    def _upsert_labels(
        self,
        connection: sqlite3.Connection,
        labels: Iterable[ReviewLabel],
        batch_id: str | None = None,
    ) -> None:
        connection.executemany(
            """
            INSERT OR REPLACE INTO review_labels
            (label_id, batch_id, object_type, object_id, object_secondary_id,
             source_workflow, source_run_id, source_artifact_path, source_artifact_hash,
             verdict, type_specific_status, claim_state, recommendation, uncertainty_class,
             reviewer_id, reviewer_role, reviewed_at, workflow_version,
             calibration_state_hash, dag_run_id, cache_key, cache_scope, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    label.label_id,
                    batch_id,
                    label.object_type,
                    label.object_id,
                    label.object_secondary_id,
                    label.source_workflow,
                    label.source_run_id,
                    label.source_artifact_path,
                    label.source_artifact_hash,
                    label.verdict,
                    label.type_specific_status,
                    label.claim_state,
                    label.recommendation,
                    label.uncertainty_class,
                    label.reviewer_id,
                    label.reviewer_role,
                    label.reviewed_at.isoformat() if label.reviewed_at else None,
                    label.workflow_version,
                    label.calibration_state_hash,
                    label.dag_run_id,
                    label.cache_key,
                    label.cache_scope,
                    _json_payload(label),
                )
                for label in labels
            ],
        )

    def list_labels(
        self,
        *,
        object_type: str | None = None,
        object_id: str | None = None,
        verdict: str | None = None,
        source_workflow: str | None = None,
        limit: int | None = None,
    ) -> list[ReviewLabel]:
        clauses: list[str] = []
        params: list[Any] = []
        for column, value in [
            ("object_type", object_type),
            ("object_id", object_id),
            ("verdict", verdict),
            ("source_workflow", source_workflow),
        ]:
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        limit_sql = "LIMIT ?" if limit is not None else ""
        if limit is not None:
            params.append(limit)
        with self.connect() as connection:
            rows = connection.execute(
                f"SELECT payload_json FROM review_labels {where} ORDER BY label_id {limit_sql}",
                params,
            ).fetchall()
        return [ReviewLabel.model_validate(json.loads(row["payload_json"])) for row in rows]

    def summary(self) -> dict[str, Any]:
        with self.connect() as connection:
            label_count = connection.execute("SELECT COUNT(*) FROM review_labels").fetchone()[0]
            batch_count = connection.execute("SELECT COUNT(*) FROM review_label_batches").fetchone()[0]
            reviewed_count = connection.execute(
                "SELECT COUNT(*) FROM review_labels WHERE verdict != 'unreviewed'"
            ).fetchone()[0]
            return {
                "schema_version": SCHEMA_VERSION,
                "label_count": label_count,
                "batch_count": batch_count,
                "reviewed_label_count": reviewed_count,
                "by_object_type": _count_by(connection, "object_type"),
                "by_verdict": _count_by(connection, "verdict"),
                "by_source_workflow": _count_by(connection, "source_workflow"),
                "by_uncertainty_class": _count_by(connection, "uncertainty_class"),
            }


def _count_by(connection: sqlite3.Connection, column: str) -> dict[str, int]:
    rows = connection.execute(
        f"SELECT {column} AS key, COUNT(*) AS count FROM review_labels GROUP BY {column} ORDER BY {column}"
    ).fetchall()
    return {row["key"]: row["count"] for row in rows if row["key"] is not None}


def _json_payload(model: ReviewLabel | ReviewLabelBatch) -> str:
    return json.dumps(model.model_dump(mode="json"), sort_keys=True)
