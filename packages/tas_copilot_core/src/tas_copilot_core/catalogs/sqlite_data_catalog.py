from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any
from typing import Iterable

from tas_copilot_contracts import (
    DataFile,
    DerivedDataFile,
    Experiment,
    PaperExperimentLink,
    Scan,
)


SCHEMA_VERSION = 2


class SQLiteDataCatalog:
    """SQLite catalog for BT-7 data files and data-derived metadata.

    This intentionally does not store the paper corpus. Paper IDs in mappings are
    references to an external corpus/retrieval store.
    """

    def __init__(self, path: str | Path, archive_root: str | Path | None = None) -> None:
        """``archive_root`` makes the catalog portable.

        Stored paths are absolute, which pins a catalog to the machine that
        built it: the full BT-7 catalog holds 171,935 absolute paths, so on
        Vista or at NIST every row resolves to nothing - indistinguishable from
        an empty archive. Recording a root-relative path alongside lets any
        machine resolve the same catalog against its own copy.
        """
        self.archive_root = Path(archive_root).expanduser().resolve() if archive_root else None
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()
        with self.connect() as connection:
            self._migrate_relative_path(connection)

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
                CREATE TABLE IF NOT EXISTS catalog_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    instrument TEXT NOT NULL,
                    facility TEXT,
                    native_experiment_id TEXT,
                    source_adapter TEXT,
                    proposal_id TEXT,
                    sample_id TEXT,
                    title TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    payload_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS scans (
                    scan_id TEXT PRIMARY KEY,
                    experiment_id TEXT,
                    instrument TEXT NOT NULL,
                    facility TEXT,
                    native_scan_id TEXT,
                    source_adapter TEXT,
                    scan_type TEXT,
                    physics_target TEXT,
                    payload_json TEXT NOT NULL,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
                );

                CREATE TABLE IF NOT EXISTS data_files (
                    datafile_id TEXT PRIMARY KEY,
                    path TEXT NOT NULL,
                    relative_path TEXT,
                    checksum TEXT,
                    scan_id TEXT,
                    experiment_id TEXT,
                    format TEXT,
                    facility TEXT,
                    instrument TEXT,
                    native_format TEXT,
                    native_scan_id TEXT,
                    native_experiment_id TEXT,
                    source_adapter TEXT,
                    payload_json TEXT NOT NULL,
                    FOREIGN KEY (scan_id) REFERENCES scans(scan_id),
                    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
                );

                CREATE TABLE IF NOT EXISTS derived_data_files (
                    datafile_id TEXT PRIMARY KEY,
                    path TEXT NOT NULL,
                    checksum TEXT,
                    scan_id TEXT,
                    experiment_id TEXT,
                    format TEXT,
                    facility TEXT,
                    instrument TEXT,
                    native_format TEXT,
                    native_scan_id TEXT,
                    native_experiment_id TEXT,
                    source_adapter TEXT,
                    tool_call_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    FOREIGN KEY (scan_id) REFERENCES scans(scan_id),
                    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
                );

                CREATE TABLE IF NOT EXISTS derived_data_inputs (
                    derived_datafile_id TEXT NOT NULL,
                    input_datafile_id TEXT NOT NULL,
                    PRIMARY KEY (derived_datafile_id, input_datafile_id),
                    FOREIGN KEY (derived_datafile_id) REFERENCES derived_data_files(datafile_id)
                );

                CREATE TABLE IF NOT EXISTS paper_experiment_links (
                    paper_id TEXT NOT NULL,
                    experiment_id TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (paper_id, experiment_id),
                    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
                );

                CREATE INDEX IF NOT EXISTS idx_scans_experiment_id ON scans(experiment_id);
                CREATE INDEX IF NOT EXISTS idx_data_files_scan_id ON data_files(scan_id);
                CREATE INDEX IF NOT EXISTS idx_data_files_experiment_id ON data_files(experiment_id);
                CREATE INDEX IF NOT EXISTS idx_data_files_path ON data_files(path);
                CREATE INDEX IF NOT EXISTS idx_data_files_facility_instrument ON data_files(facility, instrument);
                CREATE INDEX IF NOT EXISTS idx_data_files_native_format ON data_files(native_format);
                CREATE INDEX IF NOT EXISTS idx_links_experiment_id ON paper_experiment_links(experiment_id);
                """
            )
            self._ensure_column(connection, "experiments", "facility", "TEXT")
            self._ensure_column(connection, "experiments", "native_experiment_id", "TEXT")
            self._ensure_column(connection, "experiments", "source_adapter", "TEXT")
            self._ensure_column(connection, "scans", "facility", "TEXT")
            self._ensure_column(connection, "scans", "native_scan_id", "TEXT")
            self._ensure_column(connection, "scans", "source_adapter", "TEXT")
            for table in ["data_files", "derived_data_files"]:
                self._ensure_column(connection, table, "facility", "TEXT")
                self._ensure_column(connection, table, "instrument", "TEXT")
                self._ensure_column(connection, table, "native_format", "TEXT")
                self._ensure_column(connection, table, "native_scan_id", "TEXT")
                self._ensure_column(connection, table, "native_experiment_id", "TEXT")
                self._ensure_column(connection, table, "source_adapter", "TEXT")
            connection.execute(
                "INSERT OR REPLACE INTO catalog_meta(key, value) VALUES (?, ?)",
                ("schema_version", str(SCHEMA_VERSION)),
            )

    def _migrate_relative_path(self, connection) -> None:
        """Add relative_path to catalogs created before portable paths."""
        columns = {row[1] for row in connection.execute("PRAGMA table_info(data_files)")}
        if "relative_path" not in columns:
            connection.execute("ALTER TABLE data_files ADD COLUMN relative_path TEXT")
        if self.archive_root:
            connection.execute(
                "INSERT OR REPLACE INTO catalog_meta (key, value) VALUES (?, ?)",
                ("archive_root", str(self.archive_root)),
            )

    def _relative_path(self, path_text: str) -> str | None:
        if not self.archive_root:
            return None
        try:
            return str(Path(path_text).relative_to(self.archive_root))
        except ValueError:
            return None

    def configured_archive_root(self) -> Path | None:
        """The root this catalog was built against, if it recorded one."""
        if self.archive_root:
            return self.archive_root
        with self.connect() as connection:
            row = connection.execute("SELECT value FROM catalog_meta WHERE key = 'archive_root'").fetchone()
        return Path(row[0]) if row and row[0] else None

    def resolve_datafile_path(self, datafile_id: str, *, archive_root: str | Path | None = None) -> Path | None:
        """Where this file lives on THIS machine.

        Relative-first: a catalog moved between machines resolves against the
        local root and only falls back to the absolute path recorded at build
        time. Returns None when neither exists, so a caller reports a missing
        file rather than treating the whole archive as empty.
        """
        root = Path(archive_root).expanduser().resolve() if archive_root else self.configured_archive_root()
        with self.connect() as connection:
            row = connection.execute(
                "SELECT path, relative_path FROM data_files WHERE datafile_id = ?", (datafile_id,)
            ).fetchone()
        if row is None:
            return None
        absolute, relative = row[0], row[1]
        if root and relative:
            candidate = Path(root) / relative
            if candidate.exists():
                return candidate
        if absolute and Path(absolute).exists():
            return Path(absolute)
        return None

    def backfill_relative_paths(self, *, archive_root: str | Path | None = None, marker: str = "bt7") -> dict[str, Any]:
        """Populate relative_path on rows cataloged before portable paths."""
        from tas_copilot_core.workflows.catalog_date_backfill import _relative_to_archive

        root = Path(archive_root).expanduser().resolve() if archive_root else self.archive_root
        updated = 0
        skipped = 0
        with self.connect() as connection:
            self._migrate_relative_path(connection)
            rows = connection.execute("SELECT datafile_id, path FROM data_files").fetchall()
            payload = []
            for datafile_id, path_text in rows:
                relative = _relative_to_archive(str(path_text), root, marker)
                if relative and relative != str(path_text):
                    payload.append((relative, datafile_id))
                    updated += 1
                else:
                    skipped += 1
            connection.executemany("UPDATE data_files SET relative_path = ? WHERE datafile_id = ?", payload)
            if root:
                connection.execute(
                    "INSERT OR REPLACE INTO catalog_meta (key, value) VALUES (?, ?)", ("archive_root", str(root))
                )
        return {"updated": updated, "skipped": skipped, "archive_root": str(root) if root else ""}

    def upsert_experiments(self, records: Iterable[Experiment]) -> None:
        with self.connect() as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO experiments
                (experiment_id, instrument, facility, native_experiment_id, source_adapter, proposal_id, sample_id, title, start_date, end_date, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        record.experiment_id,
                        record.instrument,
                        record.metadata.get("facility"),
                        record.metadata.get("native_experiment_id"),
                        record.metadata.get("source_adapter"),
                        record.proposal_id,
                        record.sample_id,
                        record.title,
                        record.start_date,
                        record.end_date,
                        record.model_dump_json(),
                    )
                    for record in records
                ],
            )

    def upsert_scans(self, records: Iterable[Scan]) -> None:
        with self.connect() as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO scans
                (scan_id, experiment_id, instrument, facility, native_scan_id, source_adapter, scan_type, physics_target, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        record.scan_id,
                        record.experiment_id,
                        record.instrument,
                        record.metadata.get("facility"),
                        record.metadata.get("native_scan_id"),
                        record.metadata.get("source_adapter"),
                        record.scan_type,
                        record.physics_target,
                        record.model_dump_json(),
                    )
                    for record in records
                ],
            )

    def upsert_datafiles(self, records: Iterable[DataFile]) -> None:
        with self.connect() as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO data_files
                (datafile_id, path, relative_path, checksum, scan_id, experiment_id, format, facility, instrument, native_format, native_scan_id, native_experiment_id, source_adapter, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        record.datafile_id,
                        record.path,
                        self._relative_path(str(record.path)),
                        record.checksum,
                        record.scan_id,
                        record.experiment_id,
                        record.format,
                        record.metadata.get("facility"),
                        record.metadata.get("instrument"),
                        record.metadata.get("native_format"),
                        record.metadata.get("native_scan_id"),
                        record.metadata.get("native_experiment_id"),
                        record.metadata.get("source_adapter"),
                        record.model_dump_json(),
                    )
                    for record in records
                ],
            )

    def upsert_derived_datafiles(self, records: Iterable[DerivedDataFile]) -> None:
        materialized = list(records)
        with self.connect() as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO derived_data_files
                (datafile_id, path, checksum, scan_id, experiment_id, format, facility, instrument, native_format, native_scan_id, native_experiment_id, source_adapter, tool_call_id, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        record.datafile_id,
                        record.path,
                        record.checksum,
                        record.scan_id,
                        record.experiment_id,
                        record.format,
                        record.metadata.get("facility"),
                        record.metadata.get("instrument"),
                        record.metadata.get("native_format"),
                        record.metadata.get("native_scan_id"),
                        record.metadata.get("native_experiment_id"),
                        record.metadata.get("source_adapter"),
                        record.tool_call_id,
                        record.model_dump_json(),
                    )
                    for record in materialized
                ],
            )
            for record in materialized:
                connection.execute(
                    "DELETE FROM derived_data_inputs WHERE derived_datafile_id = ?",
                    (record.datafile_id,),
                )
                connection.executemany(
                    """
                    INSERT OR REPLACE INTO derived_data_inputs
                    (derived_datafile_id, input_datafile_id)
                    VALUES (?, ?)
                    """,
                    [(record.datafile_id, input_id) for input_id in record.input_datafile_ids],
                )

    def upsert_paper_experiment_links(self, records: Iterable[PaperExperimentLink]) -> None:
        with self.connect() as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO paper_experiment_links
                (paper_id, experiment_id, confidence, status, payload_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        record.paper_id,
                        record.experiment_id,
                        record.confidence,
                        record.status,
                        record.model_dump_json(),
                    )
                    for record in records
                ],
            )

    def list_experiments(self) -> list[Experiment]:
        return self._list_payloads("experiments", Experiment)

    def list_scans(self) -> list[Scan]:
        return self._list_payloads("scans", Scan)

    def get_scan(self, scan_id: str) -> Scan | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM scans WHERE scan_id = ?",
                (scan_id,),
            ).fetchone()
        if row is None:
            return None
        return Scan.model_validate_json(row["payload_json"])

    def list_datafiles(self) -> list[DataFile]:
        return self._list_payloads("data_files", DataFile)

    def list_derived_datafiles(self) -> list[DerivedDataFile]:
        return self._list_payloads("derived_data_files", DerivedDataFile)

    def list_paper_experiment_links(self) -> list[PaperExperimentLink]:
        return self._list_payloads("paper_experiment_links", PaperExperimentLink)

    def mapping_candidate_experiments(self, limit: int | None = None) -> list[dict]:
        """Return experiment records shaped for paper-to-experiment mapping plugins."""

        with self.connect() as connection:
            query = """
                SELECT e.payload_json,
                       COUNT(DISTINCT s.scan_id) AS scan_count,
                       COUNT(DISTINCT d.datafile_id) AS datafile_count,
                       GROUP_CONCAT(DISTINCT s.scan_type) AS scan_types
                FROM experiments e
                LEFT JOIN scans s ON s.experiment_id = e.experiment_id
                LEFT JOIN data_files d ON d.experiment_id = e.experiment_id
                GROUP BY e.experiment_id
                ORDER BY e.experiment_id
            """
            parameters: tuple[int, ...] = ()
            if limit is not None:
                query += " LIMIT ?"
                parameters = (int(limit),)
            rows = connection.execute(query, parameters).fetchall()

        candidates = []
        for row in rows:
            experiment = Experiment.model_validate_json(row["payload_json"])
            payload = experiment.model_dump(mode="json")
            payload["id"] = experiment.experiment_id
            payload["name"] = experiment.title
            payload["metadata"] = {
                **experiment.metadata,
                "scan_count": int(row["scan_count"] or 0),
                "datafile_count": int(row["datafile_count"] or 0),
                "scan_types": [item for item in str(row["scan_types"] or "").split(",") if item],
            }
            candidates.append(payload)
        return candidates

    def datafiles_for_scan(self, scan_id: str) -> list[DataFile]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM data_files WHERE scan_id = ? ORDER BY datafile_id",
                (scan_id,),
            ).fetchall()
        return [DataFile.model_validate_json(row["payload_json"]) for row in rows]

    def lineage_for_derived_file(self, datafile_id: str) -> dict:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM derived_data_files WHERE datafile_id = ?",
                (datafile_id,),
            ).fetchone()
            inputs = connection.execute(
                """
                SELECT input_datafile_id FROM derived_data_inputs
                WHERE derived_datafile_id = ?
                ORDER BY input_datafile_id
                """,
                (datafile_id,),
            ).fetchall()
        if row is None:
            return {"datafile_id": datafile_id, "found": False, "input_datafile_ids": []}
        derived = json.loads(row["payload_json"])
        return {
            "datafile_id": datafile_id,
            "found": True,
            "tool_call_id": derived.get("tool_call_id"),
            "input_datafile_ids": [item["input_datafile_id"] for item in inputs],
            "parameters": derived.get("parameters", {}),
        }

    def summary(self) -> dict[str, int | str]:
        with self.connect() as connection:
            counts = {
                table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in [
                    "experiments",
                    "scans",
                    "data_files",
                    "derived_data_files",
                    "paper_experiment_links",
                ]
            }
        return {"path": str(self.path), **counts}

    def metrics(self) -> dict[str, int | float | str]:
        with self.connect() as connection:
            page_count = int(connection.execute("PRAGMA page_count").fetchone()[0])
            page_size = int(connection.execute("PRAGMA page_size").fetchone()[0])
            freelist_count = int(connection.execute("PRAGMA freelist_count").fetchone()[0])
            data_file_count = int(connection.execute("SELECT COUNT(*) FROM data_files").fetchone()[0])
            facility_count = int(connection.execute("SELECT COUNT(DISTINCT facility) FROM data_files WHERE facility IS NOT NULL").fetchone()[0])
            instrument_count = int(connection.execute("SELECT COUNT(DISTINCT instrument) FROM data_files WHERE instrument IS NOT NULL").fetchone()[0])
            distinct_scan_count = int(connection.execute("SELECT COUNT(DISTINCT scan_id) FROM data_files WHERE scan_id IS NOT NULL").fetchone()[0])
            distinct_experiment_count = int(connection.execute("SELECT COUNT(DISTINCT experiment_id) FROM data_files WHERE experiment_id IS NOT NULL").fetchone()[0])
        db_size_bytes = page_count * page_size
        return {
            "path": str(self.path),
            "page_count": page_count,
            "page_size": page_size,
            "freelist_count": freelist_count,
            "db_size_bytes": db_size_bytes,
            "data_file_count": data_file_count,
            "facility_count": facility_count,
            "instrument_count": instrument_count,
            "distinct_scan_count": distinct_scan_count,
            "distinct_experiment_count": distinct_experiment_count,
            "bytes_per_data_file": 0.0 if data_file_count == 0 else db_size_bytes / data_file_count,
        }

    def _list_payloads(self, table: str, model):
        with self.connect() as connection:
            rows = connection.execute(f"SELECT payload_json FROM {table}").fetchall()
        return [model.model_validate_json(row["payload_json"]) for row in rows]

    @staticmethod
    def _ensure_column(connection: sqlite3.Connection, table: str, column: str, column_type: str) -> None:
        columns = {row["name"] for row in connection.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in columns:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")
