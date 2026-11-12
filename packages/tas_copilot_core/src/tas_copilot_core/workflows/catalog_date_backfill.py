"""Backfill measurement dates into a path-only data catalog.

The full BT-7 catalog was built in ``path_only`` mode: 171,935 data files
cataloged without opening any of them ("cataloged by explicit path only; file
content, stat, checksum, and hydration were not attempted"). That is fast, but
the measurement date lives INSIDE the file, so the catalog cannot answer
"what ran on 2018-05-25" at all.

The gap is not academic. BT-7 archive directories are named by beamtime cycle,
not month - ``201804`` holds 2018-04-26 through 2018-06-14 - so a date question
degrades into guessing at a path whose name actively misleads. A user looking
for May 2018 data reasonably tries ``201805``, which does not exist, and the
overnight-summary tool then reports nothing found.

This workflow reads only the header of each file (a bounded prefix, not the
whole scan) and records the date in a ``data_file_dates`` table alongside the
existing rows, plus start/end dates on ``experiments``. A separate table keeps
the DataFile contract unchanged while making dates queryable:

    SELECT path FROM data_file_dates WHERE measured_date = '2018-05-25';
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from tas_copilot_contracts import ProvenanceRecord

# Only the header is read. BT-7 files put #Date within the first block; scan
# tables can be megabytes and are irrelevant here.
_HEADER_BYTES = 8192
# Tolerates the spacing variation that made a naive grep miss real dates.
_DATE_LINE = re.compile(r"^#\s*Date\s+(\S+)", re.IGNORECASE | re.MULTILINE)


def _parse_header_date(path: Path, *, header_bytes: int = _HEADER_BYTES) -> str | None:
    """ISO date (YYYY-MM-DD) from a scan file header, or None."""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            head = handle.read(header_bytes)
    except OSError:
        return None
    match = _DATE_LINE.search(head)
    if not match:
        return _parse_legacy_header_date(head)
    raw = match.group(1).strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%m/%d/%Y", "%d-%b-%Y"):
        try:
            return datetime.strptime(raw.split("T")[0] if fmt == "%Y-%m-%d" else raw, fmt).date().isoformat()
        except ValueError:
            continue
    # Last resort: a leading ISO-looking date.
    iso = re.match(r"(\d{4})-(\d{2})-(\d{2})", raw)
    return f"{iso.group(1)}-{iso.group(2)}-{iso.group(3)}" if iso else None


# Pre-ICE BT-7 files carry no "#Date" line; the date is the second quoted field
# of line 1, e.g.
#   'DPPCM008.BT7' '19-APR-1994 18:45' 'I' ...
#   'CCTO4307.bt7' 'Dec  4 2002 03:12' 'Q' ...
# Every file before ~2005 is this format, which is why a header-only ICE parser
# left 25,013 files undated and made the gap look like an era with no dates.
_LEGACY_QUOTED = re.compile(r"'([^']*)'")
_LEGACY_DATE_FORMATS = ("%d-%b-%Y %H:%M", "%b %d %Y %H:%M", "%d-%b-%Y", "%b %d %Y")


def _parse_legacy_header_date(head: str) -> str | None:
    """Date from the legacy (pre-ICE) BT-7 first line, or None."""
    first_line = head.split("\n", 1)[0]
    fields = _LEGACY_QUOTED.findall(first_line)
    for field in fields[1:3]:
        candidate = " ".join(field.split())
        for fmt in _LEGACY_DATE_FORMATS:
            try:
                return datetime.strptime(candidate, fmt).date().isoformat()
            except ValueError:
                continue
    return None


def _relative_to_archive(path_text: str, archive_root: Path | None, marker: str) -> str:
    """Path relative to the archive root, so the catalog is machine-portable.

    Catalogs currently store absolute paths, which pins them to the machine that
    built them: the same catalog on Vista or at NIST resolves nothing. A
    root-relative path plus a locally supplied root works anywhere. When no root
    is given, fall back to the first occurrence of a well-known marker segment
    (e.g. "bt7/"), which recovers portability for archives already cataloged.
    """
    text = str(path_text)
    if archive_root:
        try:
            return str(Path(text).relative_to(archive_root))
        except ValueError:
            pass
    marker_token = f"/{marker.strip('/')}/"
    index = text.find(marker_token)
    if index >= 0:
        return text[index + 1 :]
    return text


def _resolve_local(path_text: str, relative_path: str, archive_root: Path | None) -> Path | None:
    """Where this file actually is on THIS machine, or None."""
    absolute = Path(path_text)
    if absolute.exists():
        return absolute
    if archive_root:
        candidate = archive_root / relative_path
        if candidate.exists():
            return candidate
    return None


def backfill_catalog_dates(arguments: dict[str, Any]) -> dict[str, Any]:
    """Read header dates for cataloged files and make them queryable.

    Arguments: ``catalog_sqlite`` (required), optional ``limit``,
    ``path_prefix`` to scope a run, and ``header_bytes``.
    """
    catalog_path = Path(str(arguments.get("catalog_sqlite") or "")).expanduser()
    if not catalog_path.exists():
        return {
            "workflow_name": "catalog-date-backfill",
            "status": "missing_input",
            "missing_inputs": ["catalog_sqlite"],
            "errors": [{"error": f"catalog sqlite not found: {catalog_path}"}],
        }
    limit = int(arguments.get("limit") or 0)
    path_prefix = str(arguments.get("path_prefix") or "")
    archive_root_arg = str(arguments.get("archive_root") or "")
    archive_root = Path(archive_root_arg).expanduser().resolve() if archive_root_arg else None
    marker = str(arguments.get("path_marker") or "bt7")
    header_bytes = int(arguments.get("header_bytes") or _HEADER_BYTES)

    connection = sqlite3.connect(catalog_path)
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS data_file_dates (
                datafile_id TEXT PRIMARY KEY,
                path TEXT,
                relative_path TEXT,
                experiment_id TEXT,
                measured_date TEXT,
                source TEXT,
                status TEXT
            )
            """
        )
        connection.execute("CREATE INDEX IF NOT EXISTS idx_data_file_dates_date ON data_file_dates (measured_date)")
        connection.commit()

        query = "SELECT datafile_id, path, experiment_id FROM data_files"
        params: list[Any] = []
        if path_prefix:
            query += " WHERE path LIKE ?"
            params.append(f"{path_prefix}%")
        query += " ORDER BY path"
        if limit > 0:
            query += f" LIMIT {limit}"

        rows = connection.execute(query, params).fetchall()
        dated = 0
        undated = 0
        missing_file = 0
        payload: list[tuple[str, str, str, str | None, str, str]] = []
        for datafile_id, path_text, experiment_id in rows:
            relative_path = _relative_to_archive(str(path_text), archive_root, marker)
            path = _resolve_local(str(path_text), relative_path, archive_root)
            if path is None:
                missing_file += 1
                payload.append((datafile_id, str(path_text), relative_path, experiment_id or "", None, "file_header", "file_not_found"))
                continue
            measured = _parse_header_date(path, header_bytes=header_bytes)
            if measured:
                dated += 1
                payload.append((datafile_id, str(path_text), relative_path, experiment_id or "", measured, "file_header", "ok"))
            else:
                undated += 1
                payload.append((datafile_id, str(path_text), relative_path, experiment_id or "", None, "file_header", "no_date_in_header"))

        connection.executemany(
            """
            INSERT INTO data_file_dates (datafile_id, path, relative_path, experiment_id, measured_date, source, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(datafile_id) DO UPDATE SET
                path=excluded.path, relative_path=excluded.relative_path, experiment_id=excluded.experiment_id,
                measured_date=excluded.measured_date, source=excluded.source, status=excluded.status
            """,
            payload,
        )

        # Experiment start/end follow from the files, so a date question can be
        # answered per experiment without rescanning.
        experiments_updated = 0
        for experiment_id, first, last in connection.execute(
            """
            SELECT experiment_id, MIN(measured_date), MAX(measured_date)
            FROM data_file_dates
            WHERE measured_date IS NOT NULL AND experiment_id != ''
            GROUP BY experiment_id
            """
        ).fetchall():
            connection.execute(
                "UPDATE experiments SET start_date = ?, end_date = ? WHERE experiment_id = ?",
                (first, last, experiment_id),
            )
            experiments_updated += 1
        connection.commit()

        coverage = connection.execute(
            "SELECT COUNT(*), SUM(CASE WHEN measured_date IS NOT NULL THEN 1 ELSE 0 END) FROM data_file_dates"
        ).fetchone()
    finally:
        connection.close()

    return {
        "workflow_name": "catalog-date-backfill",
        "status": "ok",
        "catalog_sqlite": str(catalog_path),
        "files_examined": len(rows),
        "archive_root": str(archive_root) if archive_root else None,
        "portability": (
            "data_file_dates stores relative_path; pass --archive-root on another machine to resolve the same "
            "catalog there. The data_files.path column itself is still absolute, which pins the base catalog to "
            "the machine that built it."
        ),
        "dated": dated,
        "undated": undated,
        "missing_file": missing_file,
        "experiments_updated": experiments_updated,
        "catalog_dated_total": coverage[1] or 0,
        "catalog_date_rows": coverage[0] or 0,
        "note": (
            "Dates come from each file's own header, not the directory name: BT-7 archive "
            "directories are beamtime cycles (201804 spans 2018-04-26..2018-06-14), so path "
            "components must never be read as year-month."
        ),
        "query_example": "SELECT path FROM data_file_dates WHERE measured_date = '2018-05-25'",
        "provenance": [
            ProvenanceRecord(
                source_type="data_catalog",
                source_id=str(catalog_path),
                method="catalog_date_backfill.file_header",
            ).model_dump(mode="json")
        ],
    }
