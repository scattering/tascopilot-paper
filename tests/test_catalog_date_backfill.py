from __future__ import annotations

import sqlite3
from pathlib import Path

from tas_copilot_core.workflows.catalog_date_backfill import _parse_header_date, backfill_catalog_dates


def _catalog_with_files(tmp_path: Path) -> tuple[Path, Path]:
    data = tmp_path / "bt7" / "201804" / "24858" / "data"
    data.mkdir(parents=True)
    # Spacing varies between files; a naive single-space grep for "#Date " missed
    # real dates in the archive, so the parser must tolerate it.
    (data / "a.bt7").write_text("#ICE 1.0\n#Date    2018-05-25T03:14:00\n#Columns QX QY\n1 2\n", encoding="utf-8")
    (data / "b.bt7").write_text("#ICE 1.0\n#Date 2018-05-27T11:00:00\n#Columns QX QY\n1 2\n", encoding="utf-8")
    (data / "c.bt7").write_text("#ICE 1.0\n#Columns QX QY\n1 2\n", encoding="utf-8")  # no date at all

    db = tmp_path / "catalog.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE data_files (datafile_id TEXT, path TEXT, experiment_id TEXT)")
    conn.execute("CREATE TABLE experiments (experiment_id TEXT, start_date TEXT, end_date TEXT)")
    conn.executemany(
        "INSERT INTO data_files VALUES (?, ?, ?)",
        [
            ("df-a", str(data / "a.bt7"), "exp-1"),
            ("df-b", str(data / "b.bt7"), "exp-1"),
            ("df-c", str(data / "c.bt7"), "exp-1"),
            ("df-gone", str(data / "missing.bt7"), "exp-1"),
        ],
    )
    conn.execute("INSERT INTO experiments VALUES ('exp-1', NULL, NULL)")
    conn.commit()
    conn.close()
    return db, data


def test_header_date_parsing_tolerates_spacing_and_missing_dates(tmp_path: Path) -> None:
    f = tmp_path / "x.bt7"
    f.write_text("#Date    2018-05-25T03:14:00\n", encoding="utf-8")
    assert _parse_header_date(f) == "2018-05-25"
    f.write_text("#Date 2018-05-25\n", encoding="utf-8")
    assert _parse_header_date(f) == "2018-05-25"
    f.write_text("#Columns QX QY\n1 2\n", encoding="utf-8")
    assert _parse_header_date(f) is None


def test_legacy_pre_ice_files_still_yield_dates(tmp_path: Path) -> None:
    """Pre-ICE BT-7 files have no #Date line; the date is on line 1.

    Parsing only the ICE header left every file before ~2005 undated (25,013 of
    171,935), which looked like an era that simply never recorded dates. It did
    - in a different layout, with two different formats across the 1990s and
    2000s.
    """
    f = tmp_path / "legacy.bt7"
    f.write_text(
        "'DPPCM008.BT7' '19-APR-1994 18:45' 'I'        170.    6  'TIME'   34  'RAW'\n"
        "  Filename         Date            Scan       Mon    Prf  Base   #pts  Type\n",
        encoding="utf-8",
    )
    assert _parse_header_date(f) == "1994-04-19"

    # 2000s variant, with the double space before a single-digit day.
    f.write_text(
        "'CCTO4307.bt7' 'Dec  4 2002 03:12' 'Q'         32.    1  'TIME'   31  'RAW'\n",
        encoding="utf-8",
    )
    assert _parse_header_date(f) == "2002-12-04"

    # A quoted first field that is not a date must not be mistaken for one.
    f.write_text("'somefile.bt7' 'I' 'RAW'\n", encoding="utf-8")
    assert _parse_header_date(f) is None


def test_backfill_makes_dates_queryable_and_sets_experiment_span(tmp_path: Path) -> None:
    """The catalog was built path-only, so no file was ever opened and no date
    was captured. Dates live inside the files; this makes them queryable."""
    db, _ = _catalog_with_files(tmp_path)
    report = backfill_catalog_dates({"catalog_sqlite": str(db)})

    assert report["status"] == "ok"
    assert report["files_examined"] == 4
    assert report["dated"] == 2
    assert report["undated"] == 1  # header present, no date line
    assert report["missing_file"] == 1  # catalog row whose file is gone
    assert report["experiments_updated"] == 1

    conn = sqlite3.connect(db)
    # The question that could not be answered before: what ran on a given day.
    hits = [row[0] for row in conn.execute("SELECT path FROM data_file_dates WHERE measured_date = '2018-05-25'")]
    assert len(hits) == 1 and hits[0].endswith("a.bt7")
    # Experiment span comes from the files, not the directory name.
    assert conn.execute("SELECT start_date, end_date FROM experiments WHERE experiment_id='exp-1'").fetchone() == (
        "2018-05-25",
        "2018-05-27",
    )
    # Failures are recorded rather than silently dropped.
    statuses = dict(conn.execute("SELECT datafile_id, status FROM data_file_dates"))
    assert statuses["df-c"] == "no_date_in_header"
    assert statuses["df-gone"] == "file_not_found"
    conn.close()


def test_backfill_is_idempotent_and_scopable(tmp_path: Path) -> None:
    db, data = _catalog_with_files(tmp_path)
    backfill_catalog_dates({"catalog_sqlite": str(db)})
    second = backfill_catalog_dates({"catalog_sqlite": str(db)})
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*) FROM data_file_dates").fetchone()[0] == 4, "re-running must not duplicate rows"
    conn.close()
    assert second["dated"] == 2

    scoped = backfill_catalog_dates({"catalog_sqlite": str(db), "path_prefix": str(data / "a")})
    assert scoped["files_examined"] == 1


def test_backfill_reports_missing_catalog(tmp_path: Path) -> None:
    report = backfill_catalog_dates({"catalog_sqlite": str(tmp_path / "nope.sqlite")})
    assert report["status"] == "missing_input"
    assert "catalog_sqlite" in report["missing_inputs"]


def test_relative_paths_make_the_catalog_portable(tmp_path: Path) -> None:
    """A catalog built on one machine must be usable on another.

    Stored paths are absolute, so the same catalog resolves nothing on Vista or
    at NIST - every row would report file_not_found. Recording a root-relative
    path and accepting a local --archive-root fixes that.
    """
    from tas_copilot_core.workflows.catalog_date_backfill import _relative_to_archive, _resolve_local

    built_on = "/Users/someone/bt7_from_stampede/bt7/201804/24858/data/a.bt7"
    # Without a root, the marker segment still recovers the portable portion.
    assert _relative_to_archive(built_on, None, "bt7") == "bt7/201804/24858/data/a.bt7"
    # With an explicit root, relativity is exact.
    assert _relative_to_archive(built_on, Path("/Users/someone/bt7_from_stampede"), "bt7") == "bt7/201804/24858/data/a.bt7"

    # Now simulate "same catalog, different machine": the absolute path does not
    # exist here, but the file is present under a local root.
    local_root = tmp_path / "elsewhere"
    (local_root / "bt7" / "201804" / "24858" / "data").mkdir(parents=True)
    local_file = local_root / "bt7" / "201804" / "24858" / "data" / "a.bt7"
    local_file.write_text("#Date 2018-05-25T01:00:00\n", encoding="utf-8")

    assert _resolve_local(built_on, "bt7/201804/24858/data/a.bt7", None) is None
    assert _resolve_local(built_on, "bt7/201804/24858/data/a.bt7", local_root) == local_file

    # End to end: a catalog holding foreign absolute paths still backfills.
    db = tmp_path / "foreign.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE data_files (datafile_id TEXT, path TEXT, experiment_id TEXT)")
    conn.execute("CREATE TABLE experiments (experiment_id TEXT, start_date TEXT, end_date TEXT)")
    conn.execute("INSERT INTO data_files VALUES ('df-1', ?, 'exp-1')", (built_on,))
    conn.execute("INSERT INTO experiments VALUES ('exp-1', NULL, NULL)")
    conn.commit()
    conn.close()

    without_root = backfill_catalog_dates({"catalog_sqlite": str(db)})
    assert without_root["missing_file"] == 1, "foreign absolute path cannot resolve without a root"

    with_root = backfill_catalog_dates({"catalog_sqlite": str(db), "archive_root": str(local_root)})
    assert with_root["dated"] == 1
    conn = sqlite3.connect(db)
    row = conn.execute("SELECT relative_path, measured_date FROM data_file_dates").fetchone()
    conn.close()
    assert row == ("bt7/201804/24858/data/a.bt7", "2018-05-25")


def test_catalog_store_paths_are_portable(tmp_path: Path) -> None:
    """A catalog must resolve on a machine other than the one that built it.

    data_files.path is absolute, so the full BT-7 catalog's 171,935 rows resolve
    to nothing on Vista or at NIST - which reads exactly like an empty archive.
    Storing relative_path and resolving relative-first fixes that.
    """
    from tas_copilot_contracts import DataFile
    from tas_copilot_core.catalogs.sqlite_data_catalog import SQLiteDataCatalog

    built_root = tmp_path / "machine_a"
    data_dir = built_root / "bt7" / "201804" / "24858" / "data"
    data_dir.mkdir(parents=True)
    scan_file = data_dir / "a.bt7"
    scan_file.write_text("#Date 2018-05-25T01:00:00\n", encoding="utf-8")

    db = tmp_path / "portable.sqlite"
    store = SQLiteDataCatalog(db, archive_root=built_root)
    store.upsert_datafiles([DataFile(datafile_id="df-1", path=str(scan_file), format="ncnr_bt7_ice_nice")])

    import sqlite3

    conn = sqlite3.connect(db)
    stored = conn.execute("SELECT path, relative_path FROM data_files WHERE datafile_id='df-1'").fetchone()
    conn.close()
    assert stored[1] == "bt7/201804/24858/data/a.bt7", "relative path recorded alongside the absolute one"
    assert store.resolve_datafile_path("df-1") == scan_file

    # Same catalog, second machine: absolute path is meaningless, local copy works.
    moved_root = tmp_path / "machine_b"
    moved_dir = moved_root / "bt7" / "201804" / "24858" / "data"
    moved_dir.mkdir(parents=True)
    (moved_dir / "a.bt7").write_text("#Date 2018-05-25T01:00:00\n", encoding="utf-8")
    scan_file.unlink()  # the build machine's copy is gone

    assert store.resolve_datafile_path("df-1") is None, "no local root, no absolute file -> honestly unresolvable"
    assert store.resolve_datafile_path("df-1", archive_root=moved_root) == moved_dir / "a.bt7"


def test_existing_catalog_migrates_and_backfills_relative_paths(tmp_path: Path) -> None:
    """Catalogs built before portable paths must upgrade in place, not be rebuilt."""
    import sqlite3

    from tas_copilot_core.catalogs.sqlite_data_catalog import SQLiteDataCatalog

    db = tmp_path / "legacy.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE catalog_meta (key TEXT PRIMARY KEY, value TEXT)")
    # The real pre-migration schema, minus relative_path - that is exactly what
    # the 171,935-row BT-7 catalog looks like today.
    conn.execute(
        """
        CREATE TABLE data_files (
            datafile_id TEXT PRIMARY KEY, path TEXT NOT NULL, checksum TEXT, scan_id TEXT,
            experiment_id TEXT, format TEXT, facility TEXT, instrument TEXT, native_format TEXT,
            native_scan_id TEXT, native_experiment_id TEXT, source_adapter TEXT, payload_json TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "INSERT INTO data_files (datafile_id, path, payload_json) VALUES "
        "('df-old', '/Users/someone/bt7_from_stampede/bt7/201804/24858/data/a.bt7', '{}')"
    )
    conn.commit()
    conn.close()

    store = SQLiteDataCatalog(db)
    result = store.backfill_relative_paths(marker="bt7")
    assert result["updated"] == 1

    conn = sqlite3.connect(db)
    assert conn.execute("SELECT relative_path FROM data_files").fetchone()[0] == "bt7/201804/24858/data/a.bt7"
    conn.close()
