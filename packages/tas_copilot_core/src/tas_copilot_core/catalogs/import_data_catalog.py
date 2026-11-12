from __future__ import annotations

from pathlib import Path

from tas_copilot_contracts import DataFile, DerivedDataFile, Experiment, PaperExperimentLink, Scan

from .jsonl_catalog import JsonlCatalog
from .sqlite_data_catalog import SQLiteDataCatalog


def import_jsonl_catalog_dir(catalog_dir: str | Path, db_path: str | Path) -> dict:
    """Import data-side JSONL fixtures into SQLite.

    This imports experiment/scan/datafile metadata and paper-to-experiment link
    references. It does not import the paper corpus itself.
    """

    catalog_path = Path(catalog_dir)
    db = SQLiteDataCatalog(db_path)
    experiments = JsonlCatalog(catalog_path / "experiments.jsonl", Experiment).load()
    scans = JsonlCatalog(catalog_path / "scans.jsonl", Scan).load()
    datafiles = JsonlCatalog(catalog_path / "datafiles.jsonl", DataFile).load()
    mappings = JsonlCatalog(catalog_path / "mappings.jsonl", PaperExperimentLink).load()
    derived_path = catalog_path / "derived_datafiles.jsonl"
    derived = JsonlCatalog(derived_path, DerivedDataFile).load() if derived_path.exists() else []

    db.upsert_experiments(experiments)
    db.upsert_scans(scans)
    db.upsert_datafiles(datafiles)
    db.upsert_derived_datafiles(derived)
    db.upsert_paper_experiment_links(mappings)
    return db.summary()

