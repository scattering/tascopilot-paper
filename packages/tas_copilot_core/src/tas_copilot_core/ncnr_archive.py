"""Read-only client for the public NCNR data archive.

The archive (``https://ncnr.nist.gov/pub/ncnrdata/<instrument>/<cycle>/<id>/``)
publishes each running experiment as plain files behind an HTML listing.
This module parses those listings, discovers the experiment currently being
written, and mirrors only what changed into a local cache. It never uses the
``zipdownload.py`` CGI (the zip is regenerated per call and is hundreds of MB
for a long experiment).

Listing rows look like::

    <tr><td>NAME.dat.bt7</td><td class="links"><a href="URL" download>download</a></td>
        <td>2026-09-04 08:00:37-04:00</td><td class="filesize">3M</td></tr>
    <tr><td><a href="URL/">16</a></td><td>2026-08-31 14:10:03</td></tr>   (directory)

Sizes are humanised ("3M"), so change detection uses the modification time.
"""
from __future__ import annotations

import html
import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import requests

DEFAULT_BASE = "https://ncnr.nist.gov/pub/ncnrdata"
DEFAULT_METADATA_API = "https://ncnr.nist.gov/ncnrdata/metadata/api/v1/experiments.php"
USER_AGENT = "tascopilot-live-summary/0.1 (+https://tripleaxis.org/summary/)"

_ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.I)
_CELL_RE = re.compile(r"<td[^>]*>(.*?)</td>", re.S | re.I)
_HREF_RE = re.compile(r'href="([^"]+)"', re.I)
_TAG_RE = re.compile(r"<[^>]+>")
_MTIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")


@dataclass(frozen=True)
class ArchiveEntry:
    name: str
    href: str
    mtime: str
    size: str
    is_dir: bool


def parse_listing(text: str, base_url: str) -> list[ArchiveEntry]:
    """Parse one archive listing page into entries. Header rows (``<th>``) and
    the navigation breadcrumbs are skipped because they carry no ``<td>`` cells."""
    entries: list[ArchiveEntry] = []
    base = base_url.rstrip("/") + "/"
    for row in _ROW_RE.findall(text):
        cells = _CELL_RE.findall(row)
        if not cells:
            continue
        first = cells[0]
        link = _HREF_RE.search(first)
        name = html.unescape(_TAG_RE.sub("", first)).strip()
        if not name or name in {"Name", "Parent Directory", ".."}:
            continue
        mtime = ""
        size = ""
        for cell in cells[1:]:
            plain = html.unescape(_TAG_RE.sub("", cell)).strip()
            if _MTIME_RE.match(plain):
                mtime = plain
            elif 'class="filesize"' in cell or plain and re.fullmatch(r"[\d.]+[KMGT]?B?", plain):
                size = plain
        if link:
            href = html.unescape(link.group(1))
            entries.append(ArchiveEntry(name=name, href=href, mtime=mtime, size=size, is_dir=href.endswith("/")))
            continue
        download = _HREF_RE.search(cells[1]) if len(cells) > 1 else None
        href = html.unescape(download.group(1)) if download else base + name
        entries.append(ArchiveEntry(name=name, href=href, mtime=mtime, size=size, is_dir=False))
    return entries


def strip_console_html(raw: str) -> str:
    """NICE's console_history.html is one big <pre>-like document with heavy
    CSS; keep the non-empty text lines."""
    text = html.unescape(_TAG_RE.sub("", raw))
    return "\n".join(line for line in text.splitlines() if line.strip())


class NCNRArchive:
    def __init__(self, base_url: str = DEFAULT_BASE, *, metadata_api: str = DEFAULT_METADATA_API, timeout: float = 60.0, session: requests.Session | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.metadata_api = metadata_api
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", USER_AGENT)

    # -- listings -----------------------------------------------------------
    def url(self, *parts: str) -> str:
        return "/".join([self.base_url, *[p.strip("/") for p in parts]]) + "/"

    def listing(self, *parts: str, sort_newest_first: bool = False) -> list[ArchiveEntry]:
        url = self.url(*parts)
        if sort_newest_first:
            url += "?C=M&O=D"
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return parse_listing(response.text, self.url(*parts))

    def cycles(self, instrument: str) -> list[str]:
        names = [e.name.strip("/") for e in self.listing(instrument) if e.is_dir and re.fullmatch(r"\d{6}", e.name.strip("/"))]
        return sorted(names, reverse=True)

    def experiments(self, instrument: str, cycle: str) -> list[ArchiveEntry]:
        return [e for e in self.listing(instrument, cycle) if e.is_dir and re.fullmatch(r"\d+", e.name.strip("/"))]

    def newest_data_mtime(self, instrument: str, cycle: str, experiment_id: str) -> str:
        entries = [e for e in self.listing(instrument, cycle, experiment_id, "data", sort_newest_first=True) if not e.is_dir and e.mtime]
        return max((e.mtime for e in entries), default="")

    def discover_current_experiment(self, instrument: str, *, max_candidates: int = 6) -> dict[str, Any]:
        """The experiment whose data directory received the most recent file,
        looked for in the newest cycle first (falls back to the previous cycle
        when the newest one has no data yet)."""
        for cycle in self.cycles(instrument)[:2]:
            candidates = sorted(self.experiments(instrument, cycle), key=lambda e: e.mtime, reverse=True)[:max_candidates]
            best: tuple[str, str] | None = None
            for entry in candidates:
                exp_id = entry.name.strip("/")
                try:
                    latest = self.newest_data_mtime(instrument, cycle, exp_id)
                except requests.RequestException:
                    continue
                if latest and (best is None or latest > best[1]):
                    best = (exp_id, latest)
            if best:
                return {"instrument": instrument, "cycle": cycle, "experiment_id": best[0], "latest_data_mtime": best[1]}
        raise RuntimeError(f"no experiment with data found for {instrument}")

    def experiment_metadata(self, instrument: str, experiment_id: str) -> dict[str, Any]:
        try:
            response = self.session.get(self.metadata_api, params={"instrument": instrument, "id": experiment_id}, timeout=self.timeout)
            response.raise_for_status()
            rows = response.json()
        except (requests.RequestException, ValueError):
            return {}
        if not rows:
            return {}
        row = dict(rows[0])
        for key in ("participant_names", "participant_orcids"):
            if isinstance(row.get(key), str):
                try:
                    row[key] = json.loads(row[key])
                except ValueError:
                    pass
        return row

    # -- mirroring ----------------------------------------------------------
    def sync_files(self, instrument: str, cycle: str, experiment_id: str, cache_dir: Path, *, include: Callable[[str], bool], subdir: str = "data") -> dict[str, Any]:
        """Download files under ``<exp>/<subdir>/`` accepted by ``include`` whose
        listing mtime differs from the cached manifest. Returns what changed."""
        cache_dir = Path(cache_dir)
        target = cache_dir / subdir
        target.mkdir(parents=True, exist_ok=True)
        manifest_path = cache_dir / f"{subdir}_manifest.json"
        manifest: dict[str, dict[str, str]] = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        entries = [e for e in self.listing(instrument, cycle, experiment_id, subdir) if not e.is_dir]
        downloaded: list[str] = []
        failed: list[str] = []
        for entry in entries:
            if not include(entry.name):
                continue
            known = manifest.get(entry.name)
            if known and known.get("mtime") == entry.mtime and (target / entry.name).exists():
                continue
            try:
                self._download(entry.href, target / entry.name)
            except requests.RequestException as exc:
                failed.append(f"{entry.name}: {exc}")
                continue
            manifest[entry.name] = {"mtime": entry.mtime, "size": entry.size}
            downloaded.append(entry.name)
        manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
        return {"listed": len(entries), "downloaded": downloaded, "failed": failed, "manifest": manifest_path}

    def fetch_named(self, instrument: str, cycle: str, experiment_id: str, cache_dir: Path, names: Iterable[str], *, subdir: str = "data") -> list[str]:
        """Fetch specific files (e.g. the NeXus twins of mesh rows) if absent."""
        target = Path(cache_dir) / subdir
        target.mkdir(parents=True, exist_ok=True)
        fetched: list[str] = []
        for name in names:
            path = target / name
            if path.exists():
                continue
            try:
                self._download(self.url(instrument, cycle, experiment_id, subdir).rstrip("/") + "/" + name, path)
                fetched.append(name)
            except requests.RequestException:
                continue
        return fetched

    def sync_console(self, instrument: str, cycle: str, experiment_id: str, cache_dir: Path) -> dict[str, Any]:
        """Append-only fetch of console_history.html with an HTTP Range request;
        falls back to a full download when the server does not honour ranges.
        The stripped text is rewritten each time from the raw cache."""
        cache_dir = Path(cache_dir)
        raw_path = cache_dir / "console_history.html"
        txt_path = cache_dir / "console_history.txt"
        url = self.url(instrument, cycle, experiment_id, "console").rstrip("/") + "/console_history.html"
        offset = raw_path.stat().st_size if raw_path.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        response = self.session.get(url, headers=headers, timeout=self.timeout)
        status = response.status_code
        if status == 416:
            appended = 0
        elif status == 206:
            with raw_path.open("ab") as handle:
                handle.write(response.content)
            appended = len(response.content)
        elif status == 200:
            raw_path.write_bytes(response.content)
            appended = len(response.content)
        else:
            response.raise_for_status()
            appended = 0
        if appended or not txt_path.exists():
            txt_path.write_text(strip_console_html(raw_path.read_text(errors="replace")))
        return {"status": status, "appended_bytes": appended, "text_path": txt_path}

    def fetch_queue(self, instrument: str, cycle: str, experiment_id: str, cache_dir: Path) -> Path | None:
        url = self.url(instrument, cycle, experiment_id, "queue").rstrip("/") + "/queue"
        path = Path(cache_dir) / "queue.json"
        try:
            self._download(url, path)
        except requests.RequestException:
            return path if path.exists() else None
        return path

    def _download(self, url: str, path: Path) -> None:
        tmp = path.with_suffix(path.suffix + ".part")
        with self.session.get(url, stream=True, timeout=self.timeout) as response:
            response.raise_for_status()
            with tmp.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 16):
                    handle.write(chunk)
        tmp.replace(path)


def reset_cache_if_experiment_changed(cache_dir: Path, discovered: dict[str, Any]) -> bool:
    """The page follows one experiment; when the archive's current experiment
    changes, the old cache is removed rather than archived."""
    cache_dir = Path(cache_dir)
    marker = cache_dir / "experiment.json"
    previous = json.loads(marker.read_text()) if marker.exists() else {}
    changed = bool(previous) and (previous.get("cycle"), previous.get("experiment_id")) != (discovered.get("cycle"), discovered.get("experiment_id"))
    if changed:
        shutil.rmtree(cache_dir, ignore_errors=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(discovered, indent=1))
    return changed


__all__ = ["ArchiveEntry", "NCNRArchive", "parse_listing", "strip_console_html", "reset_cache_if_experiment_changed", "asdict"]
