"""Driver half of the adaptive point session.

Pairs with scripts/nice_point_server.py, which holds ONE connection open for
the whole run and executes each point as its own one-point NICE trajectory.
Keeping the connection saves the per-step process start and Ice handshake; the
trajectory is what makes the point land at the Q that was asked for, which a
bare api.move does not (see that script's docstring).

One trajectory is one data file, and an adaptive run cannot batch points into a
single trajectory because point n+1 depends on point n.  So the run produces one
file per point.  point_manifest.tsv in the spool dir records the order, and each
file prefix carries the step number and the Q and E it was taken at.

The two halves talk through a spool directory rather than in-process because
they must run under different interpreters: the policy needs numpy/sklearn and
the NICE client needs zeroc-ice, and no python on the driver host has both.
"""
from __future__ import annotations

import json
import hashlib
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

# Mirrors nice_bt7_replay: a non-local host is a real instrument and needs the
# explicit opt-in. Kept in sync deliberately -- this path must not be a way to
# reach hardware that the replay path would have refused.
_LOCALHOSTS = {"127.0.0.1", "localhost", "::1"}


class NiceSessionError(RuntimeError):
    """The session could not be started, or died while serving points."""


class NicePointSession:
    """One NICE measurement, points submitted as the policy chooses them."""

    def __init__(
        self,
        *,
        server_script: str | Path,
        spool_dir: str | Path,
        host: str,
        port: int,
        client_name: str,
        file_prefix: str,
        control_nodes: list[str],
        python_executable: str,
        title: str = "tas-copilot adaptive session",
        entry: str = "entry",
        allow_live_host: bool = False,
        timeout_seconds: float = 900.0,
        startup_timeout: float = 120.0,
        poll_seconds: float = 0.25,
        mode: str = "point",
        enabled: bool = False,
        owner_id: str | None = None,
    ) -> None:
        if host not in _LOCALHOSTS and not allow_live_host:
            raise NiceSessionError(
                f"NICE session against a non-local host {host!r} needs --allow-live-host"
            )
        self.server_script = Path(server_script)
        if not self.server_script.exists():
            raise NiceSessionError(f"point server script not found: {self.server_script}")
        if mode not in {"point", "full_scan_bridge"}:
            raise NiceSessionError("unsupported session mode")
        self.mode, self.enabled = mode, enabled
        self._started_once = False
        self.allow_live_host = allow_live_host
        self.owner_id = owner_id
        self.session_id = uuid4().hex
        self.base_spool = Path(spool_dir)
        self.spool = self.base_spool if mode == "point" else self.base_spool / f"session_{self.session_id}"
        self.host = host
        self.port = int(port)
        self.client_name = client_name
        self.file_prefix = file_prefix
        self.control_nodes = list(control_nodes)
        self.python_executable = python_executable
        self.title = title
        self.entry = entry
        self.timeout_seconds = float(timeout_seconds)
        self.startup_timeout = float(startup_timeout)
        self.poll_seconds = float(poll_seconds)
        if mode == "full_scan_bridge" and any(not math.isfinite(value) or value <= 0 for value in
                (self.timeout_seconds, self.startup_timeout, self.poll_seconds)):
            raise NiceSessionError("bridge durations must be finite and positive")
        self._proc: subprocess.Popen | None = None
        self._log = None

    # --------------------------------------------------------------- lifecycle
    @property
    def session_trace(self) -> Path:
        return self.spool / "session_trace.jsonl"

    @property
    def server_log(self) -> Path:
        return self.spool / "server.log"

    @property
    def point_manifest(self) -> Path:
        """Order, file prefix, requested vs achieved Q/E and counts, per point."""
        return self.spool / "point_manifest.tsv"

    def start(self) -> None:
        if self.mode == "full_scan_bridge" and (not self.enabled or not self.owner_id):
            raise NiceSessionError("full-scan bridge needs explicit enablement and owner_id")
        if self.mode == "full_scan_bridge":
            if self._started_once:
                raise NiceSessionError("a bridge session cannot be started twice; create a new session namespace")
            self._started_once = True
        self.spool.mkdir(parents=True, exist_ok=True)
        argv = [
            self.python_executable,
            str(self.server_script),
            "--spool", str(self.spool),
            "--host", self.host,
            "--port", str(self.port),
            "--client-name", self.client_name,
            "--file-prefix", self.file_prefix,
            "--control-nodes", ",".join(self.control_nodes),
            "--title", self.title,
            "--entry", self.entry,
            # Generous next to a single count, but finite: if the driver dies
            # the server must not sit holding a connection to the instrument.
            "--idle-timeout", str(max(600.0, self.timeout_seconds * 2)),
        ]
        if self.mode == "full_scan_bridge":
            argv += ["--mode", self.mode, "--session-id", self.session_id,
                     "--owner-id", self.owner_id, "--enable-full-scan"]
            if self.allow_live_host:
                argv.append("--allow-live-host")
        self._log = self.server_log.open("w", encoding="utf-8")
        self._proc = subprocess.Popen(
            argv, stdout=self._log, stderr=subprocess.STDOUT, text=True
        )

        ready = self.spool / "READY"
        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if ready.exists():
                if self.mode == "point":
                    return
                try:
                    marker = json.loads(ready.read_text())
                except (ValueError, OSError):
                    marker = {}
                if marker.get("session_id") == self.session_id and marker.get("owner_id") == self.owner_id:
                    return
            if self._proc.poll() is not None:
                raise NiceSessionError(
                    f"point server exited during startup (rc={self._proc.returncode}); "
                    f"see {self.server_log}:\n{self._tail_log()}"
                )
            time.sleep(self.poll_seconds)
        self.stop()
        raise NiceSessionError(
            f"point server did not become ready within {self.startup_timeout:.0f}s; "
            f"see {self.server_log}:\n{self._tail_log()}"
        )

    def execute(self, step: int, nodes: dict[str, Any], preset_time: float) -> dict[str, Any]:
        """Submit one point and block until the instrument has counted it."""
        if self.mode != "point":
            raise NiceSessionError("use command() for a full-scan bridge")
        if self._proc is None:
            raise NiceSessionError("session was never started")
        index = f"{step:03d}"
        request = {
            "step": step,
            "nodes": nodes,
            "preset_time": float(preset_time),
        }
        # Write via rename so the server cannot pick up a partial request.
        tmp = self.spool / f"request_{index}.json.tmp"
        tmp.write_text(json.dumps(request, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.spool / f"request_{index}.json")

        response = self.spool / f"response_{index}.json"
        deadline = time.time() + self.timeout_seconds + float(preset_time)
        while time.time() < deadline:
            if response.exists():
                return json.loads(response.read_text(encoding="utf-8"))
            if self._proc.poll() is not None:
                raise NiceSessionError(
                    f"point server died while counting step {step} "
                    f"(rc={self._proc.returncode}); see {self.server_log}:\n{self._tail_log()}"
                )
            time.sleep(self.poll_seconds)
        raise NiceSessionError(
            f"no response for step {step} within "
            f"{self.timeout_seconds + float(preset_time):.0f}s; see {self.server_log}"
        )

    def command(self, operation: str, payload: dict[str, Any], *, command_id: str,
                timeout_seconds: float) -> dict[str, Any]:
        """Versioned RPC to the existing NICE Task; never retry an ambiguous call.

        timeout_seconds is explicit wall time, independent of monitor preset.
        A repeated identical command waits for its original response and is not
        re-enqueued. A new session has its own namespace and nonce.
        """
        if self.mode != "full_scan_bridge" or not self.enabled or self._proc is None:
            raise NiceSessionError("full-scan bridge is not explicitly enabled and started")
        if not math.isfinite(timeout_seconds) or not timeout_seconds > 0:
            raise NiceSessionError("a positive command timeout is required")
        request = {"protocol": "tascopilot.nice-bridge.v1", "session_id": self.session_id,
                   "owner_id": self.owner_id, "command_id": command_id,
                   "operation": operation, "payload": payload}
        digest = hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        name = hashlib.sha256(command_id.encode()).hexdigest()
        path = self.spool / f"command_{name}.json"
        response = self.spool / f"command_response_{name}.json"
        request["request_hash"] = digest
        if path.exists():
            if json.loads(path.read_text()) != request:
                raise NiceSessionError("command identity reused for a different request")
        else:
            temporary = path.with_suffix(".json.tmp")
            with temporary.open("x", encoding="utf-8") as handle:
                json.dump(request, handle, sort_keys=True, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if response.exists():
                result = json.loads(response.read_text())
                if any(result.get(key) != request[key] for key in
                       ("protocol", "session_id", "owner_id", "command_id", "request_hash")):
                    raise NiceSessionError("stale or mismatched bridge response")
                if result.get("status") != "complete":
                    raise NiceSessionError(f"bridge {result.get('status')}: {result.get('error')}")
                return result["output"]
            if self._proc.poll() is not None:
                raise NiceSessionError("bridge exited; command outcome unknown and must not be replayed")
            time.sleep(self.poll_seconds)
        raise NiceSessionError("bridge command timed out; instrument outcome unknown, not cancelled")

    def stop(self, grace: float = 120.0) -> None:
        """Ask the server to close the measurement, then make sure it did."""
        if self._proc is None:
            return
        try:
            (self.spool / "STOP").write_text("", encoding="utf-8")
            deadline = time.time() + grace
            while time.time() < deadline and self._proc.poll() is None:
                time.sleep(self.poll_seconds)
            if self._proc.poll() is None:
                # measurement_end runs in the server's finally block, so give
                # SIGTERM a chance to unwind it before resorting to SIGKILL.
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait(timeout=30)
        finally:
            if self._log is not None:
                self._log.close()
                self._log = None

    # ------------------------------------------------------------------ detail
    def _tail_log(self, lines: int = 20) -> str:
        try:
            return "\n".join(self.server_log.read_text(encoding="utf-8").splitlines()[-lines:])
        except OSError:
            return "(no log)"
