"""Inference transport tests: every remote command is intercepted by a fake."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess

import pytest

from tas_copilot_core.workflows.andie_inference_jobs import (
    ArtifactRef, CheckpointRef, HPCProfile, IdentityMismatch, InferenceJobBackend,
    InferenceJobError, InferenceJobRequest, InferenceJobResult,
    RemoteExecutionDisabled, TransportError, UnknownSubmission,
)


def response(stdout="", returncode=0, stderr=""):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


class FakeRunner:
    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    def __call__(self, argv):
        self.calls.append(argv)
        assert self.answers, f"unexpected transport: {argv}"
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer(argv) if callable(answer) else answer


@pytest.fixture
def profile():
    return HPCProfile("test", "compute.example", "/remote/inference", "debug",
                      python_executable="/venv/bin/python", runtime_version="test-1")


@pytest.fixture
def job_request():
    return InferenceJobRequest({"model": {"id": "external"}, "data": [1, 2]}, ["Tc", "amplitude"],
                               provenance={"source_sha256": "1" * 64})


COMMAND = ["{python}", "{input}/worker.py", "--request", "{request}", "--result", "{output}/result.json", "--cpus", "{cpus}"]


def prepared(tmp_path, profile, job_request, runner, **kwargs):
    backend = InferenceJobBackend(profile, tmp_path, runner=runner, allow_remote=True)
    backend.prepare(job_request, COMMAND, **kwargs)
    return backend


def submitted(tmp_path, profile, job_request):
    runner = FakeRunner(response(), response(), response("12345;cluster\n"))
    backend = prepared(tmp_path, profile, job_request, runner)
    assert backend.submit(job_request.request_id)["job_id"] == "12345"
    return backend, runner


def completed(tmp_path, profile, job_request):
    backend, runner = submitted(tmp_path, profile, job_request)
    runner.answers += [response(), response("12345|COMPLETED|0:0\n12345.batch|COMPLETED|0:0\n")]
    assert backend.status(job_request.request_id)["state"] == "scheduler_completed"
    return backend, runner


def download(job_request, change=None):
    def copy(argv):
        assert argv[0] == "scp"
        target = Path(argv[-1])
        target.mkdir(exist_ok=True)
        (target / "posterior.npz").write_bytes(b"opaque external posterior bytes")
        result = InferenceJobResult(job_request.request_id, job_request.input_hash, "12345", job_request.parameter_names,
                                    "completed", [ArtifactRef.from_file("posterior.npz", target / "posterior.npz")])
        value = result.to_dict()
        if change:
            change(value, target)
        (target / "result.json").write_text(json.dumps(value))
        return response()
    return copy


def test_canonical_identity_and_serialization(job_request):
    equivalent = InferenceJobRequest({"data": [1, 2], "model": {"id": "external"}}, ["Tc", "amplitude"],
                                    provenance={"source_sha256": "1" * 64})
    assert job_request.input_hash == equivalent.input_hash
    assert InferenceJobRequest.from_dict(job_request.to_dict()) == job_request
    assert replace(job_request, parameter_names=["amplitude", "Tc"]).input_hash != job_request.input_hash
    changed = job_request.to_dict()
    changed["payload"] = {"data": [9]}
    with pytest.raises(IdentityMismatch):
        InferenceJobRequest.from_dict(changed)
    with pytest.raises(ValueError):
        InferenceJobRequest({"value": float("nan")}, ["Tc"])
    with pytest.raises(ValueError):
        InferenceJobRequest({}, ["Tc", "Tc"])


@pytest.mark.parametrize("path", [".", "../x", "/absolute", "a/../x", "a//x"])
def test_artifact_paths_are_relative_normalized(path):
    with pytest.raises(ValueError):
        ArtifactRef(path, "a" * 64)


def test_local_prepare_and_default_remote_gate(tmp_path, profile, job_request):
    runner = FakeRunner()
    backend = InferenceJobBackend(profile, tmp_path, runner=runner)
    manifest = backend.prepare(job_request, COMMAND)
    assert manifest["state"] == "prepared"
    assert not runner.calls
    script = (tmp_path / profile.profile_id / job_request.request_id / "bundle/submit.slurm").read_text()
    assert "--cpus-per-task=8" in script
    assert "#SBATCH --no-requeue" in script
    assert "#SBATCH --requeue" not in script
    assert "ANDIE_INPUT_HASH=" + job_request.input_hash in script
    assert "exec srun /venv/bin/python" in script
    for operation in (backend.submit, backend.status, backend.cancel, backend.fetch):
        with pytest.raises(RemoteExecutionDisabled):
            operation(job_request.request_id)
    assert not runner.calls


def test_duplicate_submit_and_restart_preserve_single_job(tmp_path, profile, job_request):
    backend, runner = submitted(tmp_path, profile, job_request)
    assert len(runner.calls) == 3
    restarted = InferenceJobBackend(profile, tmp_path, runner=runner, allow_remote=True)
    assert restarted.prepare(job_request, COMMAND)["job_id"] == "12345"
    assert restarted.submit(job_request.request_id)["job_id"] == "12345"
    assert len(runner.calls) == 3
    assert "BatchMode=yes" in runner.calls[0]
    assert "--parsable" in runner.calls[-1][-1]
    with pytest.raises(IdentityMismatch):
        restarted.prepare(job_request, COMMAND + ["--different"])
    with pytest.raises(IdentityMismatch):
        InferenceJobBackend(replace(profile, partition="normal"), tmp_path).read_manifest(job_request.request_id)


@pytest.mark.parametrize("last", [subprocess.TimeoutExpired("ssh", 120), response("unparseable"), response(returncode=255)])
def test_unknown_submission_never_retries_after_restart(tmp_path, profile, job_request, last):
    runner = FakeRunner(response(), response(), last)
    backend = prepared(tmp_path, profile, job_request, runner)
    with pytest.raises(UnknownSubmission):
        backend.submit(job_request.request_id)
    assert backend.read_manifest(job_request.request_id)["state"] == "submission_unknown"
    restarted = InferenceJobBackend(profile, tmp_path, runner=runner, allow_remote=True)
    for operation in (restarted.submit, restarted.status, restarted.cancel, restarted.fetch):
        with pytest.raises(UnknownSubmission):
            operation(job_request.request_id)
    assert len(runner.calls) == 3


def test_staging_failure_can_retry_before_submit(tmp_path, profile, job_request):
    runner = FakeRunner(response(returncode=1))
    backend = prepared(tmp_path, profile, job_request, runner)
    with pytest.raises(TransportError):
        backend.submit(job_request.request_id)
    manifest = backend.read_manifest(job_request.request_id)
    assert manifest["state"] == "stage_failed" and not manifest["submission_attempted"]
    runner.answers += [response(), response(), response("12345")]
    assert backend.submit(job_request.request_id)["job_id"] == "12345"


def test_empty_squeue_uses_sacct_exact_parent(tmp_path, profile, job_request):
    backend, runner = completed(tmp_path, profile, job_request)
    manifest = backend.read_manifest(job_request.request_id)
    assert manifest["state"] != "completed"
    assert manifest["scheduler"] == {"source": "sacct", "state": "COMPLETED", "exit_code": "0:0"}
    assert "squeue" in runner.calls[-2][-1] and "sacct" in runner.calls[-1][-1]
    runner.answers += [response(), response()]
    assert backend.status(job_request.request_id)["state"] == "unknown"
    assert backend.read_manifest(job_request.request_id)["job_id"] == "12345"


def test_cancel_request_is_not_confirmation(tmp_path, profile, job_request):
    backend, runner = submitted(tmp_path, profile, job_request)
    runner.answers += [response()]
    assert backend.cancel(job_request.request_id)["state"] == "cancel_requested"
    assert backend.cancel(job_request.request_id)["state"] == "cancel_requested"
    assert sum("scancel" in call[-1] for call in runner.calls) == 1
    runner.answers += [response(), response("12345|CANCELLED by 1000|0:15\n")]
    assert backend.status(job_request.request_id)["state"] == "cancelled"


def test_nonzero_exit_is_not_success(tmp_path, profile, job_request):
    backend, runner = submitted(tmp_path, profile, job_request)
    runner.answers += [response(), response("12345|COMPLETED|1:0")]
    assert backend.status(job_request.request_id)["state"] == "failed"
    runner.answers += [download(job_request)]
    with pytest.raises(IdentityMismatch):
        backend.fetch(job_request.request_id)


def test_fetch_validates_and_caches_exact_result(tmp_path, profile, job_request):
    backend, runner = completed(tmp_path, profile, job_request)
    runner.answers += [download(job_request)]
    result = backend.fetch(job_request.request_id)
    assert result.status == "completed"
    assert backend.read_manifest(job_request.request_id)["state"] == "completed"
    count = len(runner.calls)
    assert backend.fetch(job_request.request_id) == result
    assert len(runner.calls) == count
    posterior = tmp_path / profile.profile_id / job_request.request_id / "result/posterior.npz"
    posterior.write_bytes(b"changed")
    with pytest.raises(IdentityMismatch):
        backend.fetch(job_request.request_id)


@pytest.mark.parametrize("change", [
    lambda value, target: value.update(input_hash="0" * 64),
    lambda value, target: value.update(request_id="inference-" + "0" * 24),
    lambda value, target: value.update(job_id="99999"),
    lambda value, target: value.update(parameter_names=["amplitude", "Tc"]),
    lambda value, target: (target / "posterior.npz").write_bytes(b"tampered"),
])
def test_stale_or_invalid_result_never_becomes_success(tmp_path, profile, job_request, change):
    backend, runner = completed(tmp_path, profile, job_request)
    runner.answers += [download(job_request, change)]
    with pytest.raises(IdentityMismatch):
        backend.fetch(job_request.request_id)
    manifest = backend.read_manifest(job_request.request_id)
    assert manifest["state"] == "scheduler_completed"
    assert "result_sha256" not in manifest
    assert not list((tmp_path / profile.profile_id / job_request.request_id).glob(".fetch-*"))


def test_missing_current_result_does_not_pick_old_file(tmp_path, profile, job_request):
    backend, runner = completed(tmp_path, profile, job_request)
    old = tmp_path / profile.profile_id / job_request.request_id / "old"
    old.mkdir()
    download(job_request)(["scp", str(old)])
    runner.answers += [response()]
    with pytest.raises(FileNotFoundError):
        backend.fetch(job_request.request_id)
    assert backend.read_manifest(job_request.request_id)["state"] == "scheduler_completed"


def test_publish_manifest_crash_recovers_verified_result(tmp_path, profile, job_request):
    backend, runner = completed(tmp_path, profile, job_request)
    target = tmp_path / profile.profile_id / job_request.request_id / "result"
    download(job_request)(["scp", str(target)])
    count = len(runner.calls)
    assert backend.fetch(job_request.request_id).status == "completed"
    assert backend.read_manifest(job_request.request_id)["events"][-1]["action"] == "result_recovered"
    assert len(runner.calls) == count


def test_checkpoint_identity_and_artifact_validation(tmp_path, profile, job_request):
    source = tmp_path / "checkpoint.hdf"
    source.write_bytes(b"checkpoint")
    backend = InferenceJobBackend(profile, tmp_path / "state")
    checkpoint = CheckpointRef(job_request.input_hash, job_request.parameter_names,
                               ArtifactRef.from_file("session.hdf", source), backend.runtime_fingerprint(COMMAND))
    for changed in (replace(checkpoint, input_hash="0" * 64),
                    replace(checkpoint, parameter_names=["amplitude", "Tc"]),
                    replace(checkpoint, runtime_fingerprint="other")):
        with pytest.raises(IdentityMismatch):
            backend.prepare(job_request, COMMAND, checkpoint=changed, checkpoint_source=source)
    manifest = backend.prepare(job_request, COMMAND, checkpoint=checkpoint, checkpoint_source=source)
    assert manifest["checkpoint"]["input_hash"] == job_request.input_hash
    script = (tmp_path / "state" / profile.profile_id / job_request.request_id / "bundle/submit.slurm").read_text()
    assert "#SBATCH --no-requeue" in script
    assert "#SBATCH --requeue" not in script
    source.write_bytes(b"wrong")
    other = InferenceJobBackend(profile, tmp_path / "other")
    with pytest.raises(IdentityMismatch):
        other.prepare(job_request, COMMAND, checkpoint=checkpoint, checkpoint_source=source)


def test_input_artifacts_stage_next_to_request_and_cannot_change(tmp_path, profile):
    source = tmp_path / "worker.py"
    source.write_text("# inert test artifact\n")
    ref = ArtifactRef.from_file("worker.py", source)
    job_request = InferenceJobRequest({}, ["Tc"], [ref])
    runner = FakeRunner()
    backend = prepared(tmp_path / "state", profile, job_request, runner, input_sources={"worker.py": source})
    bundle = tmp_path / "state" / profile.profile_id / job_request.request_id / "bundle"
    ref.validate_file(bundle)
    (bundle / "worker.py").write_text("changed")
    with pytest.raises(IdentityMismatch):
        backend.submit(job_request.request_id)
    assert not runner.calls


def test_profiles_keep_distinct_runtimes_separate(tmp_path):
    profiles = []
    for name in ("cluster_a", "cluster_b"):
        path = tmp_path / (name + ".yaml")
        path.write_text(json.dumps({
            "scheduler": "slurm", "profile_id": name,
            "login_host": name + ".example", "remote_root": "/remote/" + name,
            "default_partition": "normal", "recorded_not_live_verified": True,
            "inference": {"cpus_per_task": 8, "python_executable": "/venv/bin/python"},
        }))
        profiles.append(HPCProfile.from_file(path))
    assert [p.profile_id for p in profiles] == ["cluster_a", "cluster_b"]
    assert all(p.recorded_not_live_verified and p.cpus_per_task == 8 for p in profiles)
    assert all(p.partition == "normal" and p.python_executable == "/venv/bin/python" for p in profiles)
    assert profiles[0].fingerprint != profiles[1].fingerprint


def test_failed_worker_result_can_be_fetched_for_diagnostics(tmp_path, profile, job_request):
    backend, runner = submitted(tmp_path, profile, job_request)
    runner.answers += [response(), response("12345|FAILED|2:0")]
    assert backend.status(job_request.request_id)["state"] == "failed"
    def failure(value, target):
        value.update(status="failed", posterior_artifacts=[], diagnostics={"error": "sampler unavailable"})
    runner.answers += [download(job_request, failure)]
    assert backend.fetch(job_request.request_id).status == "failed"
    assert backend.read_manifest(job_request.request_id)["state"] == "result_failed"


def test_running_job_cannot_publish_a_result(tmp_path, profile, job_request):
    backend, runner = submitted(tmp_path, profile, job_request)
    runner.answers += [response("12345|RUNNING")]
    assert backend.status(job_request.request_id)["state"] == "running"
    with pytest.raises(InferenceJobError):
        backend.fetch(job_request.request_id)
    assert len(runner.calls) == 4


def test_symlink_or_unregistered_extra_input_is_rejected(tmp_path, profile, job_request):
    runner = FakeRunner()
    backend = prepared(tmp_path, profile, job_request, runner)
    bundle = tmp_path / profile.profile_id / job_request.request_id / "bundle"
    (bundle / "extra").write_text("unregistered")
    with pytest.raises(IdentityMismatch):
        backend.submit(job_request.request_id)
    (bundle / "extra").unlink()
    (bundle / "extra").symlink_to(tmp_path)
    with pytest.raises(IdentityMismatch):
        backend.submit(job_request.request_id)
    assert not runner.calls
