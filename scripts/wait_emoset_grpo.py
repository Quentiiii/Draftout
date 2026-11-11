#!/usr/bin/env python3
"""Start the default GRPO launcher once two GPUs are idle; do not retry training."""

import csv
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from datetime import datetime
from zoneinfo import ZoneInfo


PROJECT = Path(__file__).resolve().parents[1]
LAUNCHER = PROJECT / "scripts/run_emoset_grpo.sh"
LOG_DIR = PROJECT / "outputs/emoset_grpo/logs"
POLL_SECONDS = 30
CONFIRM_SECONDS = 3


def gpu_snapshot():
    def query(fields):
        result = subprocess.run(
            ["nvidia-smi", fields, "--format=csv,noheader,nounits"],
            text=True, capture_output=True, check=True, timeout=15,
        )
        return list(csv.reader(result.stdout.strip().splitlines(), skipinitialspace=True))

    rows = query("--query-gpu=index,uuid,memory.used,utilization.gpu")
    if not rows:
        raise RuntimeError("No GPU information returned")
    gpus = {
        row[1]: {"index": int(row[0]), "uuid": row[1],
                 "used_mib": int(row[2]), "utilization": int(row[3]), "pids": []}
        for row in rows
    }
    for uuid, pid in query("--query-compute-apps=gpu_uuid,pid"):
        if uuid not in gpus:
            raise RuntimeError(f"Unrecognized GPU in process query: {uuid}")
        gpus[uuid]["pids"].append(int(pid))
    return sorted(gpus.values(), key=lambda gpu: gpu["index"])


def idle_gpus(snapshot):
    return [gpu for gpu in snapshot if not gpu["pids"]
            and gpu["used_mib"] <= 1024 and gpu["utilization"] <= 5]


def is_training_command(argv):
    if not argv:
        return False
    executable = Path(argv[0]).name
    if executable.startswith("python"):
        return any(arg == "-m" and argv[index + 1] == "emotion_grpo.train"
                   for index, arg in enumerate(argv[:-1]))
    if executable in ("bash", "sh", "dash", "zsh"):
        for arg in argv[1:]:
            if arg.startswith("-"):
                if not arg.startswith("--") and "c" in arg[1:]:
                    return False
                continue
            return Path(arg).name == LAUNCHER.name
        return False
    return executable == LAUNCHER.name


def existing_training():
    """Match real argv entries, not diagnostic shell text mentioning the launcher."""
    found = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            argv = (proc / "cmdline").read_bytes().decode(errors="replace").split("\0")
            argv = [arg for arg in argv if arg]
        except FileNotFoundError:
            continue
        if is_training_command(argv):
            found.append(int(proc.name))
    return found


def wait_and_run(status_path, training_log):
    state = {"watcher_pid": os.getpid(), "launcher": str(LAUNCHER),
             "training_log": str(training_log), "launch_attempted": False,
             "poll_seconds": POLL_SECONDS}

    def record(status, **updates):
        state.update(status=status, updated_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                     **updates)
        temporary = status_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(status_path)
        print(json.dumps(state, ensure_ascii=False), flush=True)

    def cancelled(signum, _frame):
        record("monitor_stopped", signal=signum)
        raise SystemExit(128 + signum)

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, cancelled)
    try:
        while True:
            active = existing_training()
            if active:
                record("not_started_existing_training", existing_pids=active)
                return 0
            try:
                snapshot = gpu_snapshot()
                available = idle_gpus(snapshot)
                record("waiting", gpu_snapshot=snapshot,
                       available_gpus=[gpu["index"] for gpu in available], query_error=None)
                if len(available) < 2:
                    time.sleep(POLL_SECONDS)
                    continue
                selected = available[:2]
                time.sleep(CONFIRM_SECONDS)
                confirmed = {gpu["uuid"]: gpu for gpu in idle_gpus(gpu_snapshot())}
                if not all(gpu["uuid"] in confirmed
                           and confirmed[gpu["uuid"]]["index"] == gpu["index"] for gpu in selected):
                    time.sleep(POLL_SECONDS)
                    continue
            except (subprocess.SubprocessError, OSError, ValueError, RuntimeError, IndexError) as exc:
                record("waiting", query_error=str(exc), available_gpus=[])
                time.sleep(POLL_SECONDS)
                continue
            active = existing_training()
            if active:
                record("not_started_existing_training", existing_pids=active)
                return 0
            devices = ",".join(str(gpu["index"]) for gpu in selected)
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = devices
            command = ["bash", str(LAUNCHER)]
            record("starting", launch_attempted=True, cuda_visible_devices=devices,
                   selected_gpu_uuids=[gpu["uuid"] for gpu in selected], command=command)
            with training_log.open("a") as output:
                child = subprocess.Popen(command, cwd=PROJECT, env=environment,
                                         stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                         start_new_session=True)
                record("running", training_pid=child.pid)
                exit_code = child.wait()
            record("finished" if exit_code == 0 else "failed", exit_code=exit_code)
            return exit_code
    except Exception as exc:
        record("failed", error=f"{type(exc).__name__}: {exc}")
        raise


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with (LOG_DIR / "auto_start.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("A GRPO waiter/training supervisor is already active.", flush=True)
            return 1
        return wait_and_run(LOG_DIR / "auto_start_status.json", LOG_DIR / "auto_start_train.log")


if __name__ == "__main__":
    raise SystemExit(main())
