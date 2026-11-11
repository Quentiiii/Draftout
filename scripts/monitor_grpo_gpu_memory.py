#!/usr/bin/env python3
"""Read-only NVML sampling until a particular training process has exited."""

import argparse
import csv
import json
import os
from pathlib import Path
import pwd
import re
import time
from datetime import datetime

import pynvml as nvml


MIB = 1024 ** 2


def timestamp():
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def process(pid):
    try:
        root = Path(f"/proc/{pid}")
        raw = (root / "stat").read_text()
        fields = raw[raw.rfind(")") + 2:].split()
        uid = root.stat().st_uid
        return {"pid": pid, "ppid": int(fields[1]), "state": fields[0],
                "start_ticks": int(fields[19]), "user": pwd.getpwuid(uid).pw_name,
                "name": (root / "comm").read_text().strip()}
    except (OSError, ValueError, KeyError):
        return None


def descendant(pid, ancestor):
    seen = set()
    while pid > 1 and pid not in seen:
        if pid == ancestor:
            return True
        seen.add(pid)
        info = process(pid)
        if info is None:
            return False
        pid = info["ppid"]
    return False


def atomic_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


class Progress:
    def __init__(self, session):
        self.logs = Path(session) / "logs"
        self.offsets = {}
        self.last_step = None
        self.driver_log = None

    def refresh(self):
        if self.driver_log is None:
            for path in self.logs.glob("worker*.out"):
                with path.open(errors="replace") as stream:
                    if "TaskRunner" in stream.read(160):
                        self.driver_log = path
                        break
        if self.driver_log is not None:
            with self.driver_log.open(errors="replace") as stream:
                stream.seek(self.offsets.get(self.driver_log, 0))
                content = stream.read()
                self.offsets[self.driver_log] = stream.tell()
            matches = re.findall(r"(?:^|\n)step:(\d+)\s+-", content)
            if matches:
                self.last_step = int(matches[-1])
        return self.last_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver-pid", type=int, required=True)
    parser.add_argument("--ray-session", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--tail-seconds", type=float, default=30)
    args = parser.parse_args()
    if args.interval <= 0 or args.tail_seconds < 0:
        parser.error("interval must be positive and tail-seconds nonnegative")
    original_driver = process(args.driver_pid)
    if original_driver is None or original_driver["state"] == "Z":
        raise SystemExit("Training driver is not running; no monitor started.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    if summary_path.exists():
        raise SystemExit("Output already contains a monitor summary; use a new directory.")
    nvml.nvmlInit()
    handles = [nvml.nvmlDeviceGetHandleByIndex(i) for i in range(nvml.nvmlDeviceGetCount())]
    metadata = [{"index": i, "uuid": nvml.nvmlDeviceGetUUID(handle)}
                for i, handle in enumerate(handles)]
    summary = {"status": "monitoring", "monitor_pid": os.getpid(),
               "driver": original_driver, "ray_session": args.ray_session,
               "started_at": timestamp(), "sampling_interval_seconds": args.interval,
               "process_sampling_interval_seconds": 1.0, "sample_count": 0,
               "last_completed_step": None, "peaks": {}, "process_peaks": {},
               "notes": "Sampled whole-device usage, including other processes; brief spikes between samples may be missed."}
    progress = Progress(args.ray_session)
    snapshots = {}
    last_process_poll = last_summary = 0.0
    exited_at = None
    samples = (args.output_dir / "gpu_memory.csv").open("x", newline="", buffering=1)
    processes = (args.output_dir / "processes.jsonl").open("x", buffering=1)
    writer = csv.writer(samples)
    writer.writerow(["timestamp", "gpu", "used_mib", "free_mib", "total_mib", "utilization_pct", "last_completed_step"])
    atomic_json(summary_path, summary)
    print(f"Monitoring PID {args.driver_pid}; samples: {args.output_dir}", flush=True)
    try:
        while True:
            loop_start = time.monotonic()
            now = timestamp()
            current_driver = process(args.driver_pid)
            alive = (current_driver is not None and current_driver["state"] != "Z"
                     and current_driver["start_ticks"] == original_driver["start_ticks"])
            if not alive and exited_at is None:
                exited_at = loop_start
                summary["driver_exit_observed_at"] = now
                summary["status"] = "collecting_exit_tail"
            poll_processes = loop_start - last_process_poll >= 1.0
            if poll_processes:
                progress.refresh()
                summary["last_completed_step"] = progress.last_step
            current = []
            for info, handle in zip(metadata, handles):
                index = info["index"]
                if poll_processes:
                    rows = []
                    for item in nvml.nvmlDeviceGetComputeRunningProcesses(handle):
                        identity = process(item.pid) or {"pid": item.pid}
                        used = item.usedGpuMemory
                        row = dict(identity, memory_mib=None if used is None or used > 2 ** 60 else used / MIB,
                                   descends_from_training_driver=descendant(item.pid, args.driver_pid))
                        rows.append(row)
                        key = f"{index}:{item.pid}:{identity.get('start_ticks', 'unknown')}"
                        previous = summary["process_peaks"].get(key)
                        if row["memory_mib"] is not None and (previous is None or row["memory_mib"] > previous["memory_mib"]):
                            summary["process_peaks"][key] = dict(row, gpu=index, timestamp=now)
                    snapshots[index] = {"timestamp": now, "processes": rows}
                    processes.write(json.dumps(dict(snapshots[index], gpu=index)) + "\n")
                memory = nvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = nvml.nvmlDeviceGetUtilizationRates(handle).gpu
                reading = dict(info, timestamp=now, used_mib=memory.used / MIB,
                               free_mib=memory.free / MIB, total_mib=memory.total / MIB,
                               utilization_pct=utilization, last_completed_step=progress.last_step)
                current.append(reading)
                writer.writerow([now, index, reading["used_mib"], reading["free_mib"],
                                 reading["total_mib"], utilization, progress.last_step])
                previous = summary["peaks"].get(str(index))
                if previous is None or reading["used_mib"] > previous["used_mib"]:
                    summary["peaks"][str(index)] = dict(reading, recent_process_snapshot=snapshots.get(index))
            summary.update(updated_at=now, driver_alive=alive, current=current,
                           sample_count=summary["sample_count"] + 1)
            if poll_processes:
                last_process_poll = loop_start
            if loop_start - last_summary >= 1.0:
                atomic_json(summary_path, summary)
                last_summary = loop_start
            if exited_at is not None and loop_start - exited_at >= args.tail_seconds:
                summary["status"] = "finished"
                summary["finished_at"] = now
                break
            time.sleep(max(0, args.interval - (time.monotonic() - loop_start)))
    except BaseException as error:
        summary.update(status="monitor_error", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        atomic_json(summary_path, summary)
        samples.close()
        processes.close()
        nvml.nvmlShutdown()


if __name__ == "__main__":
    main()
