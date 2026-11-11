#!/usr/bin/env python3
"""Wait for physical GPUs 2 and 3, then resume the current Draft experiment once."""

import fcntl
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import yaml

from wait_emoset_grpo import LOG_DIR, PROJECT, wait_and_run


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    # Share the ordinary launcher's lock throughout waiting and training.
    with (LOG_DIR / "auto_start.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("A GRPO waiter/training supervisor is already active.", flush=True)
            return 1
        config = yaml.safe_load((PROJECT / "verl/trainer/config/emoset_draft_grpo.yaml").read_text())
        experiment = config["trainer"]["experiment_name"]
        directory = PROJECT / "outputs/emoset_grpo/checkpoints" / experiment
        # This entry point is for continuation of an existing experiment.
        # The training launcher itself rechecks the contract and loads latest.
        step = int((directory / "latest_checkpointed_iteration.txt").read_text().strip())
        checkpoint = directory / f"global_step_{step}"
        saved = json.loads((checkpoint / "draft_advantage_state.json").read_text())
        if saved["global_step"] != step or saved["warmup_steps_per_launch"] != config["draft_grpo"]["warmup_steps"]:
            raise ValueError("Latest checkpoint metadata does not match the current Draft recipe.")
        contract = json.loads((directory / "prompt_contract.json").read_text())
        stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
        return wait_and_run(
            LOG_DIR / "auto_start_status.json",
            LOG_DIR / f"auto_start_draft_train_{stamp}.log",
            launcher=PROJECT / "scripts/run_emoset_draft_grpo.sh",
            requested_gpus=(2, 3),
            state_metadata={
                "experiment_name": experiment,
                "checkpoint_dir": str(directory),
                "checkpoint_at_watch_start": step,
                "resume_mode": "auto",
                "warmup_steps_on_fresh_start": config["draft_grpo"]["warmup_steps"],
                "warmup_steps_on_resume": (
                    config["draft_grpo"]["warmup_steps"] if config["draft_grpo"].get("warmup_on_resume", False) else 0
                ),
                "prompt_sha256": contract["prompt_sha256"],
                "invalid_draft_policy": contract["invalid_draft_policy"],
            },
        )


if __name__ == "__main__":
    raise SystemExit(main())
