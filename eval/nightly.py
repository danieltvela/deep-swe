#!/usr/bin/env python3
"""Overnight batched runner for the local DeepSWE harness.

Works through the whole 113-task corpus in a fixed, reproducible order, using
only a nightly window (sequential, one model endpoint), cleaning pier's docker
images between batches so the disk never fills, and persisting progress so it
resumes on later nights. Meant to be launched detached, e.g.:

    setsid nohup python3 eval/nightly.py --model mimo-v2.6-flash \
        --start-at 22:00 --budget-hours 9.5 --task-timeout-min 60 \
        >> results/nightly.log 2>&1 &

State:   results/nightly-state.json
Log:     results/nightly.log (when redirected)
Report:  regenerated at the end of every window.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlsplit

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "eval"))

import deepswe_eval as H  # noqa: E402

STATE_PATH = REPO / "results" / "nightly-state.json"
EVAL_SCRIPT = REPO / "eval" / "deepswe_eval.py"
AGENT_TIMEOUT_SEC = 10800.0  # task.toml per-task agent timeout (3 h)


def log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def build_order(manifest: dict, seed: int, languages: list[str] | None) -> list[str]:
    """Deterministic, language-balanced order over all selected tasks.

    Tasks are grouped by language, shuffled within each language with the seed,
    then interleaved round-robin so every batch spans several languages and two
    models evaluated on the same order see the same task sequence.
    """
    tasks = {
        tid: meta["language"]
        for tid, meta in manifest["tasks"].items()
        if not languages or meta["language"] in languages
    }
    by_language: dict[str, list[str]] = {}
    for tid, lang in tasks.items():
        by_language.setdefault(lang, []).append(tid)
    rng = random.Random(seed)
    for lang in by_language:
        by_language[lang].sort()
        rng.shuffle(by_language[lang])

    order: list[str] = []
    queues = [by_language[lang] for lang in sorted(by_language)]
    index = 0
    while any(queues):
        queue = queues[index % len(queues)]
        if queue:
            order.append(queue.pop(0))
        index += 1
    return order


def load_state(args: argparse.Namespace, manifest: dict) -> dict:
    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    else:
        state = {}
    if not state.get("order"):
        state = {
            "created_at": datetime.now().isoformat(),
            "seed": args.seed,
            "languages": args.languages or manifest["enabled_languages"],
            "order": build_order(manifest, args.seed, args.languages),
            "completed": {},
            "nights": [],
        }
        save_state(state)
    return state


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def parse_hhmm(value: str) -> tuple[int, int]:
    hour, minute = value.split(":")
    return int(hour), int(minute)


def next_occurrence(hhmm: tuple[int, int]) -> datetime:
    now = datetime.now()
    target = now.replace(hour=hhmm[0], minute=hhmm[1], second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return target


def window_remaining_min(start_at: tuple[int, int], end_at: tuple[int, int]) -> float:
    """Minutes left in the current daily window, or 0 when outside it.

    Windows may wrap midnight (22:00-05:00). This is what decides whether the
    daemon can open another window right away or has to wait for the next
    start_at: a window whose last batch overruns the start_at boundary (a
    00:00-23:59 window closing at 00:17, say) would otherwise idle ~24 h.
    """
    now = datetime.now()
    cur = now.hour * 60 + now.minute + now.second / 60
    start = start_at[0] * 60 + start_at[1]
    end = end_at[0] * 60 + end_at[1]
    if start < end:
        return max(0.0, end - cur) if start <= cur < end else 0.0
    if cur >= start:  # inside the evening part of a wrapping window
        return (24 * 60 - cur) + end
    if cur < end:  # inside the morning part
        return end - cur
    return 0.0


def wait_until(start_at: tuple[int, int], skip: bool) -> None:
    if skip:
        return
    target = next_occurrence(start_at)
    wait = (target - datetime.now()).total_seconds()
    log(f"waiting {wait / 3600:.2f} h until {target:%Y-%m-%d %H:%M}")
    # Sleep in chunks so a long wait cannot exceed a single background cap.
    while wait > 0:
        chunk = min(wait, 1800)
        time.sleep(chunk)
        wait -= chunk


def model_reachable(model_id: str) -> bool:
    """Return True only if the configured endpoint can actually generate.

    A plain /v1/models probe is not enough: with a different model profile
    running (MiMo and GLM are mutually exclusive on the same port), the gateway
    still answers 200 while the alias points at a backend that is not serving,
    and the whole window would burn on failing trials.

    The probe uses the same base URL the agent uses, so it exercises the real
    path (egress proxy -> port-80 forwarder -> whichever server the forwarder
    targets), not just LiteLLM. It accepts only a well-formed chat completion:
    a model server that ignores the requested name can still answer 200, which
    is why a registry entry can set `agent.verify_served_alias: true` to also
    require the endpoint to advertise the requested alias on /v1/models.
    """
    models = H.load_models()
    agent = (models.get(model_id, {}) or {}).get("agent") or {}
    model_name = str(agent.get("model_name") or "")
    if not model_name:
        return False
    env = {k: str(H.expand_env(str(v))) for k, v in (agent.get("env") or {}).items()}
    base = (env.get("OPENAI_BASE_URL") or "http://172.17.0.1.nip.io/v1").rstrip("/")
    if not urlsplit(base).hostname:
        return False
    key = env.get("OPENAI_API_KEY", "")
    payload = json.dumps({
        "model": model_name.split("/")[-1],
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "temperature": 0,
    })
    proc = subprocess.run(
        ["curl", "-s", "--max-time", "90",
         "-H", f"Authorization: Bearer {key}", "-H", "Content-Type: application/json",
         "-d", payload, f"{base}/chat/completions"],
        capture_output=True, text=True, check=False,
    )
    try:
        body = json.loads(proc.stdout)
    except (json.JSONDecodeError, TypeError):
        return False
    if not body.get("choices"):
        return False
    if not agent.get("verify_served_alias"):
        return True
    # Opt-in guard for servers that answer any model name. sglang and a vLLM
    # started without the requested alias both return 200 for a name they do not
    # serve, so a whole window could be recorded under a model that never ran.
    wanted = model_name.split("/")[-1]
    listed = advertised_model_ids(base, key)
    if listed is None or wanted not in listed:
        log(
            f"endpoint for {model_id} advertises {listed} but not {wanted!r}; "
            f"another model is probably holding the port"
        )
        return False
    return True


def advertised_model_ids(base: str, key: str) -> list[str] | None:
    """Model ids the endpoint lists on /v1/models, or None if unreadable."""
    proc = subprocess.run(
        ["curl", "-s", "--max-time", "30",
         "-H", f"Authorization: Bearer {key}", f"{base}/models"],
        capture_output=True, text=True, check=False,
    )
    try:
        body = json.loads(proc.stdout)
    except (json.JSONDecodeError, TypeError):
        return None
    data = body.get("data")
    if not isinstance(data, list):
        return None
    return [str(item.get("id", "")) for item in data if isinstance(item, dict)]


def unique_run_id(night: str) -> str:
    """Run id that cannot collide with a leftover job directory.

    pier refuses to reuse a job directory whose config differs, which would make
    a batch fail and (if unguarded) be misrecorded as completed.
    """
    stamp = datetime.now().strftime("%H%M%S")
    base = f"nightly-{night}-{stamp}"
    candidate = base
    suffix = 0
    while (REPO / "jobs" / candidate).exists() or (H.RUNS_DIR / f"{candidate}.json").exists():
        suffix += 1
        candidate = f"{base}-{suffix}"
    return candidate


def run_batch(run_id: str, model: str, task_ids: list[str], timeout_multiplier: float) -> int:
    cmd = [
        sys.executable, str(EVAL_SCRIPT), "run",
        "--model", model,
        "--tasks", ",".join(task_ids),
        "--run-id", run_id,
        "--n-concurrent", "1",
        "--agent-timeout-multiplier", f"{timeout_multiplier:.4f}",
    ]
    log(f"batch {run_id}: {len(task_ids)} task(s) -> {', '.join(task_ids)}")
    proc = subprocess.run(cmd, cwd=REPO)
    return proc.returncode


def evaluated_tasks(run_id: str) -> set[str]:
    """Task ids that actually produced a trial in this run."""
    record = H.RUNS_DIR / f"{run_id}.json"
    if not record.exists():
        return set()
    try:
        data = json.loads(record.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    return {t["task_id"] for t in data.get("tasks", [])}


def discard_run(run_id: str) -> None:
    """Drop a run record that produced no usable trials, keeping the report clean."""
    (H.RUNS_DIR / f"{run_id}.json").unlink(missing_ok=True)
    shutil.rmtree(H.RUNS_DIR / run_id, ignore_errors=True)
    if H.INDEX_PATH.exists():
        try:
            index = json.loads(H.INDEX_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        index = [e for e in index if e.get("run_id") != run_id]
        H.INDEX_PATH.write_text(json.dumps(index, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="model id from eval/models.yaml")
    parser.add_argument("--start-at", default="22:00", help="local time to start")
    parser.add_argument("--end-at", default="08:00",
                        help="local time the window must end by")
    parser.add_argument("--budget-hours", type=float, default=9.5,
                        help="maximum work window length per night")
    parser.add_argument("--task-timeout-min", type=float, default=60.0,
                        help="per-task agent cap in minutes (10800s is the task's own cap)")
    parser.add_argument("--expected-task-min", type=float, default=40.0,
                        help="planning estimate for batch sizing")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--languages", help="comma-separated language filter")
    parser.add_argument("--max-tasks", type=int, help="stop after N tasks overall")
    parser.add_argument("--max-attempts", type=int, default=2,
                        help="give up on a task after this many failed batches")
    parser.add_argument("--once", action="store_true", help="run a single window then exit")
    parser.add_argument("--start-now", action="store_true", help="ignore --start-at")
    parser.add_argument("--state", help="override the state file path")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and exit without running piers")
    args = parser.parse_args()

    global STATE_PATH
    if args.state:
        STATE_PATH = Path(args.state)

    languages = (
        [l.strip() for l in args.languages.split(",") if l.strip()]
        if args.languages else None
    )
    manifest = H.load_manifest()
    state = load_state(args, manifest)
    order = state["order"]
    completed = state["completed"]
    attempts = state.setdefault("attempts", {})
    timeout_multiplier = (args.task_timeout_min * 60) / AGENT_TIMEOUT_SEC

    def pending() -> list[str]:
        return [
            t for t in order
            if t not in completed and attempts.get(t, 0) < args.max_attempts
        ]

    remaining = pending()
    log(
        f"model={args.model} order={len(order)} remaining={len(remaining)} "
        f"task_cap={args.task_timeout_min}min window={args.budget_hours}h "
        f"free={H.disk_free_gb():.0f}GB"
    )
    if not os.environ.get("LITELLM_MASTER_KEY"):
        log("WARNING: LITELLM_MASTER_KEY is not set; model calls will fail")

    if args.dry_run:
        size = max(1, int(args.budget_hours * 60 // max(args.task_timeout_min, 1)))
        log(f"dry-run: next batch would be {remaining[:size]}")
        return 0

    if not remaining:
        log("nothing to do; all tasks completed")
        subprocess.run([sys.executable, str(EVAL_SCRIPT), "report"], cwd=REPO)
        return 0

    start_at = parse_hhmm(args.start_at)
    end_at = parse_hhmm(args.end_at)
    first_window = True
    while True:
        # Open another window now only when the current daily window still has
        # room for a whole task; otherwise wait for the next start_at. Without
        # the first case, a window closing just past midnight (because its last
        # batch overran) waits for the next day's start_at, idling ~24 h.
        room = window_remaining_min(start_at, end_at)
        wait_until(start_at, args.start_now and first_window or room >= args.task_timeout_min)
        window_end = min(
            datetime.now() + timedelta(hours=args.budget_hours),
            next_occurrence(end_at),
        )
        first_window = False
        log(f"window open until {window_end:%Y-%m-%d %H:%M}")
        batches = 0
        consecutive_failures = 0

        while datetime.now() < window_end:
            remaining = pending()
            if not remaining:
                break
            if args.max_tasks and len(completed) >= args.max_tasks:
                break
            if not model_reachable(args.model):
                log(f"model alias for {args.model} cannot generate; retrying in 60s")
                time.sleep(60)
                continue

            remaining_min = (window_end - datetime.now()).total_seconds() / 60
            if remaining_min < min(15.0, args.task_timeout_min):
                log("not enough window left for another task")
                break
            size = max(1, int(remaining_min // args.task_timeout_min))
            size = min(size, len(remaining))
            batch = remaining[:size]

            night = datetime.now().strftime("%Y%m%d")
            run_id = unique_run_id(night)
            for tid in batch:
                attempts[tid] = attempts.get(tid, 0) + 1
            save_state(state)
            run_batch(run_id, args.model, batch, timeout_multiplier)
            done = evaluated_tasks(run_id)
            removed = H.cleanup_pier_images()
            log(f"cleaned {removed} pier image(s); free={H.disk_free_gb():.0f}GB")

            finished = datetime.now().isoformat()
            for tid in done:
                completed[tid] = {"run_id": run_id, "finished_at": finished}
            batches += 1

            if done:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                log(
                    f"batch {run_id} produced no trials "
                    f"({consecutive_failures} in a row); not marking tasks done"
                )
                discard_run(run_id)
                if consecutive_failures >= 3:
                    log("three consecutive empty batches; closing the window")
                    save_state(state)
                    break
            save_state(state)

        state.setdefault("nights", []).append(
            {"date": datetime.now().strftime("%Y-%m-%d"), "batches": batches,
             "completed": len(completed), "remaining": len(pending())}
        )
        save_state(state)
        given_up = [t for t in order if t not in completed
                    and attempts.get(t, 0) >= args.max_attempts]
        log(
            f"window closed: {len(completed)}/{len(order)} tasks completed, "
            f"{len(pending())} pending, {len(given_up)} given up"
        )
        if given_up:
            log(f"given up on: {', '.join(given_up)}")
        subprocess.run([sys.executable, str(EVAL_SCRIPT), "report"], cwd=REPO)

        if args.once or len(completed) >= len(order):
            break
        if args.max_tasks and len(completed) >= args.max_tasks:
            break

    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
