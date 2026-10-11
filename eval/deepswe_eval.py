#!/usr/bin/env python3
"""Local evaluation harness for the DeepSWE benchmark.

Subcommands:
  build      Build a language-filtered dataset under datasets/deep-swe-local
  list       Show detected languages/tasks and registered models
  run        Run one evaluation against a registered model via pier
  smoke      Fast end-to-end plumbing check (oracle, optionally a model)
  report     Generate a self-contained HTML report from stored results
  collect    Re-parse an existing pier job into a stored run record
  clean      Remove generated dataset or task docker images
  forwarder  Manage the model TCP forwarder (start/stop/status)

Only PyYAML is required beyond the standard library.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.stderr.write(
        "error: PyYAML is required (pip install pyyaml).\n"
    )
    raise SystemExit(2)

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO_ROOT / "eval"
TASKS_DIR = REPO_ROOT / "tasks"
DATASETS_DIR = REPO_ROOT / "datasets"
DATASET_NAME = "deep-swe-local"
DATASET_DIR = DATASETS_DIR / DATASET_NAME
DATASET_TASKS_DIR = DATASET_DIR / "tasks"
MANIFEST_PATH = DATASET_DIR / "manifest.json"
RESULTS_DIR = REPO_ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
INDEX_PATH = RESULTS_DIR / "index.json"
REPORT_PATH = RESULTS_DIR / "report.html"
JOBS_DIR = REPO_ROOT / "jobs"

LANGUAGES_CFG = EVAL_DIR / "languages.yaml"
MODELS_CFG = EVAL_DIR / "models.yaml"
FORWARDER_SCRIPT = EVAL_DIR / "llm_forwarder.py"
FORWARDER_CONTAINER = "deepswe-llm-forward"
FORWARDER_IMAGE = "python:3.12-slim"
# The pier egress proxy only permits ports 80 and 443, and the task's own
# environment directory must not contain a docker-compose.yaml (pier refuses
# build-time agent installs for compose tasks). A local model on an arbitrary
# port is therefore reached through a TCP forwarder bound to the docker bridge
# gateway on port 80.
FORWARDER_LISTEN = "172.17.0.1:80"
FORWARDER_TARGET = "127.0.0.1:4000"

# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #

def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_languages() -> dict:
    return load_yaml(LANGUAGES_CFG)


def load_models() -> dict:
    return load_yaml(MODELS_CFG).get("models", {})


SERVING_KEYS = (
    "context_length",
    "kv_cache_dtype",
    "tensor_parallel",
    "gpus",
    "reasoning_effort",
    "stack",
    "notes",
)


def build_serving(model: dict) -> dict:
    """Deployment configuration recorded alongside a run.

    Values come from the model's `serving` block; `reasoning_effort` also falls
    back to the agent kwargs so what was actually sent is recorded.
    """
    serving = {
        key: (model.get("serving") or {}).get(key, "unspecified")
        for key in SERVING_KEYS
    }
    agent_kwargs = (model.get("agent") or {}).get("kwargs") or {}
    if serving["reasoning_effort"] == "unspecified" and agent_kwargs.get(
        "reasoning_effort"
    ):
        serving["reasoning_effort"] = agent_kwargs["reasoning_effort"]
    return serving


def expand_env(value: str) -> str:
    """Expand ${VAR} / ${VAR:-default} using the process environment."""

    def repl(match: re.Match) -> str:
        name = match.group(1)
        default = match.group(2)
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            return default
        raise SystemExit(f"error: environment variable {name} is not set")

    return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}", repl, value)


# --------------------------------------------------------------------------- #
# Task classification
# --------------------------------------------------------------------------- #

REQUIRE_CMD_RE = re.compile(r"require_cmd\s+([A-Za-z0-9_.\-]+)")
DECLARED_LANG_RE = re.compile(r'^language\s*=\s*"([^"]+)"', re.M)


def is_task_dir(path: Path) -> bool:
    return path.is_dir() and (path / "task.toml").exists()


def declared_language(task_dir: Path) -> str | None:
    toml = task_dir / "task.toml"
    if not toml.exists():
        return None
    match = DECLARED_LANG_RE.search(toml.read_text(encoding="utf-8", errors="replace"))
    return match.group(1) if match else None


def required_commands(task_dir: Path) -> set[str]:
    test_sh = task_dir / "tests" / "test.sh"
    if not test_sh.exists():
        return set()
    text = test_sh.read_text(encoding="utf-8", errors="replace")
    return set(REQUIRE_CMD_RE.findall(text))


def classify(task_dir: Path, cfg: dict) -> str:
    """Return the language of a task, or 'unknown'."""
    task_id = task_dir.name
    overrides = cfg.get("overrides") or {}
    if task_id in overrides:
        return overrides[task_id]

    cmds = required_commands(task_dir)
    for family in cfg.get("families", []):
        if cmds & set(family.get("detect", [])):
            if family.get("split_by_declared"):
                declared = declared_language(task_dir)
                allowed = set(family.get("declared_languages", []))
                if declared in allowed:
                    return declared
                return family.get("default_language", family["name"])
            return family["name"]
    return "unknown"


def scan_tasks(cfg: dict) -> dict[str, dict]:
    tasks: dict[str, dict] = {}
    for path in sorted(TASKS_DIR.iterdir()):
        if not is_task_dir(path):
            continue
        tasks[path.name] = {
            "language": classify(path, cfg),
            "declared_language": declared_language(path),
        }
    return tasks


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #

def cmd_build(args: argparse.Namespace) -> int:
    cfg = load_languages()
    enabled = set(args.languages.split(",")) if args.languages else set(
        cfg.get("enabled_languages", [])
    )
    all_langs = {"javascript", "unknown"}
    for family in cfg["families"]:
        all_langs.add(family["name"])
        all_langs.update(family.get("declared_languages", []))
        if family.get("default_language"):
            all_langs.add(family["default_language"])
    unknown = enabled - all_langs
    if unknown:
        print(f"error: unknown language(s): {', '.join(sorted(unknown))}", file=sys.stderr)
        return 2

    tasks = scan_tasks(cfg)
    counts: dict[str, int] = {}
    for meta in tasks.values():
        counts[meta["language"]] = counts.get(meta["language"], 0) + 1

    print("Detected languages:")
    for lang in sorted(counts):
        mark = "keep" if lang in enabled else "drop"
        print(f"  {lang:<12} {counts[lang]:>3}  [{mark}]")

    expected = cfg.get("expected_counts") or {}
    mismatches = [
        (lang, expected[lang], counts.get(lang, 0))
        for lang in expected
        if counts.get(lang, 0) != expected[lang]
    ]
    if mismatches:
        print("\nCount mismatch vs eval/languages.yaml expected_counts:")
        for lang, want, got in mismatches:
            print(f"  {lang}: expected {want}, found {got}")
        if not args.no_strict:
            print("\nRe-run with --no-strict to build anyway.", file=sys.stderr)
            return 1

    selected = {
        tid: meta for tid, meta in tasks.items() if meta["language"] in enabled
    }
    if not selected:
        print("error: no tasks selected", file=sys.stderr)
        return 1

    if DATASET_TASKS_DIR.exists():
        shutil.rmtree(DATASET_TASKS_DIR)
    DATASET_TASKS_DIR.mkdir(parents=True)

    for task_id in sorted(selected):
        shutil.copytree(TASKS_DIR / task_id, DATASET_TASKS_DIR / task_id)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "enabled_languages": sorted(enabled),
        "counts": {lang: counts.get(lang, 0) for lang in sorted(counts)},
        "tasks": {tid: selected[tid] for tid in sorted(selected)},
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(
        f"\nBuilt {len(selected)} tasks into "
        f"{DATASET_TASKS_DIR.relative_to(REPO_ROOT)}"
    )
    print(f"Manifest: {MANIFEST_PATH.relative_to(REPO_ROOT)}")
    return 0


# --------------------------------------------------------------------------- #
# list
# --------------------------------------------------------------------------- #

def cmd_list(args: argparse.Namespace) -> int:
    cfg = load_languages()
    if not MANIFEST_PATH.exists():
        tasks = scan_tasks(cfg)
    else:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        tasks = manifest["tasks"]
    counts: dict[str, int] = {}
    for meta in tasks.values():
        counts[meta["language"]] = counts.get(meta["language"], 0) + 1

    print("Tasks by language:")
    for lang in sorted(counts):
        print(f"  {lang:<12} {counts[lang]:>3}")
    print(f"  {'TOTAL':<12} {sum(counts.values()):>3}")

    print("\nRegistered models:")
    for mid, model in load_models().items():
        print(
            f"  {mid:<28} name={model.get('name')} "
            f"quant={model.get('quantization')} lab={model.get('lab')} "
            f"variant={model.get('variant')}"
        )
    return 0


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #

def pier_binary() -> str:
    found = shutil.which("pier")
    if found:
        return found
    candidate = Path.home() / ".local" / "bin" / "pier"
    if candidate.exists():
        return str(candidate)
    raise SystemExit(
        "error: 'pier' not found. Install it with: uv tool install datacurve-pier"
    )


# --------------------------------------------------------------------------- #
# forwarder
# --------------------------------------------------------------------------- #

def docker_ps(name: str) -> dict | None:
    proc = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name=^{name}$",
         "--format", "{{json .}}"],
        capture_output=True, text=True, check=False,
    )
    lines = [l for l in proc.stdout.splitlines() if l.strip()]
    return json.loads(lines[0]) if lines else None


def forwarder_status() -> dict | None:
    return docker_ps(FORWARDER_CONTAINER)


def forwarder_start(
    listen: str = FORWARDER_LISTEN, target: str = FORWARDER_TARGET, force: bool = False
) -> None:
    existing = forwarder_status()
    if existing and existing.get("State") == "running":
        print(f"forwarder already running: {listen} -> {target}")
        return
    if existing:
        subprocess.run(["docker", "rm", "-f", FORWARDER_CONTAINER], check=False)
    if not FORWARDER_SCRIPT.exists():
        raise SystemExit(f"error: missing {FORWARDER_SCRIPT}")
    cmd = [
        "docker", "run", "-d", "--name", FORWARDER_CONTAINER,
        "--restart", "unless-stopped", "--network", "host",
        "-v", f"{FORWARDER_SCRIPT}:/llm_forwarder.py:ro",
        FORWARDER_IMAGE,
        "python", "/llm_forwarder.py", "--listen", listen, "--target", target,
    ]
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"forwarder started: {listen} -> {target}")


def forwarder_stop() -> None:
    if forwarder_status():
        subprocess.run(["docker", "rm", "-f", FORWARDER_CONTAINER], check=False)
        print("forwarder stopped")
    else:
        print("forwarder not running")


def ensure_forwarder() -> None:
    existing = forwarder_status()
    if not existing or existing.get("State") != "running":
        print("Starting model forwarder (required: egress proxy allows ports 80/443 only).")
        forwarder_start()


def cmd_forwarder(args: argparse.Namespace) -> int:
    if args.action == "start":
        forwarder_start(args.listen, args.target)
    elif args.action == "stop":
        forwarder_stop()
    else:
        status = forwarder_status()
        if status and status.get("State") == "running":
            print(f"running: {status.get('Status')} ports={status.get('Ports')}")
        else:
            print("not running")
    return 0


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise SystemExit(
            "error: dataset not built. Run: python eval/deepswe_eval.py build"
        )
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def resolve_tasks(args: argparse.Namespace, manifest: dict) -> list[str]:
    all_tasks = manifest["tasks"]

    if getattr(args, "match_run", None):
        run_path = RUNS_DIR / f"{args.match_run}.json"
        if not run_path.exists():
            raise SystemExit(f"error: run not found: {args.match_run}")
        run = json.loads(run_path.read_text(encoding="utf-8"))
        return list(run["dataset"]["task_ids"])

    if getattr(args, "tasks", None):
        ids = [t.strip() for t in args.tasks.split(",") if t.strip()]
        missing = [t for t in ids if t not in all_tasks]
        if missing:
            raise SystemExit(f"error: unknown task(s): {', '.join(missing)}")
        return ids

    langs = (
        {l.strip() for l in args.languages.split(",") if l.strip()}
        if getattr(args, "languages", None)
        else set(manifest["enabled_languages"])
    )
    ids = sorted(t for t, meta in all_tasks.items() if meta["language"] in langs)
    if not ids:
        raise SystemExit("error: no tasks match the language filter")

    n = getattr(args, "n_tasks", None)
    if n is not None and n < len(ids):
        seed = getattr(args, "sample_seed", 0) or 0
        rng = random.Random(seed)
        ids = sorted(rng.sample(ids, n))
    return ids


def build_job_config(
    *,
    run_id: str,
    task_ids: list[str],
    model: dict,
    dataset_dir: Path,
    jobs_dir: Path,
    n_concurrent: int,
    timeout_multiplier: float | None,
    agent_timeout_multiplier: float | None,
    verifier_timeout_multiplier: float | None,
    agent_override: str | None = None,
) -> dict:
    agent_cfg = dict(model.get("agent") or {})
    agent = {
        "name": agent_override or agent_cfg.get("name", "mini-swe-agent"),
        "model_name": agent_cfg.get("model_name"),
        "env": {
            key: expand_env(str(value))
            for key, value in (agent_cfg.get("env") or {}).items()
        },
        "kwargs": agent_cfg.get("kwargs") or {},
    }
    if agent_override == "oracle":
        agent["model_name"] = None
        agent["env"] = {}
        agent["kwargs"] = {}

    config: dict = {
        "job_name": run_id,
        "jobs_dir": str(jobs_dir),
        "n_concurrent_trials": n_concurrent,
        "agents": [agent],
        "datasets": [
            {"path": str(dataset_dir), "task_names": list(task_ids)}
        ],
        "environment": {"type": "docker"},
    }
    if timeout_multiplier:
        config["timeout_multiplier"] = timeout_multiplier
    if agent_timeout_multiplier:
        config["agent_timeout_multiplier"] = agent_timeout_multiplier
    if verifier_timeout_multiplier:
        config["verifier_timeout_multiplier"] = verifier_timeout_multiplier
    return config


def run_pier(config_path: Path) -> tuple[int, float]:
    cmd = [pier_binary(), "run", "--config", str(config_path)]
    print("+ " + " ".join(cmd))
    started = time.monotonic()
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    return proc.returncode, time.monotonic() - started


def parse_iso(value: str | None) -> datetime | None:
    """Parse an ISO timestamp, interpreting naive values as local time.

    pier writes job timestamps without a timezone offset; the harness writes
    UTC. Normalizing to aware datetimes keeps ordering and durations correct.
    """
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()
    return parsed


def phase_seconds(result: dict, key: str) -> float | None:
    phase = result.get(key) or {}
    start = parse_iso(phase.get("started_at"))
    end = parse_iso(phase.get("finished_at"))
    return (end - start).total_seconds() if start and end else None


def read_trajectory_metrics(trial_dir: Path) -> dict:
    """Token/reasoning metrics from the ATIF trajectory, when present.

    pier's trajectory `final_metrics` carries prompt/completion totals plus
    `extra.total_reasoning_tokens`, `extra.total_text_tokens` and
    `extra.peak_context_tokens`, which is what separates reasoning-heavy models
    from fast ones.
    """
    path = trial_dir / "agent" / "trajectory.json"
    if not path.exists():
        return {}
    try:
        trajectory = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    final = trajectory.get("final_metrics") or {}
    extra = final.get("extra") or {}
    return {
        "input_tokens": final.get("total_prompt_tokens"),
        "output_tokens": final.get("total_completion_tokens"),
        "reasoning_tokens": extra.get("total_reasoning_tokens"),
        "text_tokens": extra.get("total_text_tokens"),
        "peak_context_tokens": extra.get("peak_context_tokens"),
        "steps": final.get("total_steps"),
    }


def collect_trials(job_dir: Path, task_languages: dict[str, str]) -> list[dict]:
    records: list[dict] = []
    if not job_dir.exists():
        return records
    for trial_dir in sorted(job_dir.iterdir()):
        result_path = trial_dir / "result.json"
        if not result_path.exists():
            continue
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        rewards = (result.get("verifier_result") or {}).get("rewards") or {}
        raw_name = result.get("task_name") or trial_dir.name
        # task.toml names are "<org>/<task>"; the dataset keys are directory names.
        task_name = raw_name.split("/")[-1]
        started = parse_iso(result.get("started_at"))
        finished = parse_iso(result.get("finished_at"))
        duration = (finished - started).total_seconds() if started and finished else None
        exception = result.get("exception_info") or {}

        agent_result = result.get("agent_result") or {}
        trajectory = read_trajectory_metrics(trial_dir)
        input_tokens = trajectory.get("input_tokens", agent_result.get("n_input_tokens"))
        output_tokens = trajectory.get("output_tokens", agent_result.get("n_output_tokens"))
        reasoning_tokens = trajectory.get("reasoning_tokens")
        text_tokens = trajectory.get("text_tokens")
        cache_tokens = agent_result.get("n_cache_tokens")
        peak_context = trajectory.get("peak_context_tokens") or agent_result.get(
            "peak_context_tokens"
        )
        steps = (
            trajectory.get("steps")
            or result.get("n_agent_steps")
            or agent_result.get("n_agent_steps")
        )
        agent_sec = phase_seconds(result, "agent_execution")
        environment_sec = phase_seconds(result, "environment_setup")
        verifier_sec = phase_seconds(result, "verifier")

        reasoning_ratio = (
            reasoning_tokens / output_tokens
            if reasoning_tokens is not None and output_tokens
            else None
        )
        output_tokens_per_sec = (
            output_tokens / agent_sec if output_tokens and agent_sec else None
        )
        seconds_per_step = agent_sec / steps if agent_sec and steps else None

        records.append(
            {
                "task_id": task_name,
                "task_name": raw_name,
                "language": task_languages.get(task_name, "unknown"),
                "trial_name": result.get("trial_name") or trial_dir.name,
                "reward": rewards.get("reward"),
                "f2p_passed": rewards.get("f2p_passed"),
                "f2p_total": rewards.get("f2p_total"),
                "p2p_passed": rewards.get("p2p_passed"),
                "p2p_total": rewards.get("p2p_total"),
                "partial": rewards.get("partial"),
                "apply_failed": rewards.get("apply_failed"),
                "status": "error" if exception else "ok",
                "exception_type": exception.get("exception_type"),
                "duration_sec": duration,
                "agent_execution_sec": agent_sec,
                "environment_setup_sec": environment_sec,
                "verifier_sec": verifier_sec,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "reasoning_tokens": reasoning_tokens,
                "text_tokens": text_tokens,
                "cache_tokens": cache_tokens,
                "peak_context_tokens": peak_context,
                "steps": steps,
                "reasoning_ratio": reasoning_ratio,
                "output_tokens_per_sec": output_tokens_per_sec,
                "seconds_per_step": seconds_per_step,
                "trial_dir": str(trial_dir.relative_to(REPO_ROOT)),
            }
        )
    return records


def _sum_field(records: list[dict], key: str) -> float | None:
    values = [r[key] for r in records if r.get(key) is not None]
    return sum(values) if values else None


def summarize(records: list[dict]) -> dict:
    n = len(records)
    resolved = sum(1 for r in records if r.get("reward") == 1)
    partials = [r["partial"] for r in records if r.get("partial") is not None]
    by_language: dict[str, dict] = {}
    for rec in records:
        bucket = by_language.setdefault(
            rec["language"], {"n": 0, "resolved": 0, "partial_sum": 0.0, "partial_n": 0}
        )
        bucket["n"] += 1
        if rec.get("reward") == 1:
            bucket["resolved"] += 1
        if rec.get("partial") is not None:
            bucket["partial_sum"] += rec["partial"]
            bucket["partial_n"] += 1
    for bucket in by_language.values():
        bucket["pass_rate"] = bucket["resolved"] / bucket["n"] if bucket["n"] else 0.0
        bucket["mean_partial"] = (
            bucket["partial_sum"] / bucket["partial_n"] if bucket["partial_n"] else 0.0
        )
        del bucket["partial_sum"], bucket["partial_n"]

    # Cost and token aggregates. agent_hours is the model's own wall time with
    # the sequential default (one trial at a time), so it approximates GPU-hours.
    input_tokens = _sum_field(records, "input_tokens")
    output_tokens = _sum_field(records, "output_tokens")
    reasoning_tokens = _sum_field(records, "reasoning_tokens")
    agent_seconds = _sum_field(records, "agent_execution_sec")
    steps = _sum_field(records, "steps")
    agent_hours = agent_seconds / 3600 if agent_seconds is not None else None

    def per_task(value: float | None) -> float | None:
        return value / n if (value is not None and n) else None

    def per_resolved(value: float | None) -> float | None:
        return value / resolved if (value is not None and resolved) else None

    reasoning_ratio = (
        reasoning_tokens / output_tokens
        if reasoning_tokens is not None and output_tokens
        else None
    )

    return {
        "n_tasks": n,
        "n_resolved": resolved,
        "pass_rate": resolved / n if n else 0.0,
        "mean_partial": sum(partials) / len(partials) if partials else 0.0,
        "n_errors": sum(1 for r in records if r["status"] == "error"),
        "by_language": by_language,
        "tokens": {
            "input": input_tokens,
            "output": output_tokens,
            "reasoning": reasoning_tokens,
            "reasoning_ratio": reasoning_ratio,
            "output_per_task": per_task(output_tokens),
            "output_per_resolved": per_resolved(output_tokens),
        },
        "time": {
            "agent_seconds": agent_seconds,
            "agent_hours": agent_hours,
            "agent_hours_per_task": per_task(agent_hours),
            "agent_hours_per_resolved": per_resolved(agent_hours),
            "seconds_per_step": (
                agent_seconds / steps if agent_seconds and steps else None
            ),
        },
        "steps": steps,
        "steps_per_task": per_task(steps),
    }


def store_run(run: dict) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / f"{run['run_id']}.json").write_text(
        json.dumps(run, indent=2), encoding="utf-8"
    )
    index = []
    if INDEX_PATH.exists():
        try:
            index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            index = []
    index = [entry for entry in index if entry.get("run_id") != run["run_id"]]
    index.append(
        {
            "run_id": run["run_id"],
            "model": run["model"],
            "agent": run["endpoint"]["agent"],
            "started_at": run["started_at"],
            "duration_sec": run["duration_sec"],
            "summary": run["summary"],
            "n_tasks": run["summary"]["n_tasks"],
            "result_file": f"runs/{run['run_id']}.json",
        }
    )
    index.sort(
        key=lambda entry: parse_iso(entry.get("started_at"))
        or datetime.min.replace(tzinfo=timezone.utc)
    )
    INDEX_PATH.write_text(json.dumps(index, indent=2), encoding="utf-8")


def cmd_collect(args: argparse.Namespace) -> int:
    """Re-parse an existing pier job into a stored run record."""
    manifest = load_manifest()
    jobs_dir = Path(args.jobs_dir).resolve()
    job_dir = jobs_dir / args.job
    if not job_dir.exists():
        raise SystemExit(f"error: job directory not found: {job_dir}")

    cfg_path = job_dir / "config.json"
    job_cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    dataset_cfg = (job_cfg.get("datasets") or [{}])[0]
    task_ids = dataset_cfg.get("task_names") or sorted(manifest["tasks"])
    agent_cfg = (job_cfg.get("agents") or [{}])[0]

    models = load_models()
    if args.model and args.model in models:
        model = models[args.model]
        model_id = args.model
    else:
        model = {
            "name": args.name or agent_cfg.get("model_name") or args.model or "unknown",
            "quantization": args.quantization,
            "lab": args.lab,
            "variant": args.variant,
        }
        model_id = args.model or "unknown"

    job_result_path = job_dir / "result.json"
    job_result = (
        json.loads(job_result_path.read_text(encoding="utf-8"))
        if job_result_path.exists()
        else {}
    )
    started = parse_iso(job_result.get("started_at"))
    finished = parse_iso(job_result.get("finished_at"))
    duration = (finished - started).total_seconds() if started and finished else None

    task_languages = {
        t: manifest["tasks"].get(t, {}).get("language", "unknown") for t in task_ids
    }
    records = collect_trials(job_dir, task_languages)
    summary = summarize(records)
    run_id = args.run_id or args.job
    agent_env = agent_cfg.get("env") or {}

    run = {
        "run_id": run_id,
        "kind": args.kind,
        "model": {
            "id": model_id,
            "name": model.get("name"),
            "quantization": model.get("quantization"),
            "lab": model.get("lab"),
            "variant": model.get("variant"),
        },
        "endpoint": {
            "agent": agent_cfg.get("name"),
            "model_name": agent_cfg.get("model_name"),
            "base_url": agent_env.get("OPENAI_BASE_URL"),
        },
        "serving": build_serving(model),
        "started_at": started.isoformat() if started else None,
        "finished_at": finished.isoformat() if finished else None,
        "duration_sec": duration,
        "dataset": {
            "languages": sorted({m["language"] for m in manifest["tasks"].values()}),
            "n_tasks": len(task_ids),
            "task_ids": task_ids,
        },
        "pier": {
            "job_name": args.job,
            "jobs_dir": str(jobs_dir),
            "config_path": str(cfg_path.relative_to(REPO_ROOT)) if cfg_path.exists() else None,
            "returncode": None,
        },
        "summary": summary,
        "tasks": records,
    }
    store_run(run)
    print(
        f"Collected {len(records)} trial(s) from {args.job}: "
        f"{summary['n_resolved']}/{summary['n_tasks']} resolved"
    )
    print(f"Stored: results/runs/{run_id}.json")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    manifest = load_manifest()
    models = load_models()
    if args.model and args.model in models:
        model = models[args.model]
    elif args.agent == "oracle" and not args.model:
        model = {
            "name": "oracle",
            "quantization": "n/a",
            "lab": "n/a",
            "variant": "n/a",
        }
    else:
        raise SystemExit(
            f"error: unknown model '{args.model}'. Known: {', '.join(models)}"
        )

    task_ids = resolve_tasks(args, manifest)
    task_languages = {t: manifest["tasks"][t]["language"] for t in task_ids}

    run_id = args.run_id or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + f"_{args.model or args.agent or 'run'}"
    )
    jobs_dir = Path(args.jobs_dir).resolve()
    config = build_job_config(
        run_id=run_id,
        task_ids=task_ids,
        model=model,
        dataset_dir=DATASET_TASKS_DIR,
        jobs_dir=jobs_dir,
        n_concurrent=args.n_concurrent,
        timeout_multiplier=args.timeout_multiplier,
        agent_timeout_multiplier=args.agent_timeout_multiplier,
        verifier_timeout_multiplier=args.verifier_timeout_multiplier,
        agent_override=args.agent,
    )

    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "job.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    print(f"Run id     : {run_id}")
    print(f"Model      : {args.model or model.get('name')} ({model.get('name')})")
    print(f"Tasks      : {len(task_ids)}")
    print(f"Job config : {config_path.relative_to(REPO_ROOT)}")

    if args.dry_run:
        print("\n--dry-run: config written, pier not invoked.")
        return 0

    if args.agent != "oracle" and not getattr(args, "no_forwarder", False):
        ensure_forwarder()

    started_at = datetime.now(timezone.utc)
    returncode, elapsed = run_pier(config_path)
    finished_at = datetime.now(timezone.utc)

    job_dir = jobs_dir / run_id
    records = collect_trials(job_dir, task_languages)
    summary = summarize(records)
    agent_cfg = model.get("agent") or {}

    run = {
        "run_id": run_id,
        "kind": args.kind,
        "model": {
            "id": args.model or model.get("name"),
            "name": model.get("name"),
            "quantization": model.get("quantization"),
            "lab": model.get("lab"),
            "variant": model.get("variant"),
        },
        "endpoint": {
            "agent": args.agent or agent_cfg.get("name", "mini-swe-agent"),
            "model_name": None if args.agent == "oracle" else agent_cfg.get("model_name"),
            "base_url": (agent_cfg.get("env") or {}).get("OPENAI_BASE_URL"),
        },
        "serving": build_serving(model),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_sec": elapsed,
        "dataset": {
            "languages": sorted({m["language"] for m in manifest["tasks"].values()}),
            "n_tasks": len(task_ids),
            "task_ids": task_ids,
        },
        "pier": {
            "job_name": run_id,
            "jobs_dir": str(jobs_dir),
            "config_path": str(config_path.relative_to(REPO_ROOT)),
            "returncode": returncode,
        },
        "summary": summary,
        "tasks": records,
    }
    store_run(run)

    print(f"\npier exit code: {returncode}")
    print(f"Trials parsed : {len(records)}")
    print(
        f"Resolved      : {summary['n_resolved']}/{summary['n_tasks']} "
        f"({summary['pass_rate']:.1%})"
    )
    print(f"Mean partial  : {summary['mean_partial']:.3f}")
    print(f"Duration      : {elapsed:.1f}s")
    print(f"Stored        : results/runs/{run_id}.json")

    if not records:
        print(
            "\nwarning: no trial results found; check the pier output above.",
            file=sys.stderr,
        )
        return 1
    return 0


# --------------------------------------------------------------------------- #
# smoke
# --------------------------------------------------------------------------- #

def pick_smoke_tasks(manifest: dict, per_language: bool, explicit: str | None) -> list[str]:
    if explicit:
        ids = [t.strip() for t in explicit.split(",") if t.strip()]
        missing = [t for t in ids if t not in manifest["tasks"]]
        if missing:
            raise SystemExit(f"error: unknown task(s): {', '.join(missing)}")
        return ids

    def patch_size(task_id: str) -> int:
        patch = TASKS_DIR / task_id / "tests" / "test.patch"
        return patch.stat().st_size if patch.exists() else 1 << 30

    if per_language:
        chosen = []
        for lang in manifest["enabled_languages"]:
            candidates = [
                t for t, meta in manifest["tasks"].items() if meta["language"] == lang
            ]
            if candidates:
                chosen.append(min(candidates, key=patch_size))
        return chosen

    return [min(manifest["tasks"], key=patch_size)]


def cmd_smoke(args: argparse.Namespace) -> int:
    manifest = load_manifest()
    task_ids = pick_smoke_tasks(manifest, args.per_language, args.task)
    print(f"Smoke tasks: {', '.join(task_ids)}\n")

    rc = 0

    print("=== 1/2 oracle smoke (validates dataset, compose, verifier, parsing) ===")
    oracle_args = argparse.Namespace(
        model=args.model or next(iter(load_models())),
        languages=None,
        tasks=",".join(task_ids),
        n_tasks=None,
        sample_seed=None,
        match_run=None,
        run_id=datetime.now(timezone.utc).strftime("smoke-oracle-%Y%m%dT%H%M%SZ"),
        jobs_dir=args.jobs_dir,
        n_concurrent=args.n_concurrent,
        timeout_multiplier=args.timeout_multiplier,
        agent_timeout_multiplier=args.timeout_multiplier,
        verifier_timeout_multiplier=args.timeout_multiplier,
        agent="oracle",
        dry_run=args.dry_run,
        kind="smoke-oracle",
    )
    rc |= cmd_run(oracle_args)

    if args.model:
        print("\n=== 2/2 model smoke (validates container -> proxy -> model) ===")
        model_args = argparse.Namespace(
            model=args.model,
            languages=None,
            tasks=",".join(task_ids),
            n_tasks=None,
            sample_seed=None,
            match_run=None,
            run_id=datetime.now(timezone.utc).strftime("smoke-model-%Y%m%dT%H%M%SZ"),
            jobs_dir=args.jobs_dir,
            n_concurrent=args.n_concurrent,
            timeout_multiplier=1.0,
            agent_timeout_multiplier=args.agent_timeout_multiplier,
            verifier_timeout_multiplier=args.verifier_timeout_multiplier,
            agent=None,
            dry_run=args.dry_run,
            kind="smoke-model",
        )
        rc |= cmd_run(model_args)
    else:
        print("\n(no --model given: skipped model smoke)")

    return rc


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { --bg:#0f1419; --panel:#1a2029; --line:#2b3442; --fg:#e6edf3;
          --muted:#8b98a5; --accent:#4c9aff; --ok:#3fb950; --bad:#f85149; }
  * { box-sizing: border-box; }
  body { margin:0; font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",
         Roboto,Helvetica,Arial,sans-serif; background:var(--bg); color:var(--fg); }
  header { padding:24px 32px; border-bottom:1px solid var(--line); }
  h1 { margin:0 0 4px; font-size:20px; }
  h2 { font-size:16px; margin:28px 0 12px; }
  .muted { color:var(--muted); font-size:12px; }
  main { padding:24px 32px 64px; max-width:1500px; }
  .panel { background:var(--panel); border:1px solid var(--line);
           border-radius:8px; padding:16px; margin-bottom:20px; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th, td { text-align:left; padding:8px 10px; border-bottom:1px solid var(--line);
           white-space:nowrap; }
  th { color:var(--muted); font-weight:600; cursor:pointer; user-select:none; }
  tr:hover td { background:#202836; }
  .num { text-align:right; font-variant-numeric:tabular-nums; }
  .ok { color:var(--ok); } .bad { color:var(--bad); }
  .bar { height:8px; border-radius:4px; background:#2b3442; overflow:hidden;
         min-width:80px; }
  .bar > span { display:block; height:100%; background:var(--accent); }
  select, input { background:#0d1117; color:var(--fg); border:1px solid var(--line);
                  border-radius:6px; padding:6px 8px; font-size:13px; }
  .row { display:flex; gap:16px; flex-wrap:wrap; align-items:center; }
  .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
           gap:12px; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:8px;
          padding:14px; }
  .card .v { font-size:24px; font-weight:600; }
  .scroll { overflow-x:auto; }
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <div class="muted">Generated __GENERATED__ &middot; __NRUNS__ run(s)</div>
</header>
<main>
  <div class="cards" id="cards"></div>

  <details class="panel">
    <summary style="cursor:pointer;font-weight:600">How to read these metrics</summary>
    <ul style="margin-top:10px">
      <li><b>reward (pass/fail)</b>: 1 only when <i>every</i> fail-to-pass test passes and
        <i>no</i> pass-to-pass test regresses. This is the official, binary DeepSWE score.</li>
      <li><b>f2p (fail-to-pass)</b>: tests that failed before the change and must pass after it.
        They measure how much of the requested behaviour was actually implemented.</li>
      <li><b>p2p (pass-to-pass)</b>: tests that already passed and must keep passing. They are the
        regression guard.</li>
      <li><b>partial</b>: (f2p_passed + p2p_passed) / (f2p_total + p2p_total). Pass-to-pass tests
        usually outnumber fail-to-pass tests many times over, so a task with 4/6 f2p and 62/62 p2p
        still reads ~0.97. Treat <b>mean f2p</b> as the honest graded signal, not partial.</li>
      <li><b>Near-miss</b>: tasks with f2p ≥ 0.9 that still score 0. They rank models that are
        close but not exact.</li>
      <li><b>p2p regressions</b>: tasks where a previously passing test broke. A real failure even
        when most f2p pass.</li>
      <li><b>95% CI</b>: Wilson interval on the pass rate. With few tasks it is wide, so read
        small differences between models with care.</li>
    </ul>
  </details>

  <h2>Models (aggregated)</h2>
  <div class="panel">
    <div class="muted" id="modelsNote"></div>
    <div class="scroll"><table id="modelsTable"></table></div>
  </div>
  <div class="panel">
    <div class="muted">Pass rate vs agent-hours per task, one point per model.
      Highlighted points are on the Pareto frontier (no other model is both cheaper
      and better). Circle size scales with output tokens per task. Agent-hours are the
      model's own wall time, which with the sequential default approximates GPU-hours.</div>
    <div id="pareto"></div>
  </div>

  <h2>Runs
    <label class="muted" style="font-size:12px;font-weight:400;margin-left:12px">
      <input type="checkbox" id="showPlumbing"> show plumbing runs (oracle / smoke)
    </label>
  </h2>
  <div class="panel scroll"><table id="runsTable"></table></div>

  <h2>Run detail</h2>
  <div class="panel">
    <div class="row">
      <label>Run <select id="runSelect"></select></label>
      <label>Language <select id="langSelect"></select></label>
      <label>Search <input id="search" type="search" placeholder="task id..."></label>
    </div>
    <div id="detailSummary" class="muted" style="margin:12px 0"></div>
    <div class="scroll"><table id="tasksTable"></table></div>
  </div>

  <h2>Compare runs (shared tasks)</h2>
  <div class="panel">
    <div class="row">
      <label>A <select id="cmpA"></select></label>
      <label>B <select id="cmpB"></select></label>
    </div>
    <div id="cmpSummary" class="muted" style="margin:12px 0"></div>
    <div class="scroll"><table id="cmpTable"></table></div>
  </div>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById('data').textContent);
const runs = DATA.runs;

function el(tag, text, cls) {
  const e = document.createElement(tag);
  if (text !== undefined) e.textContent = text;
  if (cls) e.className = cls;
  return e;
}
function pct(x) { return (x * 100).toFixed(1) + '%'; }

function renderCards() {
  const cards = document.getElementById('cards');
  const modelRuns = runs.filter(isEvalRun);
  // A task evaluated more than once (say, re-run after a protocol change) counts
  // once per model, with the most recent result — same rule as the Models table.
  const byTime = [...modelRuns].sort(
    (a, b) => String(a.started_at || '').localeCompare(String(b.started_at || ''))
  );
  const taskMap = new Map();
  byTime.forEach(r => (r.tasks || []).forEach(t => {
    taskMap.set(`${(r.model || {}).id || r.run_id}|${t.task_id}`, t);
  }));
  const tasks = [...taskMap.values()];
  const totalTasks = tasks.length;
  const totalResolved = tasks.filter(t => t.reward === 1).length;
  const totalDur = modelRuns.reduce((a, r) => a + (r.duration_sec || 0), 0);
  const fracs = [];
  let nearMiss = 0;
  tasks.forEach(t => {
    if (t.f2p_total) {
      fracs.push(t.f2p_passed / t.f2p_total);
      if (t.reward !== 1 && (t.f2p_passed / t.f2p_total) >= 0.9) nearMiss++;
    }
  });
  const meanF2p = fracs.length ? fracs.reduce((a, b) => a + b, 0) / fracs.length : null;
  const items = [
    ['Evaluation runs', modelRuns.length],
    ['Tasks evaluated', totalTasks],
    ['Resolved', totalResolved],
    ['Overall pass rate', totalTasks ? pct(totalResolved / totalTasks) : '-'],
    ['Mean f2p (graded)', meanF2p === null ? '-' : pct(meanF2p)],
    ['Near-misses', nearMiss],
    ['Total wall time', (totalDur / 3600).toFixed(1) + ' h'],
  ];
  items.forEach(([k, v]) => {
    const c = el('div', null, 'card');
    c.appendChild(el('div', v, 'v'));
    c.appendChild(el('div', k, 'muted'));
    cards.appendChild(c);
  });
}

function tableHeader(table, cols) {
  const tr = el('tr');
  cols.forEach(c => {
    const th = el('th', c.label);
    th.dataset.key = c.key;
    th.onclick = () => sortTable(table, c.key);
    tr.appendChild(th);
  });
  table.appendChild(tr);
}

function sortTable(table, key) {
  const rows = Array.from(table.querySelectorAll('tr')).slice(1);
  const dir = table.dataset.sortKey === key && table.dataset.sortDir === 'asc' ? -1 : 1;
  table.dataset.sortKey = key; table.dataset.sortDir = dir === 1 ? 'asc' : 'desc';
  rows.sort((a, b) => {
    const av = a.dataset['v_' + key] ?? '', bv = b.dataset['v_' + key] ?? '';
    const an = parseFloat(av), bn = parseFloat(bv);
    if (!isNaN(an) && !isNaN(bn)) return (an - bn) * dir;
    return av.localeCompare(bv) * dir;
  });
  rows.forEach(r => table.appendChild(r));
}

function renderRunsTable() {
  const table = document.getElementById('runsTable');
  table.innerHTML = '';
  const cols = [
    { key: 'run_id', label: 'Run' },
    { key: 'kind', label: 'Kind' },
    { key: 'name', label: 'Model' },
    { key: 'quantization', label: 'Quant' },
    { key: 'lab', label: 'Lab' },
    { key: 'variant', label: 'Variant' },
    { key: 'agent', label: 'Agent' },
    { key: 'n_tasks', label: 'Tasks' },
    { key: 'n_resolved', label: 'Resolved' },
    { key: 'pass_rate', label: 'Pass rate' },
    { key: 'mean_partial', label: 'Mean partial' },
    { key: 'duration_sec', label: 'Duration' },
    { key: 'started_at', label: 'Started' },
  ];
  tableHeader(table, cols);
  const showPlumbing = document.getElementById('showPlumbing').checked;
  runs.filter(r => showPlumbing || isEvalRun(r)).forEach(r => {
    const m = r.model || {};
    const vals = {
      run_id: r.run_id,
      kind: r.kind || 'eval',
      name: m.name, quantization: m.quantization, lab: m.lab, variant: m.variant,
      agent: r.endpoint ? r.endpoint.agent : '',
      n_tasks: r.summary.n_tasks,
      n_resolved: r.summary.n_resolved,
      pass_rate: r.summary.pass_rate,
      mean_partial: r.summary.mean_partial,
      duration_sec: r.duration_sec,
      started_at: r.started_at
        ? new Date(r.started_at).toISOString().replace('T', ' ').slice(0, 19) + ' UTC'
        : '',
    };
    const tr = el('tr');
    cols.forEach(c => {
      const td = el('td');
      tr.dataset['v_' + c.key] = vals[c.key] ?? '';
      if (c.key === 'pass_rate' || c.key === 'mean_partial') {
        td.textContent = pct(vals[c.key] || 0);
        td.className = 'num ' + ((vals[c.key] || 0) >= 0.5 ? 'ok' : 'bad');
      } else if (c.key === 'duration_sec') {
        td.textContent = (vals[c.key] || 0).toFixed(0) + ' s';
        td.className = 'num';
      } else if (['n_tasks', 'n_resolved'].includes(c.key)) {
        td.textContent = vals[c.key];
        td.className = 'num';
      } else {
        td.textContent = vals[c.key] ?? '';
      }
      tr.appendChild(td);
    });
    table.appendChild(tr);
  });
}

function fillRunOptions(id, def) {
  const sel = document.getElementById(id);
  sel.innerHTML = '';
  runs.forEach((r, i) => { const o = el('option', r.run_id); o.value = i; sel.appendChild(o); });
  sel.value = def;
}

function renderRunSelectors() {
  fillRunOptions('runSelect', 0);
  fillRunOptions('cmpA', 0);
  fillRunOptions('cmpB', Math.min(1, runs.length - 1));
}

function currentRun() { return runs[document.getElementById('runSelect').value | 0]; }

function renderLangSelect() {
  const run = currentRun();
  const sel = document.getElementById('langSelect');
  sel.innerHTML = '';
  const o = el('option', 'all'); o.value = 'all'; sel.appendChild(o);
  Object.keys(run.summary.by_language || {}).sort().forEach(l => {
    const op = el('option', l); op.value = l; sel.appendChild(op);
  });
}

function renderDetail() {
  const run = currentRun();
  const lang = document.getElementById('langSelect').value;
  const q = document.getElementById('search').value.toLowerCase();
  const summary = document.getElementById('detailSummary');
  summary.textContent = `${run.run_id} · ${run.summary.n_resolved}/${run.summary.n_tasks} `
    + `resolved (${pct(run.summary.pass_rate)}) · mean partial `
    + `${run.summary.mean_partial.toFixed(3)} · ${(run.duration_sec || 0).toFixed(0)} s`;

  const table = document.getElementById('tasksTable');
  table.innerHTML = '';
  const cols = [
    { key: 'task_id', label: 'Task' }, { key: 'language', label: 'Language' },
    { key: 'reward', label: 'Reward' }, { key: 'f2p', label: 'f2p' },
    { key: 'f2pPct', label: 'f2p %' }, { key: 'p2p', label: 'p2p' },
    { key: 'partial', label: 'Partial' },
    { key: 'status', label: 'Status' }, { key: 'duration_sec', label: 'Duration' },
  ];
  tableHeader(table, cols);
  run.tasks
    .filter(t => lang === 'all' || t.language === lang)
    .filter(t => !q || t.task_id.toLowerCase().includes(q))
    .forEach(t => {
      const f2pPct = t.f2p_total ? t.f2p_passed / t.f2p_total : null;
      const vals = {
        task_id: t.task_id, language: t.language,
        reward: t.reward, partial: t.partial,
        f2p: t.f2p_total ? `${t.f2p_passed}/${t.f2p_total}` : '',
        f2pPct: f2pPct,
        p2p: t.p2p_total ? `${t.p2p_passed}/${t.p2p_total}` : '',
        status: t.status, duration_sec: t.duration_sec,
      };
      const tr = el('tr');
      cols.forEach(c => {
        const td = el('td');
        tr.dataset['v_' + c.key] = vals[c.key] ?? '';
        if (c.key === 'reward') {
          const near = t.reward !== 1 && f2pPct !== null && f2pPct >= 0.9;
          td.textContent = (vals[c.key] === 1 ? 'pass' : (vals[c.key] === 0 ? 'fail' : 'n/a'))
            + (near ? ' (near)' : '');
          td.className = vals[c.key] === 1 ? 'ok' : (vals[c.key] === 0 ? 'bad' : 'muted');
        } else if (c.key === 'f2pPct') {
          td.textContent = f2pPct === null ? '' : pct(f2pPct);
          td.className = 'num ' + (f2pPct === null ? '' : (f2pPct >= 0.9 ? 'ok' : 'bad'));
        } else if (c.key === 'partial') {
          td.textContent = vals[c.key] === null || vals[c.key] === undefined
            ? '' : vals[c.key].toFixed(3);
          td.className = 'num';
        } else if (c.key === 'duration_sec') {
          td.textContent = vals[c.key] == null ? '' : vals[c.key].toFixed(0) + ' s';
          td.className = 'num';
        } else {
          td.textContent = vals[c.key] ?? '';
        }
        tr.appendChild(td);
      });
      table.appendChild(tr);
    });
}

function renderCompare() {
  const a = runs[document.getElementById('cmpA').value | 0];
  const b = runs[document.getElementById('cmpB').value | 0];
  const bById = Object.fromEntries(b.tasks.map(t => [t.task_id, t]));
  const shared = a.tasks.filter(t => bById[t.task_id]);
  const table = document.getElementById('cmpTable');
  table.innerHTML = '';
  const cols = [
    { key: 'task_id', label: 'Task' }, { key: 'language', label: 'Language' },
    { key: 'a', label: a.run_id.slice(0, 24) }, { key: 'b', label: b.run_id.slice(0, 24) },
  ];
  tableHeader(table, cols);
  let aWin = 0, bWin = 0;
  shared.forEach(t => {
    const o = bById[t.task_id];
    if ((t.reward || 0) > (o.reward || 0)) aWin++;
    else if ((o.reward || 0) > (t.reward || 0)) bWin++;
    const tr = el('tr');
    const vals = { task_id: t.task_id, language: t.language, a: t.reward, b: o.reward };
    cols.forEach(c => {
      const td = el('td');
      tr.dataset['v_' + c.key] = vals[c.key] ?? '';
      if (c.key === 'a' || c.key === 'b') {
        td.textContent = vals[c.key] === 1 ? 'pass' : (vals[c.key] === 0 ? 'fail' : 'n/a');
        td.className = vals[c.key] === 1 ? 'ok' : (vals[c.key] === 0 ? 'bad' : 'muted');
      } else td.textContent = vals[c.key] ?? '';
      tr.appendChild(td);
    });
    table.appendChild(tr);
  });
  document.getElementById('cmpSummary').textContent =
    `${shared.length} shared task(s) · A wins ${aWin} · B wins ${bWin} · ties ${shared.length - aWin - bWin}`;
}

document.getElementById('runSelect').onchange = () => { renderLangSelect(); renderDetail(); };
document.getElementById('langSelect').onchange = renderDetail;
document.getElementById('search').oninput = renderDetail;
document.getElementById('cmpA').onchange = renderCompare;
document.getElementById('cmpB').onchange = renderCompare;
document.getElementById('showPlumbing').onchange = renderRunsTable;

function isModelRun(r) {
  return ((r.endpoint || {}).agent || '') !== 'oracle';
}

// Plumbing checks (oracle, or a smoke run with a tiny timeout) must not pollute
// the quality/cost aggregates.
function isEvalRun(r) {
  return isModelRun(r) && !String(r.kind || 'eval').startsWith('smoke');
}

function wilson(k, n) {
  if (!n) return null;
  const z = 1.96, p = k / n, d = 1 + z * z / n;
  const centre = (p + z * z / (2 * n)) / d;
  const half = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d;
  return [Math.max(0, centre - half), Math.min(1, centre + half)];
}

function coreMetrics(tasks) {
  const n = tasks.length;
  const resolved = tasks.filter(t => t.reward === 1).length;
  const fracs = tasks.filter(t => t.f2p_total).map(t => t.f2p_passed / t.f2p_total);
  const meanF2p = fracs.length ? fracs.reduce((a, b) => a + b, 0) / fracs.length : null;
  const nearMiss = tasks.filter(
    t => t.reward !== 1 && t.f2p_total && (t.f2p_passed / t.f2p_total) >= 0.9
  ).length;
  const p2pRegressions = tasks.filter(
    t => t.p2p_total && t.p2p_passed !== t.p2p_total
  ).length;
  const sum = (key) => tasks.filter(t => t[key]).reduce((a, t) => a + t[key], 0);
  const agentHours = sum('agent_execution_sec') ? sum('agent_execution_sec') / 3600 : null;
  const outTok = sum('output_tokens'), reaTok = sum('reasoning_tokens'), steps = sum('steps');
  const ci = wilson(resolved, n);
  const per = (v, d) => (v !== null && d ? v / d : null);
  return {
    n: n, resolved: resolved, passRate: n ? resolved / n : 0,
    ciText: ci ? `${pct(ci[0])}\u2013${pct(ci[1])}` : '',
    meanF2p: meanF2p, nearMiss: nearMiss, p2pRegressions: p2pRegressions,
    agentHours: agentHours,
    agentHoursPerTask: per(agentHours, n),
    agentHoursPerResolved: per(agentHours, resolved),
    outputPerTask: per(outTok || null, n),
    outputPerResolved: per(outTok || null, resolved),
    reasoningRatio: (reaTok && outTok) ? reaTok / outTok : null,
    stepsPerTask: per(steps || null, n),
    secondsPerStep: (sum('agent_execution_sec') && steps) ? sum('agent_execution_sec') / steps : null,
  };
}

// All runs of the same model id are aggregated into one row, so a model that was
// evaluated across many nightly batches (or under updated metadata) shows up once.
function aggregateByModel(evalRuns) {
  const groups = new Map();
  evalRuns.forEach(r => {
    const id = (r.model || {}).id || r.run_id;
    if (!groups.has(id)) groups.set(id, { id: id, runs: [], tasks: [] });
    const g = groups.get(id);
    g.runs.push(r);
    g.tasks.push(...(r.tasks || []));
  });
  const pickLatest = (byTime, getter) => {
    for (let i = byTime.length - 1; i >= 0; i--) {
      const v = getter(byTime[i]);
      if (v !== undefined && v !== null && v !== '' && v !== 'unspecified') return v;
    }
    return 'unspecified';
  };
  return [...groups.values()].map(g => {
    const byTime = [...g.runs].sort(
      (a, b) => String(a.started_at || '').localeCompare(String(b.started_at || ''))
    );
    // A task evaluated more than once counts once, with the most recent result.
    const taskMap = new Map();
    byTime.forEach(r => (r.tasks || []).forEach(t => taskMap.set(t.task_id, t)));
    const tasks = [...taskMap.values()];
    const serving = {};
    ['context_length', 'kv_cache_dtype', 'tensor_parallel', 'gpus', 'reasoning_effort', 'stack'].forEach(k => {
      serving[k] = pickLatest(byTime, r => (r.serving || {})[k]);
    });
    return {
      id: g.id,
      name: pickLatest(byTime, r => (r.model || {}).name),
      quantization: pickLatest(byTime, r => (r.model || {}).quantization),
      lab: pickLatest(byTime, r => (r.model || {}).lab),
      variant: pickLatest(byTime, r => (r.model || {}).variant),
      serving: serving,
      nRuns: g.runs.length,
      tasks: tasks,
      m: coreMetrics(tasks),
    };
  });
}

function renderModels() {
  const table = document.getElementById('modelsTable');
  const groups = aggregateByModel(runs.filter(isEvalRun));
  const cols = [
    { key: 'name', label: 'Model' }, { key: 'lab', label: 'Lab' },
    { key: 'variant', label: 'Variant' }, { key: 'quantization', label: 'Quant' },
    { key: 'context_length', label: 'Context' }, { key: 'kv_cache_dtype', label: 'KV' },
    { key: 'tensor_parallel', label: 'TP' }, { key: 'gpus', label: 'GPUs' },
    { key: 'reasoning_effort', label: 'Reasoning' },
    { key: 'nRuns', label: 'Runs' }, { key: 'n', label: 'Tasks' },
    { key: 'passRate', label: 'Pass rate' }, { key: 'ciText', label: '95% CI' },
    { key: 'resolved', label: 'Resolved' }, { key: 'meanF2p', label: 'Mean f2p' },
    { key: 'nearMiss', label: 'Near-miss' }, { key: 'p2pRegressions', label: 'p2p regr.' },
    { key: 'agentHours', label: 'Agent h' }, { key: 'agentHoursPerTask', label: 'Agent h/task' },
    { key: 'agentHoursPerResolved', label: 'Agent h/resolved' },
    { key: 'outputPerTask', label: 'Out tok/task' },
    { key: 'reasoningRatio', label: 'Reasoning %' },
    { key: 'secondsPerStep', label: 's/step' }, { key: 'stepsPerTask', label: 'Steps/task' },
  ];
  tableHeader(table, cols);
  groups.forEach(g => {
    const m = g.m, sv = g.serving;
    const vals = {
      name: g.name, lab: g.lab, variant: g.variant, quantization: g.quantization,
      context_length: sv.context_length, kv_cache_dtype: sv.kv_cache_dtype,
      tensor_parallel: sv.tensor_parallel, gpus: sv.gpus,
      reasoning_effort: sv.reasoning_effort,
      nRuns: g.nRuns, n: m.n, passRate: m.passRate, ciText: m.ciText, resolved: m.resolved,
      meanF2p: m.meanF2p, nearMiss: m.nearMiss, p2pRegressions: m.p2pRegressions,
      agentHours: m.agentHours, agentHoursPerTask: m.agentHoursPerTask,
      agentHoursPerResolved: m.agentHoursPerResolved, outputPerTask: m.outputPerTask,
      reasoningRatio: m.reasoningRatio, secondsPerStep: m.secondsPerStep,
      stepsPerTask: m.stepsPerTask,
    };
    const tr = el('tr');
    cols.forEach(c => {
      const td = el('td');
      tr.dataset['v_' + c.key] = vals[c.key] ?? '';
      const v = vals[c.key];
      if (c.key === 'passRate') {
        td.textContent = pct(v || 0);
        td.className = 'num ' + ((v || 0) >= 0.5 ? 'ok' : 'bad');
      } else if (c.key === 'meanF2p' || c.key === 'reasoningRatio') {
        td.textContent = v == null ? '' : pct(v); td.className = 'num';
      } else if (['agentHours', 'agentHoursPerTask', 'agentHoursPerResolved'].includes(c.key)) {
        td.textContent = v == null ? '' : Number(v).toFixed(2); td.className = 'num';
      } else if (['outputPerTask', 'stepsPerTask'].includes(c.key)) {
        td.textContent = v == null ? '' : Math.round(v).toLocaleString(); td.className = 'num';
      } else if (c.key === 'secondsPerStep') {
        td.textContent = v == null ? '' : Number(v).toFixed(1); td.className = 'num';
      } else if (['n', 'resolved', 'nRuns'].includes(c.key)) {
        td.textContent = v; td.className = 'num';
      } else if (c.key === 'nearMiss' || c.key === 'p2pRegressions') {
        td.textContent = v; td.className = 'num ' + (v ? 'bad' : '');
      } else {
        td.textContent = v ?? '';
      }
      tr.appendChild(td);
    });
    table.appendChild(tr);
  });
  const totalTasks = groups.reduce((a, g) => a + g.m.n, 0);
  const totalRuns = groups.reduce((a, g) => a + g.nRuns, 0);
  document.getElementById('modelsNote').textContent =
    `${groups.length} model(s), ${totalTasks} evaluated task(s) across ${totalRuns} run(s), ` +
    `aggregated by model id. Oracle and smoke runs are excluded. ` +
    `"Agent h/resolved" and "Out tok/resolved" stay blank when a model resolved nothing.`;
}

function svgEl(name, attrs) {
  const e = document.createElementNS('http://www.w3.org/2000/svg', name);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  return e;
}

function renderPareto() {
  const host = document.getElementById('pareto');
  host.innerHTML = '';
  const pts = aggregateByModel(runs.filter(isEvalRun))
    .map(g => ({ g: g, m: g.m }))
    .filter(p => p.m.agentHoursPerTask !== null && p.m.agentHoursPerTask > 0);
  if (!pts.length) { host.textContent = 'No agent timing data to plot yet.'; return; }

  const W = 880, H = 440, pad = { l: 64, r: 36, t: 24, b: 64 };
  const xmax = Math.max(...pts.map(p => p.m.agentHoursPerTask)) * 1.2;
  const ymax = Math.max(0.05, Math.max(...pts.map(p => p.m.passRate)) * 1.25);
  const sx = x => pad.l + (x / xmax) * (W - pad.l - pad.r);
  const sy = y => H - pad.b - (y / ymax) * (H - pad.t - pad.b);

  const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, width: '100%', role: 'img' });
  svg.style.maxWidth = W + 'px';
  svg.appendChild(svgEl('line', { x1: pad.l, y1: H - pad.b, x2: W - pad.r, y2: H - pad.b, stroke: '#2b3442' }));
  svg.appendChild(svgEl('line', { x1: pad.l, y1: pad.t, x2: pad.l, y2: H - pad.b, stroke: '#2b3442' }));

  for (let i = 0; i <= 4; i++) {
    const v = ymax * i / 4, y = sy(v);
    svg.appendChild(svgEl('line', { x1: pad.l, y1: y, x2: W - pad.r, y2: y, stroke: '#202836' }));
    const t = svgEl('text', { x: pad.l - 8, y: y + 4, fill: '#8b98a5', 'font-size': 11, 'text-anchor': 'end' });
    t.textContent = pct(v); svg.appendChild(t);
  }
  for (let i = 0; i <= 4; i++) {
    const v = xmax * i / 4, x = sx(v);
    const t = svgEl('text', { x: x, y: H - pad.b + 18, fill: '#8b98a5', 'font-size': 11, 'text-anchor': 'middle' });
    t.textContent = v.toFixed(1); svg.appendChild(t);
  }
  const xl = svgEl('text', { x: (W + pad.l - pad.r) / 2, y: H - 16, fill: '#8b98a5', 'font-size': 12, 'text-anchor': 'middle' });
  xl.textContent = 'agent-hours per task (cheaper to the left)'; svg.appendChild(xl);
  const yl = svgEl('text', { x: 16, y: (H - pad.t) / 2, fill: '#8b98a5', 'font-size': 12, 'text-anchor': 'middle', transform: `rotate(-90 16 ${(H - pad.t) / 2})` });
  yl.textContent = 'pass rate (better upwards)'; svg.appendChild(yl);

  const sorted = [...pts].sort((a, b) => a.m.agentHoursPerTask - b.m.agentHoursPerTask);
  let best = -1; const frontier = [];
  sorted.forEach(p => {
    if (p.m.passRate > best) { best = p.m.passRate; frontier.push(p); }
  });
  if (frontier.length > 1) {
    const d = frontier.map((p, i) => (i ? 'L' : 'M') + sx(p.m.agentHoursPerTask) + ',' + sy(p.m.passRate)).join(' ');
    svg.appendChild(svgEl('path', { d: d, fill: 'none', stroke: '#4c9aff', 'stroke-width': 1.5, 'stroke-dasharray': '4 3' }));
  }

  const maxOut = Math.max(...pts.map(p => p.m.outputPerTask || 0), 1);
  const onFrontier = new Set(frontier.map(p => p.g.id));
  pts.forEach(p => {
    const isFront = onFrontier.has(p.g.id);
    const radius = 6 + 10 * ((p.m.outputPerTask || 0) / maxOut);
    const cx = sx(p.m.agentHoursPerTask), cy = sy(p.m.passRate);
    const circle = svgEl('circle', {
      cx: cx, cy: cy, r: radius, fill: isFront ? '#3fb950' : '#4c9aff',
      opacity: isFront ? 0.85 : 0.55, stroke: '#0f1419', 'stroke-width': 1,
    });
    const title = svgEl('title', {});
    title.textContent = [
      p.g.name,
      `${p.g.lab} | ${p.g.quantization}`,
      `${p.m.n} task(s) from ${p.g.nRuns} run(s), pass rate ${pct(p.m.passRate)}`,
      `${p.m.agentHoursPerTask.toFixed(2)} agent-h/task`,
      `${Math.round(p.m.outputPerTask || 0).toLocaleString()} out tok/task`,
      `reasoning ${p.m.reasoningRatio == null ? 'n/a' : pct(p.m.reasoningRatio)}`,
    ].join('\\n');
    circle.appendChild(title); svg.appendChild(circle);
    const label = svgEl('text', { x: cx, y: cy - radius - 4, fill: '#e6edf3', 'font-size': 10, 'text-anchor': 'middle' });
    label.textContent = (p.g.name || p.g.id).slice(0, 20) + (p.g.quantization && p.g.quantization !== 'unspecified' ? ` ${p.g.quantization}` : '');
    svg.appendChild(label);
  });
  host.appendChild(svg);
}

renderCards();
renderModels();
renderRunsTable();
renderPareto();
if (runs.length) { renderRunSelectors(); renderLangSelect(); renderDetail(); renderCompare(); }
</script>
</body>
</html>
"""


def cmd_report(args: argparse.Namespace) -> int:
    if not INDEX_PATH.exists():
        raise SystemExit(
            "error: no results index. Run an evaluation first "
            "(python eval/deepswe_eval.py run --model <id>)."
        )
    index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    runs = []
    for entry in index:
        path = RESULTS_DIR / entry["result_file"]
        if path.exists():
            runs.append(json.loads(path.read_text(encoding="utf-8")))
    if not runs:
        raise SystemExit("error: index references no readable run files")

    title = args.title
    data = json.dumps({"runs": runs})
    html = (
        HTML_TEMPLATE.replace("__TITLE__", title)
        .replace("__GENERATED__", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
        .replace("__NRUNS__", str(len(runs)))
        .replace("__DATA__", data.replace("</", "<\\/"))
    )
    out = Path(args.output) if args.output else REPORT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out.relative_to(REPO_ROOT)} ({len(runs)} run(s))")
    return 0


# --------------------------------------------------------------------------- #
# clean
# --------------------------------------------------------------------------- #

PIER_IMAGE_RE = re.compile(r"__[A-Za-z0-9]{5,10}-(main|pier-egress-proxy)$")


def pier_image_refs() -> list[str]:
    """Docker image refs created by pier (task base images, agent and proxy builds).

    Deliberately narrow: other images on the host (model servers, databases,
    caches) must never be matched.
    """
    proc = subprocess.run(
        ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
        capture_output=True, text=True, check=False,
    )
    refs = []
    for line in proc.stdout.splitlines():
        ref = line.strip()
        if not ref or ref.startswith("<none>"):
            continue
        repo = ref.rsplit(":", 1)[0]
        if "swe-bench-202605" in repo or PIER_IMAGE_RE.search(repo):
            refs.append(ref)
    return refs


def cleanup_pier_images() -> int:
    refs = pier_image_refs()
    for ref in refs:
        subprocess.run(["docker", "rmi", "-f", ref], capture_output=True, check=False)
    # Dangling layers only; never -a, which would delete unrelated cached images.
    subprocess.run(["docker", "image", "prune", "-f"], capture_output=True, check=False)
    return len(refs)


def disk_free_gb(path: Path = REPO_ROOT) -> float | None:
    try:
        usage = shutil.disk_usage(path)
    except OSError:
        return None
    return usage.free / (1024 ** 3)


def cmd_clean(args: argparse.Namespace) -> int:
    if args.dataset:
        if DATASET_DIR.exists():
            shutil.rmtree(DATASET_DIR)
            print(f"Removed {DATASET_DIR.relative_to(REPO_ROOT)}")
        else:
            print("Dataset already absent.")
    if args.images:
        refs = pier_image_refs()
        if not refs:
            print("No pier images found.")
            return 0
        if not args.yes:
            print(f"Would remove {len(refs)} pier image(s). Re-run with --yes.")
            return 0
        removed = cleanup_pier_images()
        print(f"Removed {removed} pier image(s); {disk_free_gb():.0f} GB free.")
    if not args.dataset and not args.images:
        print("Nothing to do. Use --dataset and/or --images.")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def add_run_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", help="model id from eval/models.yaml")
    parser.add_argument("--languages", help="comma-separated language filter")
    parser.add_argument("--tasks", help="comma-separated explicit task ids")
    parser.add_argument("--n-tasks", type=int, help="max number of tasks")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--match-run", help="reuse the exact task set of a run id")
    parser.add_argument("--run-id", help="override the generated run id")
    parser.add_argument("--jobs-dir", default=str(JOBS_DIR))
    parser.add_argument("--n-concurrent", type=int, default=1)
    parser.add_argument("--timeout-multiplier", type=float, default=None)
    parser.add_argument("--agent-timeout-multiplier", type=float, default=None)
    parser.add_argument("--verifier-timeout-multiplier", type=float, default=None)
    parser.add_argument("--agent", help="override the agent (e.g. oracle)")
    parser.add_argument("--no-forwarder", action="store_true",
                        help="do not auto-start the model forwarder")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--kind", default="eval", help=argparse.SUPPRESS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="build the filtered dataset")
    p_build.add_argument("--languages", help="override enabled languages")
    p_build.add_argument("--no-strict", action="store_true",
                         help="do not fail on expected_counts mismatch")
    p_build.set_defaults(func=cmd_build)

    p_list = sub.add_parser("list", help="list tasks and models")
    p_list.set_defaults(func=cmd_list)

    p_run = sub.add_parser("run", help="run an evaluation")
    add_run_flags(p_run)
    p_run.set_defaults(func=cmd_run)

    p_smoke = sub.add_parser("smoke", help="fast plumbing check")
    p_smoke.add_argument("--model", help="also run a model smoke with this model id")
    p_smoke.add_argument("--task", help="explicit smoke task id(s)")
    p_smoke.add_argument("--per-language", action="store_true",
                         help="one smoke task per enabled language")
    p_smoke.add_argument("--jobs-dir", default=str(JOBS_DIR))
    p_smoke.add_argument("--n-concurrent", type=int, default=1)
    p_smoke.add_argument("--timeout-multiplier", type=float, default=0.15)
    p_smoke.add_argument("--agent-timeout-multiplier", type=float, default=0.02)
    p_smoke.add_argument("--verifier-timeout-multiplier", type=float, default=0.5)
    p_smoke.add_argument("--dry-run", action="store_true")
    p_smoke.set_defaults(func=cmd_smoke)

    p_report = sub.add_parser("report", help="generate the HTML report")
    p_report.add_argument("--output", help="output path (default results/report.html)")
    p_report.add_argument("--title", default="DeepSWE local evaluation report")
    p_report.set_defaults(func=cmd_report)

    p_collect = sub.add_parser("collect", help="re-parse an existing pier job")
    p_collect.add_argument("--job", required=True, help="job name under jobs/")
    p_collect.add_argument("--model", help="model id from eval/models.yaml")
    p_collect.add_argument("--name", help="model name if --model is unknown")
    p_collect.add_argument("--quantization", default="unspecified")
    p_collect.add_argument("--lab", default="unspecified")
    p_collect.add_argument("--variant", default="unspecified")
    p_collect.add_argument("--run-id", help="override the stored run id")
    p_collect.add_argument("--jobs-dir", default=str(JOBS_DIR))
    p_collect.add_argument("--kind", default="collected")
    p_collect.set_defaults(func=cmd_collect)

    p_clean = sub.add_parser("clean", help="remove generated dataset/images")
    p_clean.add_argument("--dataset", action="store_true")
    p_clean.add_argument("--images", action="store_true")
    p_clean.add_argument("--yes", action="store_true")
    p_clean.set_defaults(func=cmd_clean)

    p_fwd = sub.add_parser("forwarder", help="manage the model TCP forwarder")
    p_fwd.add_argument("action", choices=["start", "stop", "status"])
    p_fwd.add_argument("--listen", default=FORWARDER_LISTEN)
    p_fwd.add_argument("--target", default=FORWARDER_TARGET)
    p_fwd.set_defaults(func=cmd_forwarder)

    args = parser.parse_args(argv)
    if args.command == "run" and not args.model and args.agent != "oracle":
        parser.error("run requires --model (or --agent oracle)")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
