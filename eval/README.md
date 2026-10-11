# Local DeepSWE evaluation harness

Run the DeepSWE benchmark against models you serve locally, store the results
with model metadata, and generate an HTML report to compare runs.

The harness is `eval/deepswe_eval.py` (stdlib + PyYAML), with configuration in
`eval/languages.yaml` and `eval/models.yaml` and a tiny TCP forwarder in
`eval/llm_forwarder.py`.

## How model traffic reaches the container

DeepSWE tasks run with `network_mode = "no-network"`, so task containers have no
direct internet access. pier injects an egress proxy (Squid) that only allows the
hosts named in the agent's `OPENAI_BASE_URL`, **on ports 80 and 443 only**, and
the task's `environment/` directory must not contain a `docker-compose.yaml`
(pier refuses build-time agent installs for compose tasks).

That rules out pointing the agent straight at a local model on an arbitrary port.
The harness solves it with a small TCP forwarder:

```
task container ──(HTTP_PROXY)──> pier egress proxy (Squid)
   └─ allows 172.17.0.1.nip.io:80
        └─ 172.17.0.1 = docker bridge gateway = host
             └─ deepswe-llm-forward container (host network)
                  └─ 127.0.0.1:4000 = LiteLLM ──> your vLLM/llama.cpp/...
```

`172.17.0.1.nip.io` is a public wildcard DNS name that resolves to `172.17.0.1`,
so no `extra_hosts` or DNS changes are needed. The forwarder is started
automatically by `run` and `smoke`; manage it with:

```bash
python3 eval/deepswe_eval.py forwarder start                       # default: 127.0.0.1:4000 (LiteLLM)
python3 eval/deepswe_eval.py forwarder start --target 127.0.0.1:8005
python3 eval/deepswe_eval.py forwarder status
python3 eval/deepswe_eval.py forwarder stop
```

The forwarder binds `172.17.0.1:80` (the docker gateway, not the LAN) and runs on
the host network so it can reach a loopback-only model server such as LiteLLM.

## Prerequisites

- Docker.
- [`pier`](https://pypi.org/project/datacurve-pier/): `uv tool install datacurve-pier`.
- PyYAML for the system Python (`python3 -c "import yaml"`).
- A model server. Either LiteLLM on `127.0.0.1:4000` (default forwarder target)
  or a server on another loopback port with a matching `--target`.
- Port 80 free on `172.17.0.1` (the harness uses it for the forwarder).

## Quick start

```bash
uv tool install datacurve-pier

# 1. Build the language-filtered dataset (113 tasks: rust, typescript, javascript, python, go)
python3 eval/deepswe_eval.py build

# 2. Verify the pipeline end to end (oracle, then a real model turn)
python3 eval/deepswe_eval.py smoke --model mimo-v2.6-flash

# 3. Run a small evaluation
python3 eval/deepswe_eval.py run --model mimo-v2.6-flash \
    --languages python --n-tasks 8 --sample-seed 0

# 4. Generate the HTML report
python3 eval/deepswe_eval.py report        # -> results/report.html
```

## What `build` does

`build` classifies every directory under `tasks/` by the test runner it actually
invokes (`tests/test.sh`), because `task.toml`'s `language` field is wrong for a
few tasks (for example `httpx-deterministic-cookie-store` is Python but declares
`typescript`). It copies the tasks whose language is enabled into
`datasets/deep-swe-local/tasks/` unchanged. `datasets/deep-swe-local/manifest.json`
records the task → language mapping and is the source of truth for the runner and
the report.

Detected counts with the default config:

| Language   | Tasks | In dataset |
| ---------- | ----- | ---------- |
| Rust       | 5     | yes        |
| TypeScript | 34    | yes        |
| JavaScript | 5     | yes        |
| Python     | 34    | yes        |
| Go         | 35    | yes        |

All 113 tasks are included. JavaScript is kept alongside TypeScript; the corpus
labels them separately and the report shows them as distinct languages.

### Choosing or extending languages

Edit `eval/languages.yaml`:

- `enabled_languages` — which languages `build` copies. The corpus has no Swift
  or Kotlin tasks yet; add tasks under `tasks/` and put `swift`/`kotlin` in
  `enabled_languages` (and make sure their runner is in the family's `detect`
  list) to include them.
- `families` — detection rules, evaluated in order. Order matters: `node` is
  checked before `python` because TypeScript tasks also require `python3`.
- `overrides` — pin a task's language explicitly.
- `expected_counts` — `build` fails if the corpus no longer matches; pass
  `--no-strict` to build anyway.

## Model registry

Add an entry to `eval/models.yaml`:

```yaml
models:
  my-model:
    name: my-model              # model name
    quantization: bf16          # quantization (selection axis)
    lab: Example Lab            # lab / organization
    variant: instruct           # variant
    serving:                    # deployment config, recorded with each run
      context_length: 32768
      kv_cache_dtype: fp8
      tensor_parallel: 2
      gpus: "2x RTX 5090"
      reasoning_effort: low     # or medium/high/none; also read from agent kwargs
      stack: vLLM
      notes: ""
    agent:
      name: mini-swe-agent
      model_name: openai/my-litellm-alias
      env:
        OPENAI_BASE_URL: http://172.17.0.1.nip.io/v1
        OPENAI_API_KEY: ${LITELLM_MASTER_KEY}
        MSWEA_API_KEY: ${LITELLM_MASTER_KEY}
      kwargs:
        model_class: litellm    # force chat completions, not the Responses API
```

The four identity fields (`name`, `quantization`, `lab`, `variant`) are stored
with every run and shown in the report. The `serving` block records the
deployment configuration that also drives quality and speed; leave unknown
values as `unspecified`. `env` values support `${VAR}` and `${VAR:-default}`
expansion from your shell environment.

## Running

```bash
# whole enabled dataset
python3 eval/deepswe_eval.py run --model my-model

# by language, with a deterministic subset
python3 eval/deepswe_eval.py run --model my-model --languages rust,go \
    --n-tasks 10 --sample-seed 0

# explicit tasks
python3 eval/deepswe_eval.py run --model my-model --tasks ts-pattern-match-each

# reuse the exact task set of a previous run (fair comparison across models)
python3 eval/deepswe_eval.py run --model other-model --match-run 20261005T143000Z_my-model

# inspect the generated pier JobConfig without running
python3 eval/deepswe_eval.py run --model my-model --languages python --n-tasks 2 --dry-run
```

Each run writes a reproducible pier config to `results/runs/<run-id>/job.yaml`,
invokes `pier run --config`, and stores a normalized record at
`results/runs/<run-id>.json` plus a summary entry in `results/index.json`. The
record contains the model identity fields, the endpoint, the task list, per-task
rewards and timings, the aggregated summary, and the total wall-clock duration.

Runs are **sequential by default** (`--n-concurrent 1`): one task at a time, so
each task gets the model to itself and the host is not oversubscribed. Raise
`--n-concurrent` only if you deliberately want to batch trials. Because a full
113-task run is long, prefer running it in batches (for example per language or
with `--n-tasks`/`--tasks`) and clean images between batches.

To re-parse an existing pier job (for example one you ran by hand):

```bash
python3 eval/deepswe_eval.py collect --job <job-name> --model my-model
```

## Overnight batch driver

`eval/nightly.py` walks the whole corpus in a fixed, reproducible, language-balanced
order (seeded), inside a nightly window, sequentially, cleaning pier's docker images
between batches and persisting progress so it resumes on later nights.

```bash
setsid nohup python3 eval/nightly.py --model mimo-v2.6-flash \
    --start-now --start-at 22:00 --end-at 08:00 --budget-hours 24 \
    --task-timeout-min 60 >> results/nightly.log 2>&1 < /dev/null &
```

- `--start-now` applies only to the first window; later windows wait for `--start-at`.
- `--budget-hours` caps each window; combined with `--end-at` it stops at a fixed hour.
- The batch size is derived from the remaining window and the per-task cap.
- Only tasks that actually produced trials are marked done; an empty batch is
  discarded and retried, and a task is given up after `--max-attempts` failures.
- Before each batch it does a real 1-token generation through the same base URL the
  agent uses, and accepts only a well-formed chat completion. A plain `/v1/models`
  probe is not enough: with a different model profile running (models are mutually
  exclusive on one port), the gateway still answers while the alias points at a
  backend that is not serving.
- State: `results/nightly-state.json`; log: `results/nightly.log`.
- When the corpus completes it regenerates the report and exits on its own.
- For a bounded check rather than a full night, combine `--once --max-tasks N` with a
  short `--budget-hours` (for example `--budget-hours 0.5 --task-timeout-min 24
  --max-tasks 1`): the window opens immediately with `--start-now` and the driver
  exits after the task, so it never rolls into the next night.

## Evaluating a second model

The task order is deterministic, so a second model walks the *same* sequence and the
two are comparable task by task. Give each model its own state file, and repoint the
forwarder at the new server first:

```bash
# forwarder start keeps the existing target when the container is already running
python3 eval/deepswe_eval.py forwarder stop
python3 eval/deepswe_eval.py forwarder start --target 127.0.0.1:8005

setsid nohup python3 eval/nightly.py --model glm-5.3-flash \
    --state results/nightly-state-glm.json \
    --start-at 22:00 --end-at 08:00 --budget-hours 10 \
    >> results/nightly-glm.log 2>&1 < /dev/null &
```

Two things to watch when swapping models:

- **Mutual exclusion.** Locally served profiles that use both GPUs and the same port
  cannot run at once. Switch the profile before the window opens, or the driver will
  simply wait (it detects the dead alias rather than burning the night).
- **The forwarder target.** `forwarder start --target 127.0.0.1:<port>` decides which
  server port 80 reaches. If you point it at a model server directly, address the
  model by its served name (for example `openai/glm-5.3-flash-local`); if you keep it
  on LiteLLM, add an alias there.

Write-ups per model live in `eval/reports/`.

## Report

`python3 eval/deepswe_eval.py report` writes a self-contained `results/report.html`
(no CDN, data embedded as JSON) with:

- a **Models (aggregated)** table: all runs of the same `model.id` collapse into one
  row (runs, tasks, pass rate with 95% CI, resolved, mean f2p, near-misses, p2p
  regressions, agent hours, tokens, reasoning share). A task evaluated more than
  once counts once, using the most recent result;
- a **Pareto scatter**, one point per model: pass rate against agent-hours per
  task, with non-dominated points highlighted;
- a per-run table (a "show plumbing runs" checkbox reveals oracle/smoke runs,
  which are hidden by default and excluded from every aggregate);
- **graded metrics** next to the binary score (see below);
- per-run language breakdown and a filterable/sortable task table;
- a side-by-side comparison of two runs over their shared tasks.

Aggregation happens when the report is generated, so you can inspect partial
progress at any point instead of waiting for the corpus to finish.

### Graded vs binary scoring

`reward` is binary: 1 only when every fail-to-pass test passes and no
pass-to-pass test regresses. That is the official DeepSWE score and the
leaderboard-comparable one.

`partial` is `(f2p_passed + p2p_passed) / (f2p_total + p2p_total)`. Because
pass-to-pass tests usually outnumber fail-to-pass tests many times over, a task
with 4/6 f2p and 62/62 p2p still reads ~0.97, so **partial is not a good graded
signal**. The report therefore adds:

- **mean f2p**: the fraction of fail-to-pass tests resolved, averaged per task —
  it measures how much of the requested behaviour was implemented;
- **near-miss**: tasks with f2p ≥ 0.9 that still score 0;
- **p2p regressions**: tasks where a previously passing test broke;
- **95% CI** (Wilson): the interval around the pass rate, so small differences
  between models are not over-read when n is small.

- **f2p (fail-to-pass)**: tests that failed before the change and must pass after
  it; they encode the new behaviour the task asks for.
- **p2p (pass-to-pass)**: tests that already passed and must keep passing; the
  regression guard.

### Cost and token metrics

For every trial the harness reads the ATIF trajectory's `final_metrics` and the
trial timings, and records:

- `input_tokens`, `output_tokens`, `reasoning_tokens`, `text_tokens`,
  `cache_tokens`, `peak_context_tokens`, `steps`;
- `agent_execution_sec`, `environment_setup_sec`, `verifier_sec`;
- derived `reasoning_ratio` (reasoning / output), `output_tokens_per_sec` and
  `seconds_per_step`.

`agent_hours` is the model's own wall time. With the sequential default (one
trial at a time) it approximates GPU-hours; it excludes environment build and
verification. `reasoning_ratio` is the quickest way to spot a model whose
reasoning budget inflates wall time. Oracle runs are excluded from the
efficiency table and chart.

## Smoke test

`smoke` is the fast plumbing check:

1. **oracle** on one task — validates dataset build, the verifier, and result
   parsing. The oracle applies the reference solution, so the expected reward is 1.
2. **model** (only if `--model` is given) on one task — validates the real path
   container → egress proxy → forwarder → model server.

Useful flags: `--per-language` (one task per enabled language), `--task <id>`,
and the timeout multipliers. For a first run, pre-pull the task image so the
build timeout is not consumed by the pull:

```bash
docker pull "$(grep docker_image datasets/deep-swe-local/tasks/anko-default-function-arguments/task.toml | cut -d'"' -f2)"
```

## Lessons from running the full corpus

Everything below came out of running four models over the corpus — MiMo-V2.6-Flash
and GLM-5.3-Flash over all 113 tasks, Qwen3.8-Flash-Next and Laguna-S-2.1 over the
five Rust tasks. Per-model write-ups are in `eval/reports/`.

### Audit the verifiers before trusting a number

A verifier that cannot build the project reports every test as failed, which is
indistinguishable from a model that failed everything. In one run
`oxvg-structural-selector-preservation` came back 0/6 and 0/62 because the
verification container could not reach `static.crates.io`: both the base mode and
the new mode failed with `rc=102`, and the harness counted every test as failed
because no JUnit output was produced. The check is a `grep` over the trial output:

```bash
grep -lE "Could not resolve host" jobs/*/*/verifier/test-stdout.txt
# also suspicious: identical, non-zero rc values for the base and the new mode
```

Across 194 verifier logs from these runs, exactly one was invalid. When it happens,
remove the task from `completed` in the state file, reset its `attempts` entry, and
re-run it — the second pass is usually fine.

### Scheduling traps

- `--start-at` defaults to 22:00. A window that closes after that hour sends the
  daemon to sleep until the *next* 22:00 (22 hours idle). Pass `--start-at 00:00`
  for continuous runs.
- A batch is never interrupted at the window end, so a window can close after its
  own start boundary. `nightly.py` now opens a new window immediately when the
  current daily window still has room for a whole task (`window_remaining_min`)
  instead of waiting for the next day; before that fix, a batch crossing midnight
  parked a run for 23.7 h with four tasks left.
- `--max-tasks N` is a cumulative cap on the tasks recorded in the state file, not
  a per-window limit: with 41 tasks already done, `--max-tasks 2` closes the window
  instantly. For a bounded window pass the absolute target (`--max-tasks 43`).
- Batch size is `remaining_window_minutes // task_timeout_min`, so 24 h at the
  180 min cap asks for 8 tasks and 12 h for 3-4. Bigger batches mean more task
  images alive at once (~8 GB per task).

### Stopping and resuming

Kill the daemon's process group (launched with `setsid`, the pid is the pgid) and
remove any leftover trial containers. Nothing already completed is lost: the state
advances when a batch closes. Then salvage what the interrupted batch finished:

```bash
python3 eval/deepswe_eval.py collect --job <batch> --model <model-id>
```

Mark those tasks completed in the state file and reset `attempts` for the task that
was in flight. `attempts` is incremented *before* a batch runs, so an interrupted
task carries a failed attempt; reset it, or repeated interruptions will eventually
drop the task.

### Keep the benchmark's own per-task cap

Each `task.toml` declares 180 minutes. A shorter cap truncates long tasks and
distorts the comparison: `oxvg` scored 0/6 under a 60-minute cap and 4/6 under the
benchmark's own, taking 75 minutes. When a window has to end by 08:00, move the
window rather than the cap — hence the 22:00-05:00 nightly windows.

### Serving quirks that reach the metrics

- **Reasoning in a non-standard field.** One server returned its thinking in
  `reasoning` instead of `reasoning_content` and always reported
  `reasoning_tokens = 0`, so every reasoning metric read zero for that model even
  though the text was in the trajectory.
- **Context window exhaustion.** An agent whose prompt exceeded the model's window
  (263,213 tokens against a 262,144 limit) crashed before committing; the harness
  graded an empty patch and recorded a plain zero, indistinguishable from a model
  that never understood the task. Worth checking `peak_context_tokens` for models
  with small windows.

### Reading the aggregates

- `mean f2p` per task and pooled `sum(f2p_passed)/sum(f2p_total)` answer different
  questions. Pooling weights every test equally and is dominated by the big
  Python/TypeScript suites; the per-task mean weights every task equally. Report
  both, and say which one a number is.
- Pooled **p2p** is not a discriminator: five tasks hold 75.5% of the 231,352
  regression tests in the corpus (`expr-try-catch-errors` alone holds 38.7%), so a
  single task can move the pooled percentage by ~39 points. Count tasks with
  regressions instead of pooling.
- **Near-misses matter as much as the pass rate.** At the end of the GLM run, 25 of
  its 37 failures were tasks passing 90-99% of their target tests. The band
  distribution (all-passed / 90-99 / 50-89 / under 50) is the chart that explains a
  pass rate; the harness records enough to produce it.

### Where the records live

| What | Where |
| --- | --- |
| One record per run | `results/runs/<run-id>.json` (model identity, serving block, per-task results, timings, tokens) |
| Index of runs | `results/index.json` |
| Interactive report | `results/report.html` |
| Per-trial raw data | `jobs/<batch>/<task>__<hash>/` (agent trajectory, container logs, verifier output, `result.json`) |
| Progress state | `results/nightly-state-<model>.json` |
| Written conclusions | `eval/reports/*.md` |
| Driver logs | `results/nightly*.log` |

## Troubleshooting

- **Model calls fail with a proxy/connection error.** Check the forwarder:
  `python3 eval/deepswe_eval.py forwarder status` and `docker logs deepswe-llm-forward`.
  Confirm the target is reachable on the host (`curl http://127.0.0.1:4000/v1/models`
  with the right API key).
- **Port 80 already in use on 172.17.0.1.** Stop whatever binds it, or start the
  forwarder manually with `--listen <ip>:<port>` and a base URL host that resolves
  to that IP.
- **`nip.io` is blocked.** Replace `172.17.0.1.nip.io` with another name that
  resolves to the docker gateway, and keep the forwarder in place.
- **Disk space.** Each task image is ~0.84 GB compressed. Run subsets and clean
  up with `python3 eval/deepswe_eval.py clean --images --yes` (or
  `clean --dataset` to drop the generated dataset).
- **Long runs.** The per-task agent timeout is 3 hours. Use `--n-tasks`,
  `--languages`, and `--timeout-multiplier` to keep runs bounded.
