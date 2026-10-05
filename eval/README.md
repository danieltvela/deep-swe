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

## Report

`python3 eval/deepswe_eval.py report` writes a self-contained `results/report.html`
(no CDN, data embedded as JSON) with:

- a runs comparison table (model identity, pass rate, duration);
- an **efficiency view**: per-run cost/quality metrics (agent-hours per task and
  per resolved task, output tokens per task and per resolved task, reasoning
  share, seconds per step, steps per task) plus a Pareto scatter of pass rate
  against agent-hours per task, where highlighted points are non-dominated;
- per-run language breakdown and a filterable/sortable task table;
- a side-by-side comparison of two runs over their shared tasks.

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
