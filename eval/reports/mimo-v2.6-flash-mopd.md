# DeepSWE evaluation — MiMo-V2.6-Flash-MOPD

Full-corpus local evaluation of a self-hosted model, run with `eval/deepswe_eval.py`
and `eval/nightly.py` on the DeepSWE v1.1 corpus (113 tasks).

- **Evaluation closed**: 2026-10-08 06:36 CEST
- **Corpus**: 113/113 tasks completed, 0 discarded, 0 given up
- **Report**: `results/report.html` (regenerated at close)
- **Raw data**: `results/runs/nightly-*.json`, `jobs/nightly-*/`

## Model under test

| Field | Value |
| --- | --- |
| Name | MiMo-V2.6-Flash-MOPD |
| Lab | Xiaomi |
| Variant | MOPD |
| Quantization | W4A8 (MXFP4 experts + FP8 dense) |
| Context length | 448 512 |
| KV cache | fp8 |
| Tensor parallel | 2 |
| GPUs | 2x RTX PRO 6000 Blackwell Max-Q |
| Serving stack | vLLM (mimo-v26-omni image, DFlash spec decode, kv-offloading) |
| Reasoning tokens | 71.5 % of generated tokens |

Served via LiteLLM alias `swe-agentic`, reached from task containers through a TCP
forwarder on `172.17.0.1:80` (pier's egress proxy only allows ports 80/443).

## Protocol

- Agent: `mini-swe-agent` (chat-completions adapter, `model_class: litellm`).
- Sequential (`n_concurrent = 1`), one task at a time, per the operator's choice.
- Per-task agent cap: 60 min (the task's own cap is 180 min). **The cap never
  triggered**: the longest task took 45.1 min, so scores are not truncated.
- Task order: fixed, seeded, language-balanced interleave over all 113 tasks.
- Agent sees only `instruction.md` plus the repository at the base commit; the
  graded tests live in a separate verifier image the agent never sees.

## Headline results

| Metric | Value |
| --- | --- |
| Resolved (binary official score) | **71 / 113 = 62.8 %** |
| 95 % CI (Wilson) | [53.6 %, 71.2 %] |
| Mean f2p (graded) | 0.906 |
| Mean p2p | 0.999 |
| Near-misses (f2p ≥ 0.9 yet reward 0) | 26 |
| Clean misses (f2p = 0) | 6 |
| Tasks with a p2p regression | 8 |

## Per language

| Language | Resolved | Pass rate | Mean f2p |
| --- | --- | --- | --- |
| Go | 26/35 | 74 % | 0.91 |
| Python | 21/34 | 62 % | 0.95 |
| JavaScript | 3/5 | 60 % | 0.99 |
| TypeScript | 20/34 | 59 % | 0.87 |
| **Rust** | **1/5** | **20 %** | **0.71** |

**Rust is the weak point, and it is final**: the corpus contains only 5 Rust tasks
and all 5 were evaluated. Its pass rate is less than a third of Go's, and its mean
f2p is the only one below 0.88 — in Rust the model does not merely fail more, it
fails further from the target.

## Failure analysis

- **The dominant failure mode is "almost".** Of 42 unsolved tasks, 26 sit above
  0.9 mean f2p. The binary reward requires every f2p test to pass with no p2p
  regression, so a task that implements nearly all the requested behaviour still
  scores 0. Examples: `testem-bail-on-test-failure` (89/90 f2p, 489/489 p2p),
  `httpx-streaming-json-iteration` (107/108, 1404/1404),
  `dasel-html-document-format` (144/146, 1012/1012).
- **Clean misses (only 6)**: `boa-hierarchical-evaluation-cancellation`,
  `kgateway-consistent-hash-policy`, `obsidian-linter-auto-table-of-contents`,
  `eicrud-keyset-pagination-cursor`, `ts-pattern-match-each`,
  `scriggo-method-declarations`. These are tasks where the requested behaviour was
  not implemented at all.
- **Regressions: 8 of 113 (7 %).** Mean p2p is 0.999, so the model rarely touches
  existing tests, but when it does the task is lost even if most f2p pass. This is
  the quantified version of the "it broke my working CI while fixing it" incident:
  roughly one in fourteen tasks damages something that already worked.
- **`partial` is misleading and was not used as a graded metric.** It is
  `(f2p_passed + p2p_passed) / (f2p_total + p2p_total)`, and p2p tests outnumber
  f2p tests many times over, so it reads ~0.97 even for 4/6 f2p. Use mean f2p.

## Cost

| Metric | Value |
| --- | --- |
| Agent wall time | 31.4 h |
| Wall time inside batches | 34.0 h |
| Batches | 27 |
| Mean / max agent time per task | 16.7 min / 45.1 min |
| Input tokens | 1 339 669 439 |
| Output tokens | 12 503 481 |
| Reasoning tokens | 8 941 256 (71.5 % of output) |
| Steps | 10 190 (≈90 per task) |
| Output tokens per task | ≈110 650 |

Input tokens outnumber output by about 107:1. With a growing agent context
re-sent every step, prefill dominates the workload; prompt caching on the server
is therefore the highest-leverage knob for wall-clock time.

## Timeline

| Window | Result |
| --- | --- |
| 2026-10-06 00:29 → 07:54 | 21/113 |
| 2026-10-06 19:52 → 2026-10-07 07:52 | 64/113 |
| 2026-10-07 15:57 → 2026-10-08 06:36 | **113/113, done** |

The driver regenerated the report and exited on its own when the corpus completed.

## Caveats

- The binary reward is the official DeepSWE metric, so this number is
  leaderboard-comparable in definition; the 95 % CI is still ±9 points.
- The verifiers are the benchmark authors' work. The oracle/nop audits that would
  test them (golden solution must score 1; baseline must fail every f2p) were
  spot-checked on 3 tasks but not run across all 113. See
  `eval/README.md` for the procedure.
- A 60-minute agent cap was in force. It never triggered, so it did not affect
  these scores, but it would bind on a slower model.

## Takeaways

1. For general long-horizon repo work in Go/Python/TypeScript, this model is
   competitive: 59–74 % per language, and it almost never breaks existing tests.
2. For **Rust** it is the wrong tool: 20 % pass, mean f2p 0.71.
3. Expect "close but not exact" outcomes: around two thirds of its failures are
   one assertion short, which is expensive if the acceptance criterion is strict.
4. Around 7 % of tasks damage working behaviour — relevant when using it
   unattended on a real CI pipeline.

## Reproduce

```bash
python3 eval/deepswe_eval.py build
python3 eval/nightly.py --model mimo-v2.6-flash --start-now \
    --start-at 22:00 --end-at 08:00 --budget-hours 24 --task-timeout-min 60
python3 eval/deepswe_eval.py report
```
