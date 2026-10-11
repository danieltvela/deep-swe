# GLM-5.3-Flash — corpus completo (113/113)

Evaluación terminada el 2026-10-11 a las 09:03. Arrancó el 2026-10-07 y se detuvo y reanudó varias
veces a petición; el corpus quedó completo, sin tareas abandonadas y con los 114 verificadores
auditados y limpios.

## Configuración

| Campo | Valor |
| --- | --- |
| Modelo | GLM-5.3-Flash (Zhipu AI), variante Flash |
| Pesos | W4A16 NVFP4 (K32 experts) + FP8 WO |
| Servicio | sglang (`glm53-flash-sm120`, EAGLE spec decode), TP 2, contexto 524288, KV `fp8_e4m3` |
| GPU | 2x RTX PRO 6000 Blackwell Max-Q (perfil `swe-agentic`, exclusivo con el de MiMo) |
| Endpoint | `openai/glm-5.3-flash-local` vía `http://172.17.0.1.nip.io/v1` → forwarder del puerto 80 → `127.0.0.1:8005` (directo, sin LiteLLM) |
| Agente | mini-swe-agent, `model_class: litellm`, secuencial (`n_concurrent=1`) |
| Orden de tareas | semilla 0, idéntico al de MiMo (comparación tarea a tarea) |
| Tope por tarea | 180 min, el del propio benchmark (`agent_timeout_multiplier: 1.0`) |

## Resultados finales

**76 de 113 resueltas (67,3 %)**, intervalo de confianza del 95 % **[58,2 % – 75,2 %]**.

| Métrica | Valor |
| --- | --- |
| f2p agregado (todos los tests objetivo) | 5 723 / 5 877 = **97,38 %** |
| f2p media por tarea | 94,71 % |
| p2p agregado (todos los tests de regresión) | 231 335 / 231 352 = **99,993 %** |
| p2p media por tarea | 99,74 % |
| Casi-aciertos (f2p ≥ 0,9 sin resolver) | 25 |
| Tareas con regresión de p2p | 5 |
| Duración por tarea | mediana 16,3 min, media 19,0, máximo 78,6 |
| Tiempo de agente / de pared | 32,9 h / 35,8 h |
| Pasos | 15 654 (139 por tarea) |

## Comparación con MiMo sobre las 113 tareas

| | GLM-5.3-Flash | MiMo-V2.6-Flash-MOPD |
| --- | --- | --- |
| Resueltas | **76/113 (67,3 %)**, IC [58,2 – 75,2] | 71/113 (62,8 %), IC [53,6 – 71,2] |
| f2p agregado | **97,38 %** | 94,52 % |
| f2p media por tarea | **94,71 %** | 90,59 % |
| p2p agregado | **99,993 %** (17 tests rotos) | 99,984 % (36 rotos) |
| p2p media por tarea | 99,74 % | 99,86 % |
| Casi-aciertos | 25 | 23 |
| Tareas con regresión p2p | **5** | 8 |
| Tiempo de agente | 32,9 h | 31,4 h |
| Pasos por tarea | 139 | 90 |

| Lenguaje | GLM | MiMo |
| --- | --- | --- |
| Go | 25/35 | **26/35** |
| Python | **25/34** | 21/34 |
| TypeScript | **21/34** | 20/34 |
| Rust | **3/5** | 1/5 |
| JavaScript | 2/5 | **3/5** |

Cada modelo resuelve tareas que el otro no: **19 solo GLM** (`aiomonitor-task-snapshots-diff`,
`bandit-structured-nosec-directives`, `claude-code-by-agents-recursive-delegation`,
`dasel-html-document-format`, `dateutil-rfc5545-timezone-interop`, `expr-try-catch-errors`,
`igel-persist-feature-schema`, `kea-atomic-signal-selectors`, `obsidian-linter-auto-table-of-contents`,
`opa-rego-rule-profiling`, `pest-character-class-coalescing`, `prometheus-transactional-reload-status`,
`python-statemachine-state-data-scoping`, `returns-validated-error-accumulation`,
`scriggo-method-declarations`, `sqlfmt-create-table-ddl-formatting`, `ts-pattern-match-each`,
`vitest-duration-sharding`, `wasmi-trap-coredumps`) y **14 solo MiMo**
(`awilix-async-container-initialization`, `bandit-interprocedural-taint-checks`,
`go-git-worktree-merge-conflicts`, `happy-dom-deterministic-intersectionobserver`,
`ipython-session-bundle-replay`, `koota-pair-relation-tracking`,
`mashumaro-flattened-dataclass-fields`, `onedump-dump-encryption-pipeline`,
`prometheus-typed-label-sorting`, `scc-bounded-memory-spilling`,
`superjson-error-stack-serialization`, `tengo-callable-instance-isolation`,
`testem-per-launcher-reports`, `updo-policy-alerting`).

## Perfil de los 37 fallos

- **25 casi-aciertos** con f2p entre 0,9 y 1,0: se quedan a uno o dos tests.
- **3 con todos los tests objetivo superados y regresión de p2p**:
  `gql-incremental-graphql-delivery` (17/17, p2p 810/811), `helm-array-merge-strategies` (47/47,
  p2p 10/12) y `vulture-persistent-analysis-cache` (24/24, p2p 291/295).
- **4 lejos del objetivo**: `boa-hierarchical-evaluation-cancellation` (0/17),
  `eicrud-keyset-pagination-cursor`, `kgateway-consistent-hash-policy` (0/2) y
  `onedump-dump-encryption-pipeline`. MiMo tiene 9 en este grupo, incluidas cuatro que GLM resuelve
  enteras.

Es decir, **28 de los 37 fallos están a un test de distancia**. El patrón de GLM: implementa lo que
se le pide (f2p agregado del 97,4 %) y su punto flaco son las regresiones sobre comportamiento ya
existente, concentradas en tareas de suite pequeña, que es lo que le baja la media por tarea de p2p
por debajo de MiMo pese a romper menos tests en total.

## Notas de método

- **El tope de 60 minutos fue nuestro y sesgaba la comparación.** Los `task.toml` del corpus declaran
  `[agent] timeout_sec = 10800` (3 h, valor que los autores del benchmark subieron desde 90 min en el
  commit `0b9fabb`). Las primeras ventanas usaron un multiplicador de 1/3 y eso cortó
  `oxvg-structural-selector-preservation` a los 60,0 min con 0/6; con el tope original da 4/6, igual
  que MiMo. MiMo nunca pasó de 46,3 min, así que su resultado no se vio afectado.
- **Auditoría de verificadores.** Se revisaron los 114 verificadores de GLM y los 113 de MiMo
  buscando errores de red (`Could not resolve host`) o fallo de compilación en ambos modos
  (`rc` idéntico distinto de cero): **cero inválidos**. El procedimiento nació de un caso real en el
  run de Laguna-S-2.1 (`oxvg`, `static.crates.io` inalcanzable), donde el verificador calificó el
  estado base y dio un 0/62 falso.
- **Parada de 23,7 h corregida.** El 2026-10-11 el daemon se quedó esperando porque el último lote
  cruzó la medianoche y la ventana cerró pasada su hora de inicio. `eval/nightly.py` ahora solo espera
  a la hora de inicio cuando a la ventana diaria en curso no le queda sitio para una tarea completa
  (`window_remaining_min`), y se probó con seis casos.

## Artefactos

- Datos: `results/runs/nightly-*.json`, `results/index.json`, `jobs/nightly-*/`
- Informe interactivo: `results/report.html` (cuatro modelos, Pareto de cuatro puntos)
- Estado de la evaluación: `results/nightly-state-glm.json` (113/113, nada pendiente)
- Comparación con los otros modelos en Rust: `eval/reports/qwen3.8-flash-next.md` y
  `eval/reports/laguna-s-2.1.md`
