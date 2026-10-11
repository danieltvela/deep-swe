# Laguna-S-2.1 — Rust (5 de 5 tareas del corpus)

Evaluación del subconjunto de Rust, ejecutada el 2026-10-10 (primer cierre 11:45, repetición de una
tarea hasta las 12:30). Es la única parte del corpus evaluada para este modelo. Rust es el grupo más
pequeño del benchmark (5 de 113 tareas), así que los porcentajes son orientativos.

## Configuración

| Campo | Valor |
| --- | --- |
| Modelo | Laguna-S-2.1 (Poolside), variante S |
| Cuantización | NVFP4 |
| Servicio | vLLM (`vllm/vllm-openai:nightly`, contenedor `laguna-s-2.1-nvfp4`), puerto 8005 |
| Contexto | 1 048 576 tokens (el mayor de los cuatro modelos) |
| Paralelismo | TP=2, repartido entre las dos GPU |
| KV cache | fp8 |
| Decodificación especulativa | DFlash con 7 tokens desde un modelo borrador en `/draft` |
| Parsers | `poolside_v1` para razonamiento y herramientas; `enable_thinking: true` |
| Agente | mini-swe-agent, `model_class: litellm`, secuencial, tope por tarea 180 min (el del benchmark) |

Aviso de métricas: este servidor devuelve el razonamiento en un campo no estándar (`reasoning`) y
siempre informa `reasoning_tokens = 0`, así que la columna de razonamiento del informe sale a cero
para este modelo. El texto sí queda en las trayectorias como `reasoning_content`.

## Las 5 tareas de Rust, cuatro modelos

| Tarea | Laguna-S-2.1 | MiMo | GLM | Qwen |
| --- | --- | --- | --- | --- |
| `boa-hierarchical-evaluation-cancellation` | 0/17, p2p 7/7 | 0/17 | 0/17 | 0/17 |
| `fd-deterministic-multi-key-sorting` | 42/43, p2p 109/109 | 43/43 ✓ | 43/43 ✓ | 43/43 ✓ |
| `oxvg-structural-selector-preservation` | **6/6, p2p 60/62** | 4/6 | 4/6 | 6/6 ✓ |
| `pest-character-class-coalescing` | 82/104, p2p 249/250 | 100/104 | 104/104 ✓ | 102/104 |
| `wasmi-trap-coredumps` | 8/22, p2p 58/58 | 20/22 | 22/22 ✓ | 0/22 |

| | Resueltas | f2p medio | p2p medio | Tiempo total | Pasos/tarea | Salida total |
| --- | --- | --- | --- | --- | --- | --- |
| GLM-5.3-Flash | **3/5** | **0,733** | 1,0000 | 2,71 h | 246 | 661 641 |
| Qwen3.8-Flash-Next | 2/5 | 0,596 | 0,9992 | **1,57 h** | 142 | 572 703 |
| MiMo-V2.6-Flash-MOPD | 1/5 | 0,707 | 0,9992 | 1,86 h | 129 | 657 259 |
| Laguna-S-2.1 | 0/5 | 0,626 | 0,9927 | 3,03 h | **365** | 979 736 |

## Lectura

**Es el más flojo de los cuatro y también el que más trabaja.** No resuelve ninguna tarea, gasta 365
pasos por tarea (frente a 129-246 de los demás), 3,03 horas en total y casi un millón de tokens de
salida. Su f2p medio (0,626) queda por encima del de Qwen pero sin ninguna resolución completa.

**En `oxvg` implementa todo el comportamiento pedido** (6 de 6 tests objetivo) y falla por dos
regresiones: rompe dos tests que ya funcionaban (60/62). Es el espejo de MiMo y GLM, que se quedan en
4/6 sin romper nada, y contrasta con Qwen, que lo resuelve entero y limpio.

**En `fd` se queda a un test** (42/43) de una tarea que los otros tres resuelven, y en `pest` (82/104)
y `wasmi` (8/22) va claramente por detrás. En `boa` los cuatro fallan igual (0/17).

## Incidencia metodológica: un verificador inválido

La primera pasada dio para `oxvg` un 0/6 con **0/62** en regresión, que habría parecido una rotura
masiva. No lo era: el verificador falló en **los dos modos** (`rc=102` en base y en nuevo) con 64
errores `Could not resolve host: static.crates.io`, es decir, el contenedor de verificación no pudo
descargar las dependencias de Rust y no llegó a compilar ni el estado limpio; sin XML de JUnit, el
verificador cuenta todos los tests como fallidos.

La tarea se reencoló y se repitió: la segunda pasada salió con verificador válido (`rc=100`/`rc=0`,
cero errores de red) y dio el 6/6 con p2p 60/62 que figura en la tabla.

Se auditó **los 194 verificadores de todos los runs** buscando ese patrón (errores de red o fallo en
ambos modos): el único inválido era ese. Los 113 de MiMo, los 71 de GLM y los 5 de Qwen están limpios.

## Cómo ampliarlo

- **Resto del corpus** (108 tareas): mismo procedimiento sin `--languages rust`. A su ritmo (unos 36
  min por tarea en este subconjunto) son del orden de 65 horas.
- **Retomar GLM**: quedan 43 tareas, véase `eval/reports/pending-evaluations.md`.
- Antes de dar por bueno cualquier run futuro, conviene repetir la auditoría de verificadores: es un
  `grep` de `Could not resolve host` y `rc=` sobre `jobs/*/*/verifier/test-stdout.txt`.
