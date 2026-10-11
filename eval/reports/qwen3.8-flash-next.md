# Qwen3.8-Flash-Next — Rust (5 de 5 tareas del corpus)

Evaluación completa del subconjunto de Rust, ejecutada la noche del 2026-10-09 al 2026-10-10
(cierre a las 08:23). Es la única parte del corpus evaluada para este modelo: el resto de las 108
tareas no se ha tocado. Rust es además el grupo más pequeño del benchmark (5 de 113 tareas), así que
los porcentajes son orientativos.

## Configuración

| Campo | Valor |
| --- | --- |
| Modelo | Qwen3.8-Flash-Next (Alibaba), variante Flash |
| Cuantización | NVFP4 (`vLLM --quantization modelopt`) |
| Servicio | vLLM (`vllm/vllm-openai:nightly`, contenedor `vllm-qwen-flash-next`), puerto 8005 |
| Contexto | 262 144 tokens — el más corto de los tres modelos comparados |
| Paralelismo | TP=1 (una sola GPU) |
| Decodificación especulativa | MTP con 3 tokens |
| Endpoint | `openai/qwen3.8-flash-next` vía `http://172.17.0.1.nip.io/v1` → forwarder del puerto 80 → `127.0.0.1:8005` |
| Agente | mini-swe-agent, `model_class: litellm`, secuencial, tope por tarea 180 min (el del benchmark) |

## Las 5 tareas de Rust, modelo a modelo

| Tarea | Qwen3.8-Flash-Next | MiMo-V2.6-Flash-MOPD | GLM-5.3-Flash |
| --- | --- | --- | --- |
| `boa-hierarchical-evaluation-cancellation` | 0 — 0/17, 20,9 min | 0 — 0/17, 27,2 min | 0 — 0/17, 29,2 min |
| `fd-deterministic-multi-key-sorting` | 1 — 43/43, 13,1 min | 1 — 43/43, 11,2 min | 1 — 43/43, 10,6 min |
| `oxvg-structural-selector-preservation` | **1 — 6/6, 21,7 min** | 0 — 4/6, 33,3 min | 0 — 4/6, 78,6 min |
| `pest-character-class-coalescing` | 0 — 102/104, 16,7 min | 0 — 100/104, 14,4 min | **1 — 104/104, 17,8 min** |
| `wasmi-trap-coredumps` | 0 — 0/22, 21,5 min | 0 — 20/22, 25,8 min | **1 — 22/22, 26,5 min** |

| | Resueltas | f2p medio | p2p medio | Pared total | Agente total | Pasos/tarea | Salida total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GLM-5.3-Flash | **3/5** | **0,733** | 1,0000 | 2,71 h | 2,56 h | 246 | 661 641 |
| Qwen3.8-Flash-Next | 2/5 | 0,596 | 0,9992 | **1,57 h** | **1,46 h** | 142 | 572 703 |
| MiMo-V2.6-Flash-MOPD | 1/5 | 0,707 | 0,9992 | 1,86 h | 1,71 h | 129 | 657 259 |

## Lectura tarea por tarea

**`boa-hierarchical-evaluation-cancellation`: los tres fallan igual.** 0 de 17 tests objetivo y la
suite de regresión intacta (7/7) en los tres casos. No es una tarea que discrimine entre modelos.

**`fd-deterministic-multi-key-sorting`: los tres la resuelven.** MiMo y GLM en torno a 11 minutos con
70 y 144 pasos; Qwen en 13,1 con 122. Diferencia de coste, no de resultado.

**`oxvg-structural-selector-preservation`: solo Qwen.** Resuelve los 6 tests objetivo y los 62 de
regresión en 21,7 minutos; MiMo y GLM se quedan en 4/6 y necesitan 33 y 78 minutos. Es la tarea que
más separa a este modelo para bien.

**`pest-character-class-coalescing`: solo GLM.** GLM llega a 104/104 y 250/250; Qwen se queda a dos
tests (102/104) con un fallo de regresión (249/250), y MiMo a cuatro (100/104). Los tres tiempos son
parecidos (14-18 min), así que la diferencia está en el resultado, no en el esfuerzo.

**`wasmi-trap-coredumps`: solo GLM, y aquí Qwen no llega a ser evaluado.** El agente de Qwen creció
hasta **263 213 tokens de prompt contra el límite de 262 144** del modelo, la llamada falló con
`ContextWindowExceededError` y el agente terminó con código distinto de cero antes de hacer commit.
pier recogió un parche vacío y el verificador calificó el estado base: 0/22 en objetivo y 58/58 en
regresión. GLM la resolvió entera (22/22) y MiMo se quedó a dos tests (20/22).

## Presión de contexto: el rasgo que define a Qwen aquí

Las tareas de Rust son largas en contexto para los tres modelos, y Qwen es el que tiene la ventana
más corta (262 144 tokens frente a 448 512 de MiMo y 524 288 de GLM):

| Tarea | Qwen3.8 | MiMo | GLM |
| --- | --- | --- | --- |
| `wasmi-trap-coredumps` | 261 577 (falló al pedir 263 213) | 326 215 | 339 270 |
| `oxvg-structural-selector-preservation` | 223 930 | 299 266 | 403 681 |
| `boa-hierarchical-evaluation-cancellation` | 222 269 | 272 979 | 212 953 |
| `pest-character-class-coalescing` | 195 754 | 176 085 | 142 423 |
| `fd-deterministic-multi-key-sorting` | 181 963 | 158 124 | 159 017 |

Qwen trabaja pegado a su techo en cuatro de las cinco tareas. MiMo y GLM superan los 260 000 tokens
sin problema porque su ventana es mayor; con la ventana de Qwen, `wasmi` habría fallado igual.

## Coste y velocidad

Qwen es **el más rápido de los tres en total** (1,57 h de pared frente a 1,86 de MiMo y 2,71 de GLM)
y usa **una sola GPU** por el TP=1, mientras los otros dos ocupan las dos. Por tarea gasta 142 pasos
de media, entre los 129 de MiMo y los 246 de GLM, y 572 703 tokens de salida en total con un 62 % de
razonamiento.

Su perfil de acierto es de extremos: resuelve `oxvg` entera y en la mitad de tiempo que los otros, y
en `wasmi` se queda en cero porque no llegó a entregar nada. Eso se refleja en un f2p medio (0,596)
por debajo de MiMo (0,707) y GLM (0,733) pese a resolver una tarea más que MiMo.

## Cómo ampliarlo

- **Resto del corpus para Qwen** (108 tareas): mismo procedimiento, sin `--languages rust`, con su
  propio fichero de estado. A su ritmo (unas 19 min por tarea de media en este subconjunto) son del
  orden de 30 horas.
- **Retomar GLM**: quedan 43 tareas, véase `eval/reports/pending-evaluations.md`.

Aviso para futuras ejecuciones de este modelo: con 262 144 tokens de ventana conviene vigilar el
pico de contexto (`peak_context_tokens` en los registros), porque al pasarlo el resultado es un cero
con parche vacío y no un fallo parcial.
