# Evaluaciones pendientes de retomar

Registro actualizado el 2026-10-16: GLM-5.3-Flash y MiMo-V2.6-Flash-MOPD tienen el corpus completo
(113/113). Qwen3.8-Flash-Next-FP4-Local corre en continuo desde el 2026-10-16 22:57 con **1M de
contexto y TP=2** (repetición limpia de las 113 tareas; su fase 1 de 262.144 dejó 25 medidas).
Siguen a medias Qwen3.8-Flash-Next (7 de 113: las cinco de Rust y dos de la ventana del 2026-10-15,
cancelada a mitad) y Laguna-S-2.1 (5 de 113, solo el subconjunto de Rust). Los cinco modelos
comparten el puerto 8005, así que solo puede correr uno a la vez: este documento describe cómo
reanudar cada uno y dónde vive todo.

## Qwen3.8-Flash-Next — corpus completo, a medias (106 tareas pendientes)

Relanzada el 2026-10-15 a las 22:11 sobre el corpus completo y cancelada ese mismo día a las 23:07.
De las cuatro tareas que tocó: `etree-xml-diff-patch` (Go) y `yjs-map-conflict-detection`
(JavaScript) se resolvieron al 100 %, `mashumaro-flattened-dataclass-fields` (Python) se cortó
durante la verificación —el agente ya había terminado, pero no hay nota— y
`obsidian-linter-scoped-ignore-markers` no llegó a empezar. La cancelación no perdió nada más: el
lote cerrado quedó recogido en `results/runs/nightly-20261015-221124.json` y el informe se regeneró
a mano (el daemon no llegó a hacerlo).

| Dato | Valor |
| --- | --- |
| Fichero de estado | `results/nightly-state-qwen38-flash-next.json` (113 tareas en el orden de la semilla 0, el mismo que usaron MiMo y GLM) |
| Log | `results/nightly-qwen38.log` |
| Servidor | contenedor `vllm-qwen-flash-next`, perfil «hybrid», desde `~/services/tesla-docker`: `docker compose --profile hybrid up -d vllm-qwen-flash-nvfp4` (TP=2 sobre las dos GPU: los 123,6 GiB de pesos no caben en una sola) |
| Protocolo | ventana nocturna 22:00-05:00 con `--once`, tope por tarea 180 min (el del benchmark), secuencial, `--max-attempts 4` |

### Para retomarla

```bash
# 1. Servidor (perfil exclusivo con los de MiMo, GLM y Laguna: todos publican 8005)
cd ~/services/tesla-docker && docker compose --profile hybrid up -d vllm-qwen-flash-nvfp4

# 2. Forwarder del puerto 80 hacia el servidor
cd ~/projects/deep-swe
python3 eval/deepswe_eval.py forwarder stop
python3 eval/deepswe_eval.py forwarder start --target 127.0.0.1:8005

# 3. Ventana nocturna que acaba antes de las 08:00
setsid nohup python3 eval/nightly.py --model qwen3.8-flash-next \
    --state results/nightly-state-qwen38-flash-next.json \
    --start-now --start-at 22:00 --end-at 05:00 \
    --budget-hours 7 --task-timeout-min 180 --max-attempts 4 --once \
    >> results/nightly-qwen38.log 2>&1 < /dev/null &
```

Comprobación previa en un segundo: `python3 -c "import sys; sys.path.insert(0,'eval'); import nightly as N; print(N.model_reachable('qwen3.8-flash-next'))"` — debe imprimir `True`.

La ventana se cierra a las 05:00 porque, con el tope de 3 h por tarea, la última tarea puede
arrancar a las 04:45 y aun así terminar antes de las 08:00; con `--once` el daemon regenera el
informe y sale, en lugar de quedarse dormido hasta la noche siguiente. A ~19 min por tarea caben
unas 18 tareas por noche, así que el corpus completo necesita unas seis ventanas más: repetir ese
mismo comando cada noche es todo lo que hace falta.

Las cinco tareas de Rust se hicieron con el despliegue anterior (TP=1); entre uno y otro solo
cambia la velocidad, y el bloque `serving` de aquellos registros de ejecución sigue diciendo TP=1.

Nota sobre el orden: una ejecución con `--languages rust` **no** produce la subsecuencia de Rust del
orden completo (el generador aleatorio se consume por lenguaje y el filtro cambia el sorteo), así
que la ejecución de Rust de Qwen y la de Laguna siguen una secuencia distinta a la del corpus; la
comparación tarea a tarea no se ve afectada porque son las mismas tareas y la misma semilla.

## Qwen3.8-Flash-Next-FP4-Local — detenida a propósito (44 de 113 medidas a 1M)

Relanzada el 2026-10-16 a las 22:57 con **1M de contexto y TP=2** y **parada el 2026-10-17 a las
13:12** por decisión del usuario («ya tengo los datos que necesitaba»). Dejó 44 tareas medidas en
esta fase; las 69 restantes siguen pendientes en el estado, así que retomarla es relanzar el comando
de abajo.

**Resultados parciales de la fase 2 (44 tareas)**: 26 resueltas (59,1 %), f2p medio 91,3 %, f2p
agrupado 97,08 %, p2p agrupado 99,972 % (4 tareas con alguna regresión), 13 casi-aciertos y 2 ceros
—`boa-hierarchical-evaluation-cancellation` y `kgateway-consistent-hash-policy`, que GLM y MiMo
también fallan—. Por lenguaje: JavaScript 5/5, TypeScript 8/11, Go 8/12, Rust 2/5, Python 3/11. En
esas mismas 44 tareas GLM resolvió 28, MiMo 26 y el Qwen 26, con f2p medio 93,0 % / 91,6 % / 91,3 %.
Auditoría de los 43 verificadores de la fase: 0 con fallo de red y 0 con ambos modos fallando.

**Fase 1** (2026-10-16, 262.144 de contexto, TP=1 en la GPU 0): 25 tareas medidas —16 resueltas,
f2p medio 79,7 %, p2p 100,000 % sin una sola regresión—, parada para ampliar la ventana. Sus
resultados siguen en `results/runs/` y su estado en
`results/nightly-state-qwen3.8-flash-next-fp4-local.json`.

El motivo de subir a 1M: **3 de las 25 tareas de la fase 1 morían justo en el techo de 262.144**
(prompts de 262.145 tokens → `ContextWindowExceededError` → parche vacío). Con 1M ninguna tarea ha
vuelto a chocar (pico máximo 389.541 tokens) y esas tres quedaron así:
`dynamodb-toolbox-lazy-recursive-schemas` 0/37 → 37/37 resuelta, `effect-sse-httpapi-streaming`
0/47 → 44/47 y `wasmi-trap-coredumps` 0/22 → 20/22.

**Cierre limpio**: el lote en vuelo (`nightly-20261017-130301`) se cortó con un único intento
cancelado y sin nota; su registro sin puntuar se descartó para no ensuciar el informe, el
directorio de pier se conserva en `jobs/` y los contadores de intento de sus tres tareas
(`igel-persist-feature-schema`, `drizzle-orm-window-function-builders`,
`prometheus-transactional-reload-status`) volvieron a 0.

| Dato | Valor |
| --- | --- |
| Fichero de estado | `results/nightly-state-qwen3.8-flash-next-fp4-local-tp2.json` (113 tareas, semilla 0, orden idéntico al de MiMo, GLM y Qwen-NVFP4; 0 completadas al lanzar) |
| Entrada en el registro | `eval/models.yaml` → `qwen3.8-flash-next-fp4-local` (bloque `serving` ya actualizado a 1M/TP=2), alias servido `qwen38-flash-next`, con `agent.verify_served_alias: true` |
| Servidor | contenedor `qwen38-flash-next-fp4-local-tp2`, perfil `qwen38-fp4-local-tp2`, preset `qwen38-tp2`: TP=2, `--max-model-len 1048576` (YaRN factor 4 sobre las 262.144 del config), KV fp8, `mamba-cache-mode align`, ~40,85 GiB de pesos por GPU y ~27 GiB de RAM host para la tabla PLE |
| Log | `results/nightly-qwen38-fp4.log` (el mismo fichero que la fase 1) |
| Protocolo | continuo (`--start-at 00:00 --end-at 23:59 --budget-hours 24`, sin `--once`), tope por tarea 180 min (el del benchmark), secuencial, `--max-attempts 4` |

### Pasos para lanzarla

```bash
# 1. Parar quien ocupe las GPU y la RAM (hoy GLM: los perfiles son exclusivos y la
#    tabla PLE vive en RAM host). Con GLM cargado quedaban 16 GB disponibles.
cd ~/services/tesla-docker
docker compose stop glm53-flash-sm120
free -g

# 2. Arrancar el modelo FP4 con TP=2 (primer init: ~2 min de carga + warmup; el
#    healthcheck da 2400 s de margen)
docker compose --profile qwen38-fp4-local-tp2 up -d qwen38-flash-next-fp4-local-tp2
docker ps --filter name=qwen38-flash-next-fp4-local-tp2

# 3. Comprobar que es ÉL quien sirve el 8005 (el forwarder ya apunta a 8005, no se toca)
curl -s http://172.17.0.1.nip.io/v1/models
#    debe listar qwen38-flash-next y no glm-5.3-flash-local
python3 -c "import sys; sys.path.insert(0,'eval'); import nightly as N; print(N.model_reachable('qwen3.8-flash-next-fp4-local'))"
#    y el log del contenedor debe decir: GPU KV cache size: 6,328,464 tokens,
#    "Maximum concurrency for 1,048,576 tokens per request: 6.04x"

# 4. Lanzar (continuo: sin ventana ni --once, se para solo al completar el corpus)
cd ~/projects/deep-swe
setsid nohup python3 eval/nightly.py --model qwen3.8-flash-next-fp4-local \
    --state results/nightly-state-qwen3.8-flash-next-fp4-local-tp2.json \
    --start-now --start-at 00:00 --end-at 23:59 \
    --budget-hours 24 --task-timeout-min 180 --max-attempts 4 \
    >> results/nightly-qwen38-fp4.log 2>&1 < /dev/null &
```

Con `--start-at 00:00 --end-at 23:59 --budget-hours 24` la ventana se renueva cada medianoche con
~1 min de pausa, así que el run es continuo y no hay horario que respetar (es el mismo patrón con
el que GLM terminó su corpus). Sin `--once` el daemon sale al completar las 113 tareas, y entonces
regenera `results/report.html`. Ojo con la variante nocturna de las otras entradas
(`--end-at 05:00 --once`): ahí la ventana sí termina a una hora concreta, y un
`--budget-hours` pequeño la cortaría antes (`min(ahora + budget, próximo 05:00)`).

### Dos avisos que salieron al prepararla

- **El sondeo del daemon no distinguía el modelo.** Con el 8005 servido por GLM,
  `model_reachable('qwen3.8-flash-next-fp4-local')` devolvía `True`: sglang responde 200 a
  cualquier nombre de modelo. La entrada lleva ahora `agent.verify_served_alias: true`, que exige
  que el endpoint anuncie `qwen38-flash-next` en `/v1/models` antes de abrir ventana; con otro
  modelo en el puerto el daemon se queda en «cannot generate; retrying in 60s» en lugar de evaluar
  el modelo equivocado. La opción es por modelo: no se activó en los demás para no bloquear a los
  que van por litellm (el alias `chat-agentic` no aparece en su `/v1/models`).
- **El alias servido es `qwen38-flash-next`**, el `SERVED_MODEL_NAME` del compose, y es el que pide
  el agente. El paso 3 confirma qué anuncia el servidor de verdad antes de lanzar nada.

## GLM-5.3-Flash — COMPLETADO (113/113)

| Dato | Valor |
| --- | --- |
| Fichero de estado | `results/nightly-state-glm.json` (113 completadas, 0 pendientes) |
| Entrada en el registro | `eval/models.yaml` → `glm-5.3-flash` |
| Orden de tareas | semilla 0, idéntico al de MiMo, así que la comparación sigue siendo tarea a tarea |
| Protocolo | tope por tarea 180 min (el del benchmark), secuencial, `--max-attempts 4` |
| Log | `results/nightly-glm.log` |
| Resultados parciales | `eval/reports/glm-5.3-flash.md`, `results/runs/nightly-*.json`, `results/report.html` |

### Para retomarla

Hace falta el servidor de GLM y el forwarder apuntando a él:

```bash
# 1. Servidor (desde ~/services/tesla-docker; perfil exclusivo con el de MiMo)
docker compose --profile swe-agentic up -d glm53-flash-sm120

# 2. Forwarder del puerto 80 hacia el servidor de GLM
python3 eval/deepswe_eval.py forwarder stop
python3 eval/deepswe_eval.py forwarder start --target 127.0.0.1:8005

# 3. Reanudar (continuo, día y noche)
setsid nohup python3 eval/nightly.py --model glm-5.3-flash \
    --state results/nightly-state-glm.json \
    --start-now --start-at 00:00 --end-at 23:59 \
    --budget-hours 24 --task-timeout-min 180 --max-attempts 4 \
    >> results/nightly-glm.log 2>&1 < /dev/null &
```

Para una ventana nocturna en lugar de continua, cambia los dos últimos por
`--start-at 22:00 --end-at 05:00` (así la GPU queda libre a las 08:00 aunque una tarea
arranque a las 04:45 con su tope de 3 h).

Comprobación previa en un segundo: `python3 -c "import sys; sys.path.insert(0,'eval'); import nightly as N; print(N.model_reachable('glm-5.3-flash'))"` — debe imprimir `True`.

### Resultados acumulados (70 tareas)

47 resueltas (67,1 %), IC 95 % [55,5 % – 77,0 %], f2p medio 0,953, 19 casi-aciertos, 4 regresiones
de p2p. Por lenguaje: Go 17/20, Python 13/20, TypeScript 12/20, Rust 3/5 (completo), JavaScript 2/5.
Frente a MiMo en esas mismas 70 tareas: 47 frente a 39 resueltas y f2p 0,953 frente a 0,905.

## MiMo-V2.6-Flash-MOPD — completo, nada pendiente

113/113 tareas, estado en `results/nightly-state.json`, conclusiones en
`eval/reports/mimo-v2.6-flash-mopd.md`. Es la referencia contra la que se comparan los demás
modelos: 71 resueltas (62,8 %), f2p medio 0,906.

## Añadir otro modelo

El procedimiento completo está en `eval/README.md`, sección «Evaluating a second model». El
resumen: añadir la entrada en `eval/models.yaml` (identidad, cuantización, laboratorio, variante y
bloque `serving`), levantar su servidor, reapuntar el forwarder al puerto donde escuche, y lanzar
el driver con **su propio fichero de estado** (`--state results/nightly-state-<modelo>.json`).

Dos restricciones del entorno que conviene recordar: los perfiles de modelo son mutuamente
excluyentes porque todos publican el puerto 8005 con TP=2 sobre las dos GPU, y `forwarder start`
no cambia el destino si el contenedor ya está corriendo (hay que pararlo antes).

## Dónde está todo

| Qué | Dónde |
| --- | --- |
| Registros por lote | `results/runs/nightly-*.json` |
| Índice de runs | `results/index.json` |
| Informe interactivo | `results/report.html` (se regenera al cerrar cada ventana) |
| Directorios de trabajo de pier | `jobs/<lote>/` (cada intento guarda su `result.json` y su trayectoria) |
| Informes escritos por modelo | `eval/reports/*.md` |
| Configuración | `eval/languages.yaml`, `eval/models.yaml` |

Para forzar el informe: `python3 eval/deepswe_eval.py report`.

## Si hay que parar a mitad

Parar el daemon no pierde lo terminado: al cerrar cada lote se anota en el estado. Lo que hay que
hacer al parar es recoger lo que ya había acabado dentro del lote en curso
(`python3 eval/deepswe_eval.py collect --job <lote> --model <modelo>`) y devolver a cero el
contador de intentos de la tarea que estaba en vuelo. Así se hizo en las tres paradas de esta
sesión.
