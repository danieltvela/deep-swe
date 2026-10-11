# Evaluaciones pendientes de retomar

Registro actualizado el 2026-10-11: **ya no queda ninguna evaluación a medias**. GLM-5.3-Flash
terminó el corpus completo (113/113) el 2026-10-11 a las 09:03, así que este documento queda como
referencia del procedimiento de reanudación y de dónde vive cada cosa.

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
