# DIMMIT V1

Detección de deterioro vial combinando **visión por computador** (YOLOv10 sobre
imágenes de la vía) con **datos de sensores** (acelerómetro, giroscopio, GPS), que
se fusionan en un **score de condición 0–100 por segmento de vía**. Primera
iteración (v0) según el Reporte Técnico V1.

```
RDD2020 (Mendeley)          sensores (simulados por ahora)
      │                                │
      ▼                                ▼
 download ──► prepare ──► validate ◄── simulate
 (tar 1.4GB)  (VOC→YOLO,   (checks)    (50 Hz por segmento)
               subset)
      │                                │
      ▼                                ▼
 YOLOv10n train/predict ──► detecciones ──► features por segmento ──► score v0
                                             (severidad visual +        (0–100,
                                              rugosidad RMS)            bueno/regular/malo)
```

## Quickstart

```bash
make setup      # venv (Python 3.11 vía uv) + dependencias
make pipeline   # descarga RDD2020, prepara subset YOLO, simula sensores y valida
make train-small
make predict
make score
make test
```

`make all` corre pipeline + entrenamiento + inferencia + score de una vez. Los
resultados de la última corrida están en
[reports/resultados_2025-09-23.md](reports/resultados_2025-09-23.md).

## Datos

### Externos: RDD2020

[RDD2020](https://data.mendeley.com/datasets/5ty2wb6gvg/1) — 26 336 imágenes de
vías tomadas con smartphones montados en vehículos (India, Japón y Chequia), con
más de 31 000 instancias de daño anotadas en Pascal VOC XML. Fue el dataset del
Global Road Damage Detection Challenge 2020 (licencia CC BY-NC 3.0).

| ID | Clase VOC | Daño |
|----|-----------|------|
| 0 | D00 | Grieta longitudinal |
| 1 | D10 | Grieta transversal |
| 2 | D20 | Grieta tipo cocodrilo |
| 3 | D40 | Bache |

Los XML traen clases adicionales (D01, D44, …) que el pipeline descarta. Solo
`train.tar.gz` tiene anotaciones (test1/test2 no; se descargan con
`--all` si se necesitan). La primera iteración usa un subset reproducible de
~300 imágenes de Japón — país, tamaño, split y seed son configurables en
[configs/data_rdd2020.yaml](configs/data_rdd2020.yaml).

Si la descarga automática falla, usar "Download All" en Mendeley y dejar los
archivos en `data/raw/`.

### Internos: sensores (simulados)

Los sensores del reporte técnico aún no existen físicamente, así que
[simulate_sensors.py](src/dimmit/data/simulate_sensors.py) genera series dummy con
el **esquema definitivo**, para que el resto del pipeline no cambie cuando lleguen
datos reales:

| Columna | Descripción |
|---|---|
| `segment_id` | Segmento de vía (10 imágenes por segmento) |
| `timestamp` | 10 s a 50 Hz por segmento |
| `ax, ay, az` | Acelerómetro triaxial (m/s²; `az` incluye gravedad) |
| `gx, gy, gz` | Giroscopio (rad/s) |
| `lat, lon, speed_kmh` | GPS: ruta simulada desde Bogotá y velocidad |

La vibración vertical se correlaciona con la severidad del daño anotado en las
imágenes del segmento (más daño ⇒ más ruido y picos tipo bache), de modo que el
modelo de scores tiene señal realista para validarse: la correlación
score vs severidad simulada es ≈ −0.98 usando solo sensores.

Salidas: `data/simulated/sensors.parquet` (series) y `segments.csv`
(mapeo imagen → segmento + severidad anotada).

## Pipeline de datos

`make pipeline` ejecuta [dimmit/pipeline/run.py](src/dimmit/pipeline/run.py), que
orquesta las cuatro etapas como subprocesos y se detiene si alguna falla. Todas son
**idempotentes** (saltan lo ya hecho), y se pueden correr sueltas con
`--stages prepare,validate` o con sus targets de make.

| Etapa | Módulo | Qué hace |
|---|---|---|
| download | `dimmit.data.download` | Baja RDD2020 desde la API pública de Mendeley (reintentable; verifica tamaño) |
| prepare | `dimmit.data.prepare` | Extrae el país configurado, convierte VOC→YOLO, muestrea el subset y hace split train/val; genera `data/processed/data.yaml` |
| simulate | `dimmit.data.simulate_sensors` | Genera los sensores dummy por segmento |
| validate | `dimmit.data.validate` | Checks de calidad; sale con código ≠ 0 si algo falla |

La validación revisa: pareo imagen↔label 1:1 por split, bboxes normalizados en
[0,1] y clases en 0–3, `data.yaml` generado, esquema de sensores, ausencia de
nulos, plausibilidad física (`az > 0`) y cobertura completa imagen↔segmento.

## Modelos

### Detector: YOLOv10 (PyTorch)

Se usa el fork oficial [THU-MIG/yolov10](https://github.com/THU-MIG/yolov10) como
dependencia pip (API estilo ultralytics). Los pesos preentrenados (`yolov10n.pt`,
COCO) se descargan automáticamente del release v1.1 del fork.
[train.py](src/dimmit/models/detector/train.py) hace fine-tune según
[configs/train_small.yaml](configs/train_small.yaml) (device `mps` en Apple
Silicon, `cpu` como fallback) y
[predict.py](src/dimmit/models/detector/predict.py) exporta las detecciones del
split de validación a `reports/detections.csv`.

**Por qué PyTorch y no TensorFlow**: no existe port oficial de YOLOv10 a TF, y
reimplementar la arquitectura (entrenamiento NMS-free con doble asignación,
cabezas v10) es un esfuerzo de semanas que no aporta a esta fase. Si más adelante
se necesita TF/TFLite (p. ej. despliegue en móvil), la ruta recomendada es
exportar el modelo entrenado vía **ONNX** (`model.export(format="onnx")`), que el
fork soporta.

### Score v0: índice de condición por segmento

`score = 100 − Σ_clase (w_c · severidad_c) − w_sensor · rugosidad_norm`

- **Severidad visual** por clase: nº de detecciones × (1 + 10 × área relativa
  media de las cajas) — castiga más los daños grandes.
- **Rugosidad**: RMS de la aceleración vertical sin gravedad por segmento,
  normalizada 0–1 sobre la flota.
- Pesos, umbral de confianza y cortes de categoría (bueno ≥ 75, regular ≥ 50,
  malo < 50) en [configs/score.yaml](configs/score.yaml).

La implementación ([model.py](src/dimmit/models/score/model.py)) separa la
featurización ([features.py](src/dimmit/models/score/features.py)) del scorer:
`RuleBasedScorer` implementa el índice del reporte técnico y `LearnedScorer` es el
placeholder para entrenar un modelo (RandomForest/XGBoost) sobre las mismas
features cuando existan etiquetas reales de condición (p. ej. PCI medido en
campo). Salida: `reports/scores.csv` con score, categoría y features por segmento.

## Estructura del repositorio

```
configs/            # dataset, hiperparámetros del entrenamiento corto, pesos del score
data/               # raw → interim → processed (YOLO) + simulated (gitignored)
src/dimmit/
  data/             # download, prepare (VOC→YOLO), simulate_sensors, validate
  pipeline/run.py   # orquestador por etapas (idempotente)
  models/detector/  # train/predict YOLOv10
  models/score/     # features (visión+sensores por segmento) y score v0
  utils/io.py       # rutas, configs y parches de compatibilidad
models/             # pesos entrenados (gitignored)
reports/            # detections.csv, scores.csv, logs y reportes en markdown
tests/              # smoke tests: conversión VOC→YOLO, simulador, scorer
```

## Comandos

| Comando | Qué hace |
|---|---|
| `make setup` | Crea `.venv` (Python 3.11 vía uv) e instala dependencias |
| `make pipeline` | download → prepare → simulate → validate |
| `make download-data` / `prepare-data` / `simulate-sensors` / `validate-data` | Etapas sueltas |
| `make train-small` | Fine-tune YOLOv10n sobre el subset (3 épocas) |
| `make predict` | Inferencia sobre val → `reports/detections.csv` |
| `make score` | Score v0 por segmento → `reports/scores.csv` |
| `make test` | Smoke tests (pytest) |
| `make all` | Todo lo anterior en orden |

## Solución de problemas

- **Mendeley responde 403**: su API rechaza el user-agent de `urllib`;
  `download.py` ya envía uno de navegador. Si persiste, descargar manual a `data/raw/`.
- **`ModuleNotFoundError: huggingface_hub`**: el fork de yolov10 no la declara;
  está en `requirements.txt` — correr `make setup` de nuevo.
- **Errores `np.trapz` / `weights_only`**: ultralytics 8.1.34 (base del fork) es
  anterior a numpy 2 y torch 2.6; `apply_compat_patches()` en
  [utils/io.py](src/dimmit/utils/io.py) los corrige y ya se aplica en
  train/predict.
- **Ops no soportadas en MPS**: el Makefile exporta `PYTORCH_ENABLE_MPS_FALLBACK=1`
  para que caigan a CPU en vez de fallar.
- **Umbrales de confianza en 0.01**: son temporales para la iteración de humo
  (el detector solo entrena 3 épocas); subirlos a 0.25 cuando haya un modelo real.

## Roadmap

1. Entrenamiento real: 50+ épocas, más imágenes y los 3 países.
2. Umbrales de confianza de vuelta a 0.25 y métricas por clase decentes
   (balancear D40, que casi no aparece).
3. Sensores reales reemplazando la simulación (mismo esquema parquet).
4. `LearnedScorer` entrenado contra etiquetas reales de condición y comparado
   con el índice por reglas.
5. Export ONNX del detector si se requiere despliegue fuera de PyTorch.
