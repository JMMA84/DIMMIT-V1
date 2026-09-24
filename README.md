# DIMMIT V1

Detección de deterioro vial con visión por computador (YOLOv10) + datos de sensores,
combinados en un score de condición por segmento de vía. Primera iteración (v0) según
el Reporte Técnico V1.

## Quickstart

```bash
make setup      # venv (Python 3.11 vía uv) + dependencias
make pipeline   # descarga RDD2020, prepara subset YOLO, simula sensores y valida
make train-small
make predict
make score
make test
```

`make all` corre pipeline + entrenamiento + inferencia + score de una vez.

## Datos

- **Externos**: [RDD2020](https://data.mendeley.com/datasets/5ty2wb6gvg/1) — ~26k imágenes
  de vías (India/Japón/Chequia) con 4 clases de daño: D00 grieta longitudinal, D10
  transversal, D20 cocodrilo, D40 bache. Anotaciones Pascal VOC que el pipeline convierte
  a formato YOLO. La primera iteración usa un subset de ~300 imágenes de Japón
  (configurable en `configs/data_rdd2020.yaml`). Si la descarga automática falla, usar
  "Download All" en la página de Mendeley y dejar los archivos en `data/raw/`.
- **Internos (simulados)**: `dimmit/data/simulate_sensors.py` genera series dummy de los
  sensores del reporte técnico (acelerómetro triaxial, giroscopio, GPS a 50 Hz) por
  segmento de vía, con vibración correlacionada al daño anotado.

## Decisión: PyTorch, no TensorFlow

YOLOv10 se integra como dependencia pip del fork oficial
[THU-MIG/yolov10](https://github.com/THU-MIG/yolov10) (API estilo ultralytics, PyTorch).
No existe port oficial a TensorFlow y reescribir la arquitectura (NMS-free, v10 heads)
no se justifica para esta fase; si más adelante se requiere TF/TFLite, la ruta
recomendada es exportar el modelo entrenado vía ONNX.

## Estructura

```
configs/            # dataset, hiperparámetros del entrenamiento corto, pesos del score
data/               # raw -> interim -> processed (YOLO) + simulated (gitignored)
src/dimmit/
  data/             # download, prepare (VOC->YOLO), simulate_sensors, validate
  pipeline/run.py   # orquestador por etapas (idempotente)
  models/detector/  # train/predict YOLOv10
  models/score/     # features (visión+sensores por segmento) y score v0
models/             # pesos entrenados (gitignored)
reports/            # detections.csv, scores.csv
tests/              # smoke tests del pipeline
```

## Modelo de scores v0

`score = 100 − Σ_clase (w_c · severidad_c) − w_sensor · rugosidad_norm`, por segmento:
severidad visual = nº detecciones × (1 + 10 × área relativa media) por clase; rugosidad =
RMS de la aceleración vertical del segmento normalizada. Pesos y umbrales en
`configs/score.yaml`. La interfaz (`RuleBasedScorer` / `LearnedScorer`) deja lista la
migración a un modelo entrenado (RandomForest/XGBoost) cuando existan etiquetas reales
de condición.
