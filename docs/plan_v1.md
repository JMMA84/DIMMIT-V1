# DIMMIT v1: guía de ejecución

Todo corre en la nube: este contenedor (CPU), Kaggle (GPU para el detector) y, cuando exista
la clave, la API de Claude para las descripciones.

## Orden de ejecución

| Paso | Comando | Qué produce |
|---|---|---|
| 1 | `make setup` | entorno `.venv` con torch CPU y dependencias |
| 2 | `make data` | RDD2020 completo, índice, auditoría, pseudo-PCI y listas por split (`splits.csv` está congelado en git) |
| 3 | `make kaggle-bundle kaggle-train` | entrena YOLOv10n en Kaggle, evalúa en test e infiere todas las imágenes |
| 4 | `make kaggle-status` / `make kaggle-pull` | estado del kernel y descarga de salidas a `models/kaggle/` |
| 5 | `make predict-all WEIGHTS=… DET=…` | (solo si se entrena en CPU) detecciones + embeddings locales |
| 6 | `make v1 DET=… WEIGHTS=… IMGSZ=…` | features, sensores, contexto, red de fusión, líneas base, deriva, salidas, evaluación y HTML |
| 7 | `make importance zenodo` | modelo de importancia (UMV) y validación real del estimador de rugosidad |
| 8 | `make test validate-data` | pruebas y chequeos de calidad |

## Detector en Kaggle

- Credencial: `~/.kaggle/access_token` (nunca en el repo). Usuario de Kaggle: el de la cuenta.
- El kernel no necesita internet: el código, los wheels de ultralytics y los pesos base van en el
  dataset privado `dimmit-bundle`, y las imágenes en el dataset privado `rdd2020-raw`.
- **GPU**: Kaggle solo asigna GPU (y internet) a cuentas con el teléfono verificado
  (kaggle.com → Settings → Phone verification). Sin verificación el mismo kernel corre en CPU
  con `configs/train_cpu.yaml` adaptado (416 px, ~12 épocas, tope de 560 min).
- Al terminar: `make kaggle-pull`, luego
  `make v1 DET=models/kaggle/dimmit-yolov10-rdd2020/outputs WEIGHTS=models/kaggle/dimmit-yolov10-rdd2020/outputs/best.pt IMGSZ=<640 GPU | 416 CPU>`
  y copiar `best.pt` a `models/release/yolov10n_rdd2020.pt`.

## Activar Claude para las descripciones

1. Agregar `ANTHROPIC_API_KEY` como variable de entorno del entorno de nube y abrir una sesión nueva.
2. `make describe LLM=1 && make evaluate report` (modelo por defecto `claude-opus-5-5`,
   configurable en `configs/llm.yaml`; Message Batches con 50 % de descuento).
3. Cada texto se valida: números presentes en los hechos, estado correcto y acción permitida por la
   tabla de reglas. Lo que no pase queda con la plantilla (`fuente_descripcion`).

## Cuando lleguen los sensores reales

El simulador produce el esquema definitivo de `data/simulated/sensors.parquet`
(`segment_id, obs_id, timestamp, ax, ay, az, gx, gy, gz, lat, lon, speed_kmh` a 50 Hz) y
`sensor_frames.parquet` (hora y GPS de captura de cada imagen). Basta con escribir los datos del
Arduino/Raspberry Pi en ese esquema; las features de IMU no dependen de la orientación del
sensor. La verdad latente (`sensors_truth.parquet`) deja de existir: los objetivos de rugosidad
pasarán a ser mediciones de IRI de referencia (perfilómetro o IDU).

## Datos de campo (toma del 5-oct-2026)

1. `make campo-ingest`: lista la carpeta pública de Drive (`configs/campo.yaml: drive_folder_id`),
   descarga fotos, GPX, KML y lecturas del ultrasónico a `data/campo/raw/`, escribe el manifiesto con
   sha1 y arma `data/campo/index/{fotogramas,tramos}.parquet` (GPS interpolado por timestamp, tramos de
   `tramo_m` metros, `id_via` determinista).
2. `make kaggle-bundle` (el bundle ahora incluye `models/release/yolov10n_rdd2020.pt`) y
   `make campo-dataset` (dataset privado `dimmit-campo`).
3. `make campo-infer`: kernel `dimmit-campo-inferencia` (mismo `kaggle_kernel.py` con
   `DIMMIT_MODE=infer`); `make campo-status`; `make campo-pull` deja `detections.parquet`, `latency.json`
   y `resumen.json` en `reports/campo/kaggle/outputs/`. `make campo-local` corre lo mismo en CPU para la
   paridad.
4. `make campo`: `campo.scores` (features acotadas de YOLO, pseudo-PCI, etiqueta, profundidad alineada,
   score secundario de la red de fusión en modo solo visión), `campo.calidad` (calidad de imagen,
   confianza, PSI y AUROC de dominio frente a F ∪ T, correlación sensor ↔ YOLO con IC bootstrap),
   `campo.salidas` (CSV/JSON) y `reporting.presentacion_campo` (HTML por diapositivas).
5. Cuando lleguen nuevas tomas: misma estructura de carpetas en el Drive (`grabacion_*_Prueba_N/` con
   fotos `captura_YYYYMMDD_HHMMSS_mmm_n.jpg` y un GPX; `tomadedatos/*Profundidad*.txt`), actualizar el
   mapa `ultrasonico.mapa` en `configs/campo.yaml` y repetir 1–4.

Reportes publicados (Artifacts privados): evaluación v1 https://claude.ai/artifact/HbLoM9aeC2N5putgBeHpn2 ·
presentación de campo https://claude.ai/artifact/Qqo4QtPrQ9g95c3sLPPN8p — para actualizarlos desde otra
sesión, publicar con ese url.
