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
