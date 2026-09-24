# Resultados — Primera iteración del pipeline (23 sep 2025)

Corrida de humo de punta a punta: descarga de RDD2020, preparación del subset,
simulación de sensores, validación, entrenamiento corto de YOLOv10n, inferencia y
score v0. **Objetivo: validar que toda la tubería funciona**, no producir un modelo
útil todavía.

## Entorno

| Componente | Valor |
|---|---|
| Hardware | Apple M4 Pro (device `mps`) |
| Python | 3.11.14 (venv creado con uv) |
| PyTorch | 2.14.0 |
| YOLOv10 | fork THU-MIG (base ultralytics 8.1.34), instalado vía pip desde GitHub |

## Datos

- **RDD2020** (`train.tar.gz`, 1.4 GB) descargado de Mendeley vía API pública.
- Japón tiene **7 900 imágenes con anotaciones válidas** en las 4 clases objetivo;
  se muestrearon **300** (seed 42) → **240 train / 60 val** en formato YOLO.
- **Sensores simulados**: 30 segmentos de vía (10 imágenes por segmento),
  **15 000 muestras** (10 s a 50 Hz por segmento) de acelerómetro, giroscopio y GPS.
- **Validación del pipeline**: todos los checks en verde (pareo imagen↔label 1:1,
  bboxes normalizados, clases 0–3, sensores sin nulos, 300/300 imágenes mapeadas
  a segmento).

## Entrenamiento (YOLOv10n, 3 épocas, imgsz 640, batch 8)

Duración: ~32 s en MPS. Pesos: `models/yolov10n_small/weights/best.pt` (5.8 MB,
2.7 M parámetros, 8.2 GFLOPs).

| Clase | Instancias (val) | Precisión | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| **todas** | 120 | 0.003 | 0.227 | **0.015** | 0.005 |
| D00 grieta longitudinal | 38 | 0.003 | 0.132 | 0.003 | 0.001 |
| D10 grieta transversal | 35 | 0.000 | 0.000 | 0.000 | 0.000 |
| D20 grieta cocodrilo | 45 | 0.007 | 0.778 | 0.058 | 0.018 |
| D40 bache | 2 | 0.000 | 0.000 | 0.000 | 0.000 |

Con 240 imágenes y 3 épocas estos números son los esperados para una prueba de
humo: el modelo apenas empieza a mover pesos (D20, la clase con más área por caja,
es la única con algo de señal).

## Inferencia y score

- **Detecciones**: 46 en las 60 imágenes de val con umbral 0.01 (todas D20,
  confianza máxima 0.026). Con el umbral estándar 0.25 no hay ninguna, por eso los
  umbrales de la iteración de humo están en 0.01 (`configs/score.yaml` y
  `predict.py --conf`).
- **Scores** (30 segmentos): 27 buenos, 1 regular, 2 malos; rango 0–92.1,
  media 81.1 → `reports/scores.csv`.
- **Sanidad**: usando solo la señal de sensores, la correlación score vs severidad
  simulada es **−0.98** (el simulador y la featurización funcionan). Al incluir las
  detecciones del modelo de 3 épocas cae a ~0.02: las detecciones aún son ruido,
  lo cual es coherente con el mAP anterior.
- **Tests**: 3/3 en verde (`make test`).

## Problemas encontrados y resueltos

1. La API de Mendeley devuelve **403** al user-agent de `urllib` → header de
   navegador en `download.py`.
2. El fork de yolov10 **no declara `huggingface-hub`** → añadido a requirements.
3. ultralytics 8.1.34 es incompatible con **numpy 2** (`np.trapz` eliminado) y con
   **torch ≥ 2.6** (`weights_only=True` por defecto) → `apply_compat_patches()` en
   `src/dimmit/utils/io.py`.
4. Con cero detecciones, las features de severidad quedaban con dtype `object` →
   cast explícito a float en `features.py`.

## Próximos pasos

1. Entrenar en serio: ~50+ épocas con más imágenes (subir `n_images` en
   `configs/data_rdd2020.yaml`, épocas en `configs/train_small.yaml`), idealmente
   con los 3 países.
2. Devolver los umbrales de confianza a 0.25 cuando el detector sea real.
3. Balancear clases (D40 solo tuvo 2 instancias en val) o ponderar la pérdida.
4. Reemplazar sensores simulados por datos reales cuando existan, manteniendo el
   esquema de `sensors.parquet`.
5. Con etiquetas reales de condición de vía, activar `LearnedScorer` y comparar
   contra el índice basado en reglas.
