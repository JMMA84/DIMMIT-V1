# DIMMIT V1

Estimación de la condición de la malla vial combinando **visión por computador** (YOLOv10 sobre
imágenes de la vía), **sensores del vehículo** (acelerómetro/giroscopio MPU-6050 vía Arduino y GPS
en Raspberry Pi, simulados mientras no existan) y **contexto abierto de Bogotá** (calzadas UMV,
colegios, salud, siniestralidad, clima). Una **red neuronal de fusión** entrega sub-scores por tipo
de daño, rugosidad, un **índice de condición vial (ICV) 0–100** y un **estado de 5 niveles**; un
índice de prioridad y una descripción en español (plantilla o Claude) completan cada segmento.

```
RDD2020 completo (26 336 img) ─► auditoría + splits sin fuga ─► YOLOv10n (Kaggle GPU / CPU)
                                                         │  detecciones + embeddings 448-d
calzadas reales UMV ─► simulador físico IMU/GPS ─────────┤
Bogotá datos abiertos + Open-Meteo ─► contexto ──► importancia (etiquetas UMV) ─► prioridad
                                                         ▼
              red de fusión (visión + embedding + IMU) ─► sub-scores, ICV, estado, intervalo 90 %
                                                         ▼
           descripciones (plantilla / Claude) ─► evaluación + deriva ─► CSV + reporte HTML
```

## Resultados

- `reports/output/segmentos_scores.csv`: una fila por segmento (test y despliegue): ranking de
  prioridad, sub-scores, ICV con intervalo, estado, contexto, prioridad, descripción y acción.
- `reports/evaluation/metricas_evaluacion.csv`: todas las métricas en formato largo con IC 95 %
  por bloques y estado OK/ALERTA/FALLA contra `configs/evaluation.yaml`.
- `reports/dimmit_reporte_v1.html`: tablero autocontenido (KPIs, salud del modelo, comparación con
  líneas base, matriz de confusión, deriva, mapa y tabla de segmentos).
- Extras: `observaciones_scores.csv`, `diccionario_datos.csv`, `matrices_confusion.csv`,
  `drift_psi.csv`, `historial_corridas.csv`, `perfil_dataset.csv`.
- Resultados comentados de cada corrida: `reports/resultados_*.md`.

## Datos

| Fuente | Uso | Licencia |
|---|---|---|
| RDD2020 (Mendeley 5ty2wb6gvg): 21 041 imágenes anotadas de Japón, India y Chequia + 5 295 sin anotar (test1/test2) | detector, objetivos visuales, lote de despliegue | CC BY-NC 3.0 |
| UMV *Modelo de Priorización de Vías 2020* (93 680 calzadas) | geometría de rutas, atributos, etiquetas de importancia | CC BY 4.0 |
| IDU Estado superficial, SED Colegios, SDS IPS, SDM Siniestros | contexto y prioridad | CC BY / BY-SA 4.0 |
| Open-Meteo (reanálisis y elevación) | lluvia 1/7/30/365 días, temperatura, pendiente | CC BY 4.0 |
| Zenodo 4386256 (MIT/UMass) | validación REAL del estimador de rugosidad con perfiles láser | CC0 |

Hallazgos de la auditoría (`data/index/audit.json`): los nombres de archivo de RDD2020 **no**
siguen el recorrido del vehículo, así que los segmentos son pseudo-rutas de fotogramas
visualmente vecinos (dHash de 256 bits), y hay 59 clústeres de casi-duplicados que se fuerzan a un
solo split. El split (`data/index/splits.csv`, congelado en git) es 65 % detector / 20 % red de
fusión (F) / 15 % test (T), agrupado por bloques y estratificado por país y condición.

## Objetivos (verdad de referencia)

- **Pseudo-PCI** (ASTM D6433 simplificado) desde las cajas anotadas por humanos: densidad por
  clase sobre ~143 m² de vía visible, severidad por tamaño aparente, valores deducidos y CDV
  (`configs/labels.yaml`). Da los sub-scores por daño y el índice visual.
- **IRI** real (Golden Car) del perfil generado por el simulador → índice de rodadura.
- **ICV = 0.65 · visual + 0.35 · rodadura**; estados: bueno ≥ 86, satisfactorio ≥ 71,
  regular ≥ 56, malo ≥ 41, muy malo < 41.

## Reglas de honestidad de la evaluación

1. El titular es la parte visual contra anotaciones humanas.
2. Lo que depende de sensores está **condicionado al simulador**: la rugosidad latente se acopla solo
   parcialmente al daño visual (r ≈ −0.65), la etiqueta es el IRI realizado y la verdad latente vive
   en `sensors_truth.parquet`, que ninguna feature lee (hay una prueba que lo verifica).
3. El contexto de Bogotá no entra al modelo de condición (las imágenes no son de esas calles); un
   canario lo comprueba. El contexto alimenta la importancia, la prioridad y las descripciones.
4. Controles: mundo nulo (sensores sin relación con el daño), etiquetas permutadas, oráculo y
   deriva con control negativo (test1/test2) y positivos (solo Chequia, imágenes degradadas).

## Modelos

- **Detector**: YOLOv10n con `ultralytics==8.3.253` (reemplaza al fork THU-MIG; sin parches). En
  Kaggle: 640 px, hasta 120 épocas (`configs/train_kaggle.yaml`); respaldo en CPU reanudable.
  Exportable a ONNX/NCNN para la Raspberry Pi.
- **Red de fusión** (`models/score/fusion.py`): codificadores por modalidad con *modality dropout*
  (funciona aunque falten sensores), cabezas de densidad por daño y ln IRI que, con las fórmulas del
  pseudo-PCI, dan sub-scores y ICV interpretables; cabeza ordinal CORN; ensamble 5 folds × 2
  semillas; intervalo conformal 90 % y P(estado).
- **Líneas base**: B0 score v0 por reglas, B1 regla física calibrada, B2 gradient boosting.
- **Importancia** (`geo/importance.py`): red neuronal vs gradient boosting que predice las
  dimensiones de priorización UMV desde datos abiertos, con validación cruzada espacial.

## Comandos

Ver `docs/plan_v1.md` para el orden completo. Lo esencial:

```bash
make setup                 # venv con torch CPU
make data                  # RDD2020 completo, auditoría, objetivos, listas
make kaggle-bundle kaggle-train && make kaggle-pull   # detector en Kaggle
make v1 DET=… WEIGHTS=…    # features → sensores → contexto → fusión → evaluación → HTML
make test validate-data
```

Variables de entorno: `~/.kaggle/access_token` (Kaggle) y, opcionalmente, `ANTHROPIC_API_KEY`
para las descripciones con Claude (`make describe LLM=1`).

## Estructura

```
configs/               datos, entrenamiento (Kaggle/CPU), etiquetas, sensores, contexto, fusión, LLM, evaluación
src/dimmit/data/       descarga, preparación, auditoría, splits, simulador de sensores, validación
src/dimmit/labels/     pseudo-PCI, rodadura, ICV y estados
src/dimmit/sensors/    perfil ISO 8608, IRI Golden Car, vehículo, features IMU, validación Zenodo
src/dimmit/geo/        fuentes abiertas, geohash, contexto, importancia y prioridad
src/dimmit/models/     detector (train/predict/evaluate) y score (visión, fusión, líneas base)
src/dimmit/llm/        descripciones: plantilla y Claude (Message Batches + validación)
src/dimmit/evaluation/ métricas, deriva y ejecución de la evaluación
src/dimmit/reporting/  CSV entregables y reporte HTML
src/dimmit/cloud/      orquestación del kernel de Kaggle
tests/                 pruebas rápidas sin red
```
