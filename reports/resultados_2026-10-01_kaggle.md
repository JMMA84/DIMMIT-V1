# Resultados: corrida v1 con el detector entrenado en Kaggle (1 oct 2026)

Corrida `20261001-0859` (hora de Bogotá). Reemplaza al detector de humo de la corrida
`20260930-2219` (ver `resultados_2026-10-01.md`). Mismos splits, objetivos, sensores y semillas:
la diferencia entre ambas filas de `reports/evaluation/historial_corridas.csv` se debe al detector.

## Detector

YOLOv10n entrenado en un kernel de Kaggle **sin GPU** (la cuenta no tiene el teléfono verificado):
12 épocas a 416 px en 8.3 h, sobre las 9 420 imágenes del train del detector. Pesos en
`models/release/yolov10n_rdd2020.pt` (5.7 MB).

| Test | mAP50 | Precisión | Recall |
|---|---|---|---|
| Todos | **0.276** (ALERTA, objetivo 0.35) | 0.34 | 0.31 |
| Japón | 0.319 | 0.37 | 0.35 |
| India | 0.149 | 0.46 | 0.18 |
| Chequia | 0.198 | 0.46 | 0.15 |

AP50 por clase: grieta longitudinal 0.22, transversal 0.18, piel de cocodrilo 0.50, bache 0.20.
Presencia de daño por imagen: AUROC 0.85, F1 0.82. Latencia en CPU (lote 1, 416 px): p50 88 ms,
p95 115 ms. India y Chequia rinden peor que Japón; un entrenamiento en GPU a 640 px y más épocas
es el siguiente paso con mayor impacto.

## Antes y después

| Métrica (test) | Detector de humo | Detector Kaggle |
|---|---|---|
| mAP50 del detector | 0.033 | 0.276 |
| Spearman PCI visual (solo visión vs anotaciones) | 0.57 | **0.80** |
| MAE del ICV (puntos) | 9.04 | **6.45** [5.84–7.10] |
| Spearman del ICV | 0.90 | 0.92 |
| Kappa ponderada del estado | 0.81 | **0.86** |
| Exactitud ±1 nivel | 0.96 | 0.98 |
| Recall malo o muy malo | 0.58 | **0.76** (ALERTA, objetivo 0.80) |
| ECE del estado | 0.068 | 0.019 |
| Cobertura del intervalo 90 % | 0.88 | 0.88 (ancho ±12.9 puntos) |
| Métricas OK / ALERTA / FALLA | 17 / 4 / 8 | 21 / 4 / 4 |

Las 4 FALLA restantes son los controles positivos de deriva, que deben fallar (PSI 6.5 y 8.3;
AUROC de dominio ≈ 1.0). test1/test2 siguen estables (PSI máx 0.012 y 0.009).

## Red de fusión frente a líneas base y ablaciones (MAE del ICV en test)

| Modelo | MAE | Δ vs red (IC 95 %) |
|---|---|---|
| **Red de fusión** | **6.45** | — |
| B1 física calibrada | 10.07 | +3.6 [2.4–4.8] |
| B2 gradient boosting | 13.45 | +7.0 [5.8–8.1] |
| B0 score v0 | 60.15 | +53.7 |
| Solo visión | 8.95 | +2.5 [2.0–3.1] |
| Solo sensores | 12.31 | +5.9 [4.6–7.2] |
| Sin embedding | 6.52 | +0.07 [−0.18–0.39] |
| Con contexto (canario) | 6.44 | −0.02 [−0.18–0.13] |
| Oráculo (densidades anotadas) | 2.75 | −3.7 |

- La fusión de visión y sensores aporta sobre cada modalidad sola.
- El embedding de YOLOv10 no aporta sobre las features agregadas (IC incluye 0).
- El contexto de Bogotá tampoco (IC incluye 0), como se esperaba: se mantiene fuera del modelo de
  condición y alimenta la prioridad.
- Sub-scores (MAE): grieta longitudinal 3.7, transversal 1.6, piel de cocodrilo 5.4, baches 7.9.

## Sin cambios respecto a la corrida anterior

Sensores (R² log IRI 0.83 simulado; Spearman 0.94 con datos reales de Zenodo), importancia UMV
(Spearman 0.46), controles (canario R² −0.58, mundo nulo R² −0.51, etiquetas permutadas R² −0.36)
y fidelidad de las descripciones por plantilla (100 %).
