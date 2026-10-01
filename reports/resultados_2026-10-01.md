# Resultados: primera corrida de punta a punta v1 (1 oct 2026)

Corrida `20260930-2219` (hora de Bogotá). **Provisional**: el detector es el de humo entrenado en
CPU (YOLOv10n, 416 px, 2 épocas sobre 25 % del train). El entrenamiento completo corre en Kaggle;
cuando termine se repiten las etapas de visión en adelante (`docs/plan_v1.md`).

Reporte interactivo: `reports/dimmit_reporte_v1.html` (también publicado como Artifact privado).
Todas las cifras: `reports/evaluation/metricas_evaluacion.csv`, con IC 95 % por bootstrap de bloques.

## Datos

- RDD2020 completo: 21 041 imágenes anotadas (Japón 10 506, India 7 706, Chequia 2 829) y 5 295 de
  test1/test2 sin anotaciones. 25 045 cajas de las 4 clases objetivo; 12 195 imágenes con daño.
- La auditoría mostró que los nombres de archivo no siguen el recorrido del vehículo (distancia dHash
  entre consecutivos = entre aleatorios), así que los segmentos son pseudo-rutas de vecinos visuales.
  Hay 59 clústeres de casi-duplicados (155 imágenes) que no cruzan splits.
- Split: 13 741 train (detector) / 4 200 F (red de fusión) / 3 100 T (test) / 5 295 despliegue;
  1 375 / 420 / 310 / 531 segmentos. Cero fugas de bloque, segmento o duplicado.
- Estados de referencia en F ∪ T: muy malo 14.5 %, malo 27 %, regular 27 %, satisfactorio 17 %,
  bueno 14.5 %.

## Detector (provisional)

mAP50 test 0.033 (FALLA, esperado para el humo), latencia p95 29 ms en CPU a 416 px,
presencia de daño por imagen F1 0.74 / AUROC 0.67. El titular honesto de visión: Spearman entre
el PCI calculado sobre detecciones y el PCI de las anotaciones = 0.57 por segmento (ALERTA).

## Red de fusión (test, 310 segmentos)

| Modelo | MAE ICV | Δ vs red (IC 95 %) |
|---|---|---|
| **Red de fusión (principal)** | **9.04** [8.36–9.68] | — |
| B0 score v0 por reglas | 60.3 | +51.3 |
| B1 física calibrada (PCI_det + IRI = k·rms/√v) | 18.0 | +8.9 [7.5–10.4] |
| B2 gradient boosting | 13.9 | +4.8 [3.8–5.8] |
| Solo visión | 11.8 | +2.8 [1.8–3.5] |
| Solo sensores | 12.6 | +3.6 [2.5–4.7] |
| Sin embedding | 9.75 | +0.7 [0.3–1.1] |
| Con contexto (canario) | 8.73 | −0.3 [−0.6–0.0] |
| Oráculo (densidades anotadas) | 2.81 | −6.2 |
| Etiquetas permutadas | 19.0 | +10.0 |

- ICV: Spearman 0.90, R² 0.68. Estado: kappa ponderada 0.81, exactitud ±1 nivel 0.96,
  F1 macro 0.53, ECE 0.068 (ALERTA). Cobertura del intervalo conformal 90 %: 0.88 (OK).
- **Recall de vías malas o muy malas: 0.58 (FALLA)**, con precisión 0.98. Con el detector de humo la
  red subestima el daño; el oráculo muestra que casi todo el error viene del detector.
- Sub-scores (MAE): grieta longitudinal 4.5, transversal 1.8, piel de cocodrilo 8.7, baches 8.6.
- El modelo con contexto mejora 0.3 puntos con IC que apenas excluye 0 (una sola semilla); el
  contexto se asignó al azar, así que se interpreta como ruido y se mantiene fuera del modelo. El
  canario solo-contexto da R² = −0.58 (OK).

## Sensores

- Condicionado al simulador: R² del log IRI 0.83 (red), corr(PCI visual, IRI real) = −0.65.
- **Datos reales (Zenodo 4386256)**: el estimador rms/√v del pipeline ordena el IRI láser con
  Spearman 0.94 en 249 tramos de 200 m (0.71–0.84 dentro de cada vía); R² del log IRI 0.48 con k
  calibrado dejando una vía afuera.
- Mundo nulo: sensores sin relación con el daño no predicen el PCI visual (R² −0.51, OK).

## Contexto e importancia

Importancia (etiquetas reales UMV, validación cruzada espacial geohash-5): gradient boosting
supera a la red neuronal en las 4 dimensiones; Spearman con el IP oficial 0.46 (ALERTA). La
dimensión social es la más difícil (0.29): la UMV usa población y peticiones que no son abiertas.

## Lenguaje y deriva

- Descripciones por plantilla (sin clave de Claude): fidelidad numérica 100 % en 841 segmentos.
- Deriva: test1/test2 estables (PSI máx 0.012 y 0.015; AUROC de dominio 0.48). Los controles
  positivos alertan como deben: solo Chequia PSI 2.7 y lote degradado PSI 8.3 (AUROC 1.0).

## Próximos pasos

1. Verificar el teléfono en Kaggle para obtener GPU y reentrenar a 640 px (hoy el kernel corre en
   CPU de Kaggle a 416 px).
2. Repetir visión → fusión → evaluación con el detector final y comparar en `historial_corridas.csv`.
3. Agregar `ANTHROPIC_API_KEY` para las descripciones con Claude y el juez opcional.
4. Reemplazar el simulador por los datos del Arduino/Raspberry Pi (mismo esquema).
