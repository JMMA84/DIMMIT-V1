PY := .venv/bin/python
ENV := PYTHONPATH=src PYTORCH_ENABLE_MPS_FALLBACK=1 OMP_NUM_THREADS=2
DET ?= data/features/detector
WEIGHTS ?= models/release/yolov10n_rdd2020.pt
IMGSZ ?= 640

.PHONY: setup data labels validate-data train-cpu kaggle-bundle kaggle-train kaggle-status kaggle-pull \
        predict-all features sensors context importance fusion baselines zenodo drift describe \
        evaluate report v1 test campo-ingest campo-dataset campo-infer campo-status campo-pull campo-local campo-scores campo-report campo

setup:
	uv venv --python 3.11 --seed .venv
	.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
	.venv/bin/pip install -r requirements.txt

# --- datos -------------------------------------------------------------------------------
data:            ## RDD2020 completo: descarga, extracción, índice, auditoría, objetivos, listas
	$(ENV) python3 src/dimmit/data/download.py --all
	$(ENV) $(PY) -m dimmit.data.prepare --stages extract,index
	$(ENV) $(PY) -m dimmit.data.audit
	$(ENV) $(PY) -m dimmit.data.splits
	$(ENV) $(PY) -m dimmit.data.prepare --stages lists

validate-data:
	$(ENV) $(PY) -m dimmit.data.validate

# --- detector (GPU en Kaggle; CPU como respaldo) -------------------------------------------
kaggle-bundle:
	$(ENV) $(PY) -m dimmit.cloud.kaggle_job bundle -m "$(or $(MSG),actualización)"
kaggle-train:
	$(ENV) $(PY) -m dimmit.cloud.kaggle_job push
kaggle-status:
	$(ENV) $(PY) -m dimmit.cloud.kaggle_job status
kaggle-pull:
	$(ENV) $(PY) -m dimmit.cloud.kaggle_job pull
train-cpu:
	$(ENV) $(PY) -m dimmit.models.detector.train --config configs/train_cpu.yaml
predict-all:     ## detecciones + embeddings de todas las imágenes con los pesos WEIGHTS
	$(ENV) $(PY) -m dimmit.models.detector.predict --weights $(WEIGHTS) --list data/processed/all.txt --out $(DET) --imgsz $(IMGSZ)

# --- features, sensores, contexto ----------------------------------------------------------
features:
	$(ENV) $(PY) -m dimmit.models.score.vision --det $(DET)
sensors:
	$(ENV) $(PY) -m dimmit.data.simulate_sensors
	$(ENV) $(PY) -m dimmit.sensors.imu
	$(ENV) $(PY) -m dimmit.data.simulate_sensors --null-world
	$(ENV) $(PY) -m dimmit.sensors.imu --null-world
context:
	$(ENV) $(PY) -m dimmit.geo.sources
	$(ENV) $(PY) -m dimmit.geo.context
importance:
	$(ENV) $(PY) -m dimmit.geo.importance
zenodo:          ## validación del estimador de rugosidad con datos reales (Zenodo 4386256)
	$(ENV) $(PY) -m dimmit.sensors.zenodo

# --- modelos y evaluación ----------------------------------------------------------------
fusion:
	$(ENV) $(PY) -m dimmit.models.score.fusion
baselines:
	OMP_NUM_THREADS=1 PYTHONPATH=src $(PY) -m dimmit.models.score.baselines
drift:
	$(ENV) $(PY) -m dimmit.evaluation.drift --weights $(WEIGHTS) --imgsz $(IMGSZ)
describe:        ## salidas por segmento (plantilla; con LLM=1 usa Claude si hay ANTHROPIC_API_KEY)
	$(ENV) $(PY) -m dimmit.reporting.outputs $(if $(LLM),--llm,)
evaluate:
	$(ENV) $(PY) -m dimmit.evaluation.run_eval
report:
	$(ENV) $(PY) -m dimmit.reporting.html_report

v1: features sensors context fusion baselines drift describe evaluate report

# --- datos de campo propios (Drive -> Kaggle -> salidas) ------------------------------------
campo-ingest:    ## descarga la carpeta de Drive y arma fotogramas/tramos con id_via
	$(ENV) $(PY) -m dimmit.campo.ingest
	$(ENV) $(PY) -m dimmit.campo.tramos
campo-dataset:   ## sube data/campo/raw como dataset privado de Kaggle (dimmit-campo)
	$(ENV) $(PY) -m dimmit.cloud.kaggle_job campo-dataset -m "$(or $(MSG),toma de campo)"
campo-infer:     ## kernel de solo inferencia en Kaggle (requiere kaggle-bundle y campo-dataset)
	$(ENV) $(PY) -m dimmit.cloud.kaggle_job infer-push
campo-status:
	$(ENV) $(PY) -m dimmit.cloud.kaggle_job infer-status
campo-pull:      ## descarga detecciones de Kaggle a reports/campo/kaggle/outputs
	$(ENV) $(PY) -m dimmit.cloud.kaggle_job infer-pull
campo-local:     ## detecciones en CPU local a 416, 640 y 832 px (paridad con Kaggle y consenso multi-escala)
	for s in 416 640 832; do $(ENV) $(PY) -m dimmit.models.detector.predict --weights $(WEIGHTS) --list data/campo/index/campo.txt --out data/campo/det_local_$$s --imgsz $$s --no-embeddings; done
campo-scores:    ## umbrales de campo, score, etiqueta, calidad, correlación y las dos salidas finales
	$(ENV) $(PY) -m dimmit.campo.umbrales
	$(ENV) $(PY) -m dimmit.campo.scores
	$(ENV) $(PY) -m dimmit.campo.calidad
	$(ENV) $(PY) -m dimmit.campo.salidas
campo-report:    ## presentación HTML (reports/campo/dimmit_presentacion.html)
	$(ENV) $(PY) -m dimmit.reporting.presentacion_campo
campo: campo-scores campo-report

test:
	$(ENV) $(PY) -m pytest -q
