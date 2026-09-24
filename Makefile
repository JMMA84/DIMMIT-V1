PY := .venv/bin/python
ENV := PYTHONPATH=src PYTORCH_ENABLE_MPS_FALLBACK=1

.PHONY: setup download-data prepare-data simulate-sensors validate-data pipeline \
        train-small predict score test all

setup:
	uv venv --python 3.11 --seed .venv
	.venv/bin/pip install -r requirements.txt

download-data:
	$(ENV) python3 src/dimmit/data/download.py

prepare-data:
	$(ENV) $(PY) -m dimmit.data.prepare

simulate-sensors:
	$(ENV) $(PY) -m dimmit.data.simulate_sensors

validate-data:
	$(ENV) $(PY) -m dimmit.data.validate

pipeline:
	$(ENV) $(PY) -m dimmit.pipeline.run --stages download,prepare,simulate,validate

train-small:
	$(ENV) $(PY) -m dimmit.models.detector.train --config configs/train_small.yaml

predict:
	$(ENV) $(PY) -m dimmit.models.detector.predict

score:
	$(ENV) $(PY) -m dimmit.models.score.model

test:
	$(ENV) $(PY) -m pytest -q

all: pipeline train-small predict score
