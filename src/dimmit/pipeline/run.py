"""Orquestador del pipeline v1 (cada etapa corre como subproceso y se detiene si una falla).

Uso:
    python -m dimmit.pipeline.run --stages features,sensors,imu,context,fusion,baselines,describe,evaluate,report
El detector se entrena en Kaggle (make kaggle-train) o en CPU (make train-cpu); las etapas con
argumentos (predict, drift) se corren desde el Makefile.
"""
import argparse
import subprocess
import sys

STAGES = {
    "download": "dimmit.data.download",
    "prepare": "dimmit.data.prepare",
    "audit": "dimmit.data.audit",
    "splits": "dimmit.data.splits",
    "features": "dimmit.models.score.vision",
    "sensors": "dimmit.data.simulate_sensors",
    "imu": "dimmit.sensors.imu",
    "context": "dimmit.geo.context",
    "importance": "dimmit.geo.importance",
    "zenodo": "dimmit.sensors.zenodo",
    "fusion": "dimmit.models.score.fusion",
    "baselines": "dimmit.models.score.baselines",
    "describe": "dimmit.reporting.outputs",
    "evaluate": "dimmit.evaluation.run_eval",
    "report": "dimmit.reporting.html_report",
    "validate": "dimmit.data.validate",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stages", default="features,sensors,imu,context,fusion,baselines,describe,evaluate,report")
    args = ap.parse_args()

    for stage in args.stages.split(","):
        stage = stage.strip()
        if stage not in STAGES:
            sys.exit(f"Etapa desconocida: {stage}. Opciones: {', '.join(STAGES)}")
        print(f"\n=== etapa: {stage} ===")
        result = subprocess.run([sys.executable, "-m", STAGES[stage]])
        if result.returncode != 0:
            sys.exit(f"La etapa '{stage}' falló (código {result.returncode})")
    print("\nPipeline completo ✓")


if __name__ == "__main__":
    main()
