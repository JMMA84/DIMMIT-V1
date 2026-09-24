"""Orquestador del pipeline de datos: download -> prepare -> simulate -> validate.

Cada etapa es idempotente (salta lo ya hecho). Uso:
    python -m dimmit.pipeline.run --stages download,prepare,simulate,validate
"""
import argparse
import subprocess
import sys

STAGES = {
    "download": "dimmit.data.download",
    "prepare": "dimmit.data.prepare",
    "simulate": "dimmit.data.simulate_sensors",
    "validate": "dimmit.data.validate",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stages", default=",".join(STAGES))
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
