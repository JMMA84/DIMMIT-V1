#!/usr/bin/env bash
# DIMMIT V1 · proceso completo de campo, paso a paso y con avisos de progreso.
# Lo ejecuta `make campo-grabar` (src/dimmit/grabacion.py) dentro de una pantalla virtual para grabarlo en video;
# también se puede correr solo: bash scripts/proceso_completo.sh
#
# Pasos: entorno -> datos de Drive -> tramos -> referencia IDU -> inferencia en Kaggle -> detección local a 3 escalas
#        -> animación de cómo YOLO ve una foto (cuadrículas, vectores, caja) -> calibración contra el IDU, score, calidad y salidas -> pruebas -> figuras -> resumen
# Variables: DIMMIT_KAGGLE_CMD (lanzador del CLI de Kaggle), DIMMIT_MOSTRAR_FIGURAS=1 (abre figuras y animación en la pantalla),
#            DIMMIT_FUENTES (carpeta con la tipografía Boriboon de la plantilla, opcional)
set -o pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src PYTHONUNBUFFERED=1 PYTORCH_ENABLE_MPS_FALLBACK=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
PY=.venv/bin/python
KG=${DIMMIT_KAGGLE_CMD:-.venv/bin/kaggle}
TOTAL=11
T0=$(date +%s)
C="\033[1;36m"; G="\033[1;32m"; Y="\033[0;33m"; R="\033[1;31m"; W="\033[1;37m"; D="\033[2m"; N="\033[0m"
LINEA="━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# avisos del CLI de Kaggle que no aplican aquí (el token lo inyecta el proxy de la sesión) y advertencias de librerías
RUIDO='Authentication required|kaggle.com/account|Recommended: log in|No token to manage|kaggle auth login|rather not use OAuth|kaggle.com/settings/api|supply it to the CLI|Option [AB]:|export KAGGLE_API_TOKEN|Save the token|^First, you will need|Warning|warn\(|FutureWarning|UserWarning|^\s*$'

paso() {
  N_PASO=$1; TITULO=$2; DESC=$3; TP=$(date +%s)
  echo; echo -e "${C}${LINEA}${N}"
  echo -e "${C}  PASO ${N_PASO}/${TOTAL} · ${TITULO}${N}"
  echo -e "${W}  ${DESC}${N}"
  echo -e "${D}  inicio $(date '+%H:%M:%S') · transcurrido $(( ($(date +%s) - T0) / 60 )) min $(( ($(date +%s) - T0) % 60 )) s${N}"
  echo -e "${C}${LINEA}${N}"
}
listo() { echo -e "${G}  ✔ paso ${N_PASO}/${TOTAL} terminado en $(( $(date +%s) - TP )) s${N}"; }
correr() {
  echo -e "${Y}\$ $*${N}"
  "$@" 2>&1 | grep --line-buffered -v -E "$RUIDO"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then echo -e "${R}  ✘ falló con código $rc: $*${N}"; exit "$rc"; fi
}

clear
echo -e "${C}${LINEA}${N}"
echo -e "${W}  DIMMIT V1 · Dispositivo de Instancia y Modelado Vial para el Mantenimiento de la Infraestructura de Tránsito${N}"
echo -e "${W}  Grabación del proceso completo: datos de campo -> YOLOv10 (Kaggle) -> calibración con el IDU -> score -> etiqueta${N}"
echo -e "${D}  Universidad del Rosario · $(date '+%d-%m-%Y %H:%M:%S') · commit $(git rev-parse --short HEAD) · rama $(git branch --show-current)${N}"
echo -e "${C}${LINEA}${N}"
sleep 3

paso 1 "ENTORNO" "Máquina, versiones de las librerías y configuración de la corrida (configs/campo.yaml)"
correr uname -srm
echo "CPU: $(nproc) núcleos · RAM: $(free -g | awk '/Mem/{print $2}') GB"
correr $PY -c "import sys, torch, ultralytics, pandas, numpy, sklearn, shapely; print(f'Python {sys.version.split()[0]} · PyTorch {torch.__version__} · Ultralytics {ultralytics.__version__} · pandas {pandas.__version__} · NumPy {numpy.__version__} · scikit-learn {sklearn.__version__} · Shapely {shapely.__version__}')"
correr $PY - <<'EOF'
from dimmit.campo import load_cfg
c = load_cfg()
print(f"Toma evaluada: {c['toma']['titulo']}")
for p, s in c["pruebas"].items():
    print(f"  {p}: fotos={s['fotos']} · sitio {s.get('sitio')} · {s.get('superficie', 'asfalto')} · sensor={s['sensor']}")
print(f"Calibración con el IDU: {'sí' if c['calibracion']['usar'] else 'no'} · toma de calibración: {', '.join(c['calibracion']['pruebas'])}")
print(f"Sensor: descartar > {c['ultrasonico']['rango_max_cm']} cm y 0,00 · profundidad = distancia − {c['ultrasonico']['linea_base_cm']} cm")
print(f"Inferencia: {c['inferencia']['imgsz']} px · conf ≥ {c['inferencia']['conf']} · máx. {c['inferencia']['max_det']} cajas por foto")
EOF
listo

paso 2 "DATOS DE CAMPO (GOOGLE DRIVE)" "Lista la carpeta compartida, baja lo que falte y verifica cada archivo con su hash SHA-1"
correr $PY -m dimmit.campo.ingest
listo

paso 3 "FOTOGRAMAS Y TRAMOS" "Ordena las fotos, asigna ubicación (GPS o sitio declarado), corta tramos y genera los ID únicos"
correr $PY -m dimmit.campo.tramos
listo

paso 4 "REFERENCIA OFICIAL DEL IDU" "Descarga el estado superficial de las calzadas alrededor del CAI Rosario (Mapa de Referencia IDECA)"
correr $PY -m dimmit.campo.idu
listo

paso 5 "INFERENCIA EN LA NUBE (KAGGLE)" "Sube las fotos como dataset privado, corre YOLOv10 en un kernel de Kaggle y baja las detecciones"
correr $PY -m dimmit.cloud.kaggle_job campo-dataset -m "grabación del proceso completo $(date '+%Y-%m-%d %H:%M')"
echo "esperando a que Kaggle procese el dataset..."
for i in $(seq 1 40); do
  st=$($KG datasets status jeffbezotes/dimmit-campo 2>/dev/null | tail -1)
  echo "[kaggle] dataset: ${st:-pendiente} ($(( i * 15 )) s)"; [ "$st" = "ready" ] && break; sleep 15
done
correr $PY -m dimmit.cloud.kaggle_job infer-push
for i in $(seq 1 120); do
  st=$($PY -m dimmit.cloud.kaggle_job infer-status 2>/dev/null | grep -o 'KernelWorkerStatus\.[A-Z_]*' | tail -1)
  echo "[kaggle] kernel dimmit-campo-inferencia: ${st#KernelWorkerStatus.} · $(( i * 20 / 60 )) min $(( i * 20 % 60 )) s"
  case "$st" in *COMPLETE*) break ;; *ERROR*|*CANCEL*) echo -e "${R}  ✘ el kernel terminó con $st${N}"; exit 1 ;; esac
  sleep 20
done
rm -rf reports/campo/kaggle/outputs
correr $PY -m dimmit.cloud.kaggle_job infer-pull
correr cat reports/campo/kaggle/outputs/resumen.json
listo

paso 6 "DETECCIÓN LOCAL A 3 ESCALAS (CPU)" "416, 640 y 832 px: la de 640 comprueba la paridad con Kaggle; 416 y 832 alimentan el consenso multi-escala"
for s in 416 640 832; do
  correr $PY -m dimmit.models.detector.predict --weights models/release/yolov10n_rdd2020.pt --list data/campo/index/campo.txt \
    --out data/campo/det_local_$s --imgsz $s --conf 0.05 --max-det 30 --no-embeddings
done
correr $PY - <<'EOF'
import pandas as pd
k = pd.read_parquet("reports/campo/kaggle/outputs/detections.parquet").sort_values(["image_id", "class_id", "conf"]).reset_index(drop=True)
l = pd.read_parquet("data/campo/det_local_640/detections.parquet").sort_values(["image_id", "class_id", "conf"]).reset_index(drop=True)
d = (k["conf"] - l["conf"]).abs().max() if len(k) == len(l) else float("nan")
print(f"Paridad Kaggle ↔ local (640 px): {len(k)} vs {len(l)} cajas · diferencia máxima de confianza {d:.2e}")
EOF
listo

paso 7 "CÓMO VE YOLO UNA FOTO" "Animación con los números reales del modelo: tensor, mapas de rasgos, cuadrículas 80/40/20, los 68 números de cada celda, la caja y el umbral calibrado"
correr $PY -m dimmit.reporting.yolo_animacion --foto Prueba_6/Prueba_6_007 ${DIMMIT_FUENTES:+--fuentes "$DIMMIT_FUENTES"}
if [ "${DIMMIT_MOSTRAR_FIGURAS:-0}" = "1" ] && [ -n "$DISPLAY" ]; then
  echo "reproduciendo reports/grabacion/yolo_paso_a_paso.mp4 en pantalla completa…"
  ffplay -loglevel error -autoexit -an -noborder -left 0 -top 0 -x "${DIMMIT_PANTALLA%x*}" -y "${DIMMIT_PANTALLA#*x}" reports/grabacion/yolo_paso_a_paso.mp4
fi
listo

paso 8 "CALIBRACIÓN CON EL IDU, SCORE, CALIDAD Y SALIDAS" "Umbrales de campo -> barrido del multiplicador del umbral contra el IDU -> score -> etiqueta -> salidas 1 y 2"
correr $PY -m dimmit.campo.umbrales
correr $PY -m dimmit.campo.calibracion
correr $PY -m dimmit.campo.scores
correr $PY -m dimmit.campo.calidad
correr $PY -m dimmit.campo.salidas
listo

paso 9 "PRUEBAS AUTOMÁTICAS" "pytest: GPS, tramos, ID, sensor, filtros, superficie, etiquetas y escala del IDU"
correr $PY -m pytest -q tests
listo

paso 10 "FIGURAS DEL PÓSTER" "Resultados de la toma, diagrama de funcionamiento y validación frente al IDU"
correr $PY -m dimmit.reporting.poster_campo
correr $PY -m dimmit.reporting.diagrama_flujo
correr $PY -m dimmit.reporting.validacion_idu
if [ "${DIMMIT_MOSTRAR_FIGURAS:-0}" = "1" ] && [ -n "$DISPLAY" ]; then
  for f in reports/campo/poster/resultados_toma_*.png reports/campo/poster/validacion_idu.png reports/campo/poster/diagrama_flujo.png; do
    echo "mostrando $f"
    display -title "$f" -resize "${DIMMIT_PANTALLA:-1600x900}" -geometry +0+0 "$f" & pid=$!
    sleep 7; kill $pid 2>/dev/null; wait $pid 2>/dev/null
  done
fi
listo

paso 11 "RESUMEN" "Salida 1 (clasificación por vía) y calibración contra el IDU"
correr $PY - <<'EOF'
import json
import pandas as pd
v = pd.read_csv("reports/campo/vias_clasificacion.csv")
cols = ["nivel", "prueba", "score_calidad", "estado_idu_equivalente", "etiqueta", "profundidad_max_cm", "danos_detectados", "run_id"]
print(v[cols].round(1).to_string(index=False))
c = json.load(open("data/campo/features/calibracion_idu.json"))
print(f"\nCalibración: tau = {c['multiplicador']} · tau_campo -> {c['tau_campo_calibrado']}")
print(f"Tramos dentro del rango del IDU: {c['ajuste']['acierto_sin_calibrar']:.0%} sin calibrar -> {c['ajuste']['acierto_estado_idu']:.0%} calibrado "
      f"({c['n_tramos']} tramos, {c['n_calzadas']} calzada)")
EOF
echo
correr ls -la reports/campo/vias_clasificacion.csv reports/campo/calidad_confianza_correlacion.csv reports/campo/poster
listo

echo; echo -e "${G}${LINEA}${N}"
echo -e "${G}  PROCESO COMPLETO TERMINADO en $(( ($(date +%s) - T0) / 60 )) min $(( ($(date +%s) - T0) % 60 )) s · $(date '+%d-%m-%Y %H:%M:%S')${N}"
echo -e "${G}${LINEA}${N}"
sleep 6
