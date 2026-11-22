"""Graba en video (MP4) el proceso completo de campo: scripts/proceso_completo.sh corre en una terminal dentro de una
pantalla virtual (PyVirtualDisplay + Xvfb) y ffmpeg graba esa pantalla.

Uso: make campo-grabar   (o python -m dimmit.grabacion --tamano 1600x900 --fps 8)
Salida: reports/grabacion/dimmit_proceso_completo_<fecha>.mp4 y el registro de texto .log con la misma salida.
Requiere los programas Xvfb, xterm y ffmpeg; ImageMagick (display) para mostrar las figuras al final.
"""
import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path

from pyvirtualdisplay import Display

from dimmit.utils.io import REPO_ROOT, ensure_dir

OUT = REPO_ROOT / "reports" / "grabacion"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tamano", default="1600x900", help="resolución de la pantalla virtual y del video")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--fuente", type=int, default=13, help="tamaño de letra de la terminal")
    ap.add_argument("--script", default="scripts/proceso_completo.sh")
    ap.add_argument("--fuentes", default=os.environ.get("DIMMIT_FUENTES"), help="carpeta con Boriboon-*.otf para la animación (opcional)")
    args = ap.parse_args()
    for prog in ("Xvfb", "xterm", "ffmpeg"):
        if not shutil.which(prog):
            raise SystemExit(f"falta el programa {prog} (apt-get install xvfb xterm ffmpeg)")
    w, h = (int(v) for v in args.tamano.split("x"))
    stamp = time.strftime("%Y-%m-%d_%H%M")
    video = ensure_dir(OUT) / f"dimmit_proceso_completo_{stamp}.mp4"
    log = OUT / f"dimmit_proceso_completo_{stamp}.log"
    env = {**os.environ, "LC_ALL": "C.UTF-8", "LANG": "C.UTF-8", "DIMMIT_MOSTRAR_FIGURAS": "1", "DIMMIT_PANTALLA": args.tamano}
    if args.fuentes:
        env["DIMMIT_FUENTES"] = args.fuentes
    print(f"[grabación] pantalla virtual {w}x{h} a {args.fps} fps -> {video.relative_to(REPO_ROOT)}")
    with Display(visible=False, size=(w, h), color_depth=24) as disp:
        dpy = disp.new_display_var
        env["DISPLAY"] = dpy
        ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "x11grab", "-draw_mouse", "0", "-video_size", f"{w}x{h}",
                               "-framerate", str(args.fps), "-i", dpy, "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
                               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)], stdin=subprocess.PIPE)
        cols, rows = int(w / (args.fuente * 0.6)), int(h / (args.fuente * 1.18))
        cmd = f"bash {args.script} 2>&1 | tee {log}"
        term = subprocess.Popen(["xterm", "-display", dpy, "-geometry", f"{cols}x{rows}+0+0", "-fa", "DejaVu Sans Mono", "-fs", str(args.fuente),
                                 "-bg", "#0F0E47", "-fg", "#EDEDF7", "-cr", "#5AB7BB", "+sb", "-u8", "-T", "DIMMIT V1 · proceso completo",
                                 "-e", "bash", "-c", cmd], cwd=REPO_ROOT, env=env)
        t0 = time.time()
        while term.poll() is None:
            time.sleep(30)
            print(f"[grabación] grabando… {int(time.time() - t0) // 60} min", flush=True)
        time.sleep(1)
        ff.communicate(b"q", timeout=120)
    dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)],
                         capture_output=True, text=True).stdout.strip()
    print(f"[ok] {video.relative_to(REPO_ROOT)} · {float(dur or 0) / 60:.1f} min · {video.stat().st_size / 1e6:.1f} MB")
    print(f"[ok] {log.relative_to(REPO_ROOT)} (salida de texto del mismo proceso)")
    if "PROCESO COMPLETO TERMINADO" not in Path(log).read_text(errors="ignore"):
        raise SystemExit("el proceso no terminó bien: revisa el .log")


if __name__ == "__main__":
    main()
