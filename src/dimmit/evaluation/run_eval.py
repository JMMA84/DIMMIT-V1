"""Ejecuta el marco de evaluación completo y escribe:

  reports/evaluation/metricas_evaluacion.csv   formato largo con IC 95 % y estado OK/ALERTA/FALLA
  reports/evaluation/matrices_confusion.csv    estados (red, líneas base) y detector
  reports/evaluation/drift_psi.csv             PSI por variable y lote (+ controles)
  reports/evaluation/historial_corridas.csv    una fila por corrida (seguimiento en el tiempo)

Áreas: detector (YOLOv10), titular de visión (vs anotaciones humanas), red de fusión vs líneas
base y ablaciones, controles (canario de contexto, mundo nulo, etiquetas permutadas, oráculo),
sensores, importancia (UMV), capa de lenguaje y deriva.
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

from dimmit.evaluation import metrics as M
from dimmit.labels import pseudo_pci as pp
from dimmit.models.score.dataset import build_tables, segment_truth
from dimmit.utils.io import CLASS_KEYS, EVAL, FEATURES, INDEX, OUTPUT, REPO_ROOT, ensure_dir, load_yaml

PRED = FEATURES / "predicciones"


class Collector:
    def __init__(self, run_id, cfg):
        self.rows, self.run_id, self.th = [], run_id, cfg["umbrales"]

    def add(self, etapa, modelo, metrica, valor, conjunto="test", nivel="segmento", subgrupo="todos", objetivo="", ci=(np.nan, np.nan), n=None, n_grupos=None, umbral=None, notas=""):
        th = self.th.get(umbral) if umbral else None
        ok, warn, d = (th["ok"], th["alerta"], th["direccion"]) if th else (None, None, "")
        estado = M.status(valor, ok, warn, d) if th else "INFO"
        self.rows.append(
            {
                "run_id": self.run_id, "fecha": pd.Timestamp.now(tz="America/Bogota").strftime("%Y-%m-%d %H:%M"), "etapa": etapa, "modelo": modelo,
                "nivel": nivel, "conjunto": conjunto, "subgrupo": subgrupo, "objetivo": objetivo, "metrica": metrica,
                "valor": None if valor is None else round(float(valor), 4), "ic95_inf": round(ci[0], 4) if np.isfinite(ci[0]) else None,
                "ic95_sup": round(ci[1], 4) if np.isfinite(ci[1]) else None, "n": n, "n_grupos": n_grupos, "direccion": d,
                "umbral_ok": str(ok) if ok is not None else None, "umbral_alerta": str(warn) if warn is not None else None, "estado": estado, "notas": notas,
            }
        )

    def frame(self):
        return pd.DataFrame(self.rows)


def eval_detector(col, frame, splits, boxes):
    det_dir = REPO_ROOT / (FEATURES / "detector_source.txt").read_text().strip()
    mfile = det_dir / "detector_metrics.json"
    modelo = det_dir.name
    if mfile.exists():
        dm = json.loads(mfile.read_text())
        for key, res in dm.items():
            sub = key.replace("test_", "") if key != "test" else "todos"
            col.add("detector", modelo, "map50", res["map50"], subgrupo=sub, nivel="caja", umbral="detector_map50" if sub == "todos" else None)
            col.add("detector", modelo, "map50_95", res["map50_95"], subgrupo=sub, nivel="caja")
            col.add("detector", modelo, "precision", res["precision"], subgrupo=sub, nivel="caja")
            col.add("detector", modelo, "recall", res["recall"], subgrupo=sub, nivel="caja")
            if sub == "todos":
                for c, r in res["por_clase"].items():
                    for k in ("precision", "recall", "f1", "ap50", "ap50_95"):
                        col.add("detector", modelo, k, r[k], subgrupo=c, nivel="caja", umbral="detector_ap50_D40" if (c == "D40" and k == "ap50") else None)
    lat = det_dir / "latency.json"
    if lat.exists():
        li = json.loads(lat.read_text())
        col.add("detector", modelo, "latencia_lote1_p95_ms", li["latencia_lote1_ms_p95"], nivel="imagen", umbral="latencia_p95_ms", notas=f"dispositivo {li['dispositivo']}, imgsz {li['imgsz']}")
        col.add("detector", modelo, "ms_por_imagen_lote", li["ms_por_imagen_lote"], nivel="imagen")
    # presencia de daño por imagen (test)
    taus = json.loads((FEATURES / "vision_thresholds.json").read_text())["tau"]
    t = frame[frame["split"] == "test"]
    truth = t["image_id"].isin(boxes["image_id"])
    score = t[[f"maxconf_{c}" for c in CLASS_KEYS]].max(axis=1)
    pred = np.zeros(len(t), bool)
    for c in CLASS_KEYS:
        pred |= (t[f"maxconf_{c}"] >= taus[c]).to_numpy()
    pres = M.detection_presence(truth.to_numpy(), score.to_numpy(), pred)
    for k, v in pres.items():
        col.add("detector", modelo, f"presencia_{k}", v, nivel="imagen", umbral={"auroc": "presencia_auroc", "f1": "presencia_f1"}.get(k), n=len(t))
    for c in CLASS_KEYS:
        tr = t["image_id"].isin(boxes.loc[boxes["cls"] == c, "image_id"]).to_numpy()
        if tr.any():
            pc = M.detection_presence(tr, t[f"maxconf_{c}"].to_numpy(), (t[f"maxconf_{c}"] >= taus[c]).to_numpy())
            for k in ("auroc", "precision", "recall", "f1"):
                col.add("detector", modelo, f"presencia_{k}", pc[k], nivel="imagen", subgrupo=c)
    return modelo


def eval_vision_headline(col, frame, truth, blocks):
    """Titular honesto: detector -> PCI visual contra anotaciones humanas (sin sensores)."""
    cfg = pp.load_cfg()
    t = frame[frame["split"] == "test"]
    rho = t.groupby("segment_id")[[f"rho_det_{c}" for c in CLASS_KEYS]].mean()
    rho.columns = [f"rho_{c}" for c in CLASS_KEYS]
    pci_det = pp.pci_from_rho(rho, cfg)["pci_vis"]
    y = truth.loc[pci_det.index, "pci_vis"]
    d = pd.DataFrame({"y": y, "p": pci_det, "g": blocks.loc[pci_det.index].to_numpy()})
    r = M.regression(d["y"], d["p"])
    ci = M.block_bootstrap(d, d["g"], lambda x: M.regression(x["y"], x["p"])["spearman"], n=300)
    col.add("vision", "PCI_det (reglas sobre detecciones)", "spearman", r["spearman"], objetivo="pci_visual", ci=ci, n=len(d), n_grupos=d["g"].nunique(), umbral="vision_pci_spearman", notas="titular: solo visión vs anotaciones humanas")
    col.add("vision", "PCI_det (reglas sobre detecciones)", "mae", r["mae"], objetivo="pci_visual", n=len(d))
    for c in CLASS_KEYS:
        rr = M.regression(t[f"rho_{c}"], t[f"rho_det_{c}"])
        col.add("vision", "densidad detectada", "spearman", rr["spearman"], nivel="fotograma", subgrupo=c, objetivo=f"densidad_{c}", notas="fidelidad de features")


def eval_fusion(col, truth, blocks, cfg_lab, conf_rows):
    names = sorted({p.name.split("__")[0] for p in PRED.glob("*__test_segmento.parquet")})
    main = pd.read_parquet(PRED / "principal__test_segmento.parquet")
    ids = main.index
    y = truth.loc[ids]
    g = blocks.loc[ids].to_numpy()

    def add_reg(name, pred, target, etapa="fusion"):
        d = pd.DataFrame({"y": y[target].to_numpy(), "p": pred.loc[ids].to_numpy(), "g": g})
        r = M.regression(d["y"], d["p"])
        for k in ("mae", "rmse", "r2", "spearman", "sesgo"):
            ci = M.block_bootstrap(d, d["g"], lambda x, k=k: M.regression(x["y"], x["p"])[k], n=300) if k in ("mae", "spearman") else (np.nan, np.nan)
            um = {("icv", "mae"): "icv_mae", ("icv", "spearman"): "icv_spearman"}.get((target, k)) if name == "principal" else None
            col.add(etapa, name, k, r[k], objetivo=target, ci=ci, n=len(d), n_grupos=len(np.unique(g)), umbral=um)
        return r

    # red principal: todos los objetivos
    for target, col_p in [("icv", "icv"), ("pci_vis", "pci_vis"), ("ride", "ride"), ("iri", "iri")] + [(f"sub_{c}", f"sub_{c}") for c in CLASS_KEYS]:
        add_reg("principal", main[col_p], target)
    st_true = y["estado"].to_numpy()
    st_pred = pp.state_of(main["icv"], cfg_lab)
    o, per = M.ordinal(st_true, st_pred)
    um = {"kappa_ponderada_cuadratica": "estado_qwk", "exactitud_mas_menos_1": "estado_mas_menos_1", "recall_malo_o_peor": "estado_recall_malo_o_peor", "f1_macro": "estado_f1_macro"}
    d = pd.DataFrame({"t": st_true, "p": st_pred, "g": g})
    for k, v in o.items():
        ci = M.block_bootstrap(d, d["g"], lambda x, k=k: M.ordinal(x["t"], x["p"])[0][k], n=300) if k in um else (np.nan, np.nan)
        col.add("fusion", "principal", k, v, objetivo="estado", ci=ci, n=len(d), umbral=um.get(k))
    for s, r in per.items():
        for k in ("precision", "recall", "f1"):
            col.add("fusion", "principal", k, r[k], objetivo="estado", subgrupo=s, n=r["soporte"])
    probs = main[[f"p_{s}" for s in pp.STATE_ORDER]].to_numpy()
    col.add("fusion", "principal", "ece", M.ece(probs, st_true), objetivo="estado", umbral="estado_ece")
    cov, width = M.coverage(y["icv"], main["icv_p05"], main["icv_p95"])
    col.add("fusion", "principal", "cobertura_intervalo_90", cov, objetivo="icv", umbral="cobertura_90", notas=f"ancho medio {width:.1f} puntos")
    cm = M.confusion(st_true, st_pred)
    conf_rows.append(cm.stack().rename("n").reset_index().assign(modelo="principal"))
    # comparación pareada contra líneas base y ablaciones (delta MAE con IC por bloques)
    base_err = np.abs(main["icv"].loc[ids] - y["icv"]).to_numpy()
    for name in names:
        if name == "principal":
            continue
        p = pd.read_parquet(PRED / f"{name}__test_segmento.parquet")
        if "icv" not in p:
            continue
        p = p["icv"].reindex(ids)
        if p.isna().all():
            continue
        etapa = "linea_base" if name.startswith(("B", "C")) else "ablacion"
        r = add_reg(name, p, "icv", etapa=etapa)
        err = np.abs(p.to_numpy() - y["icv"].to_numpy())
        dd = pd.DataFrame({"d": err - base_err, "g": g}).dropna()
        ci = M.block_bootstrap(dd, dd["g"], lambda x: x["d"].mean(), n=300)
        col.add(etapa, name, "delta_mae_vs_principal", dd["d"].mean(), objetivo="icv", ci=ci, notas="positivo = la red principal es mejor")
        if name.startswith(("B", "C")):
            cmb = M.confusion(st_true, pp.state_of(p.fillna(50), cfg_lab))
            conf_rows.append(cmb.stack().rename("n").reset_index().assign(modelo=name))
        if name == "C1_canario_contexto":
            col.add("control", name, "r2", r["r2"], objetivo="icv", umbral="canario_contexto_r2", notas="contexto de Bogotá sobre imágenes de otro país: debe ser ~0")
        if name == "etiquetas_permutadas":
            col.add("control", name, "spearman", r["spearman"], objetivo="icv", umbral="permutadas_icv_spearman", notas="entrenada con etiquetas barajadas: debe ser ~0")
    # mundo nulo: sensores sin relación con el daño no deben predecir el PCI visual
    pn = PRED / "mundo_nulo_solo_sensores__test_segmento.parquet"
    if pn.exists():
        p = pd.read_parquet(pn)
        r = M.regression(y["pci_vis"], p["pci_vis"].reindex(ids))
        col.add("control", "mundo_nulo_solo_sensores", "r2", r["r2"], objetivo="pci_visual", umbral="mundo_nulo_pci_r2", notas="si fuera alto habría fuga desde las etiquetas al simulador")
        col.add("control", "mundo_nulo_solo_sensores", "spearman", r["spearman"], objetivo="pci_visual")


def eval_sensors(col, truth):
    main = pd.read_parquet(PRED / "principal__test_segmento.parquet")
    y = truth.loc[main.index, "iri"]
    r = M.regression(np.log(y), np.log(main["iri"]))
    col.add("sensores", "principal", "r2_log_iri", r["r2"], objetivo="iri", umbral="iri_r2", notas="condicionado al simulador")
    col.add("sensores", "principal", "spearman", r["spearman"], objetivo="iri", notas="condicionado al simulador")
    b1 = pd.read_parquet(PRED / "B1_fisico_calibrado__test_segmento.parquet")
    rb = M.regression(np.log(y), np.log(b1["iri"].reindex(main.index)))
    col.add("sensores", "B1 k*rms/sqrt(v)", "r2_log_iri", rb["r2"], objetivo="iri", notas="estimador físico simple")
    zen = FEATURES / "zenodo_validacion.json"
    if zen.exists():
        z = json.loads(zen.read_text())
        col.add("sensores", "estimador IRI (datos reales Zenodo 4386256)", "spearman", z["spearman"], conjunto="real_zenodo", objetivo="iri", umbral="iri_spearman_real", n=z["n"], notas=z.get("notas", ""))
        col.add("sensores", "estimador IRI (datos reales Zenodo 4386256)", "r2_log", z["r2"], conjunto="real_zenodo", objetivo="iri", n=z["n"], notas="k calibrado dejando una vía afuera")
        for via, r in z["por_via"].items():
            col.add("sensores", "estimador IRI (datos reales Zenodo 4386256)", "spearman", r["spearman_dentro_de_via"], conjunto="real_zenodo", subgrupo=via, objetivo="iri", n=r["n_tramos"], notas=f"dentro de la vía; IRI láser medio {r['iri_laser_medio']:.2f} m/km")


def eval_context(col):
    f = FEATURES / "importance_cv.json"
    if not f.exists():
        return
    d = json.loads(f.read_text())
    for target, models in d["cv_espacial_geohash5"].items():
        for name, r in models.items():
            best = d["mejor_modelo"][target] == name
            col.add("importancia", name, "spearman", r["spearman"], conjunto="umv_cv_espacial", nivel="calzada", objetivo=target, umbral="importancia_spearman" if (best and target == "ip_umv") else None)
            col.add("importancia", name, "r2", r["r2"], conjunto="umv_cv_espacial", nivel="calzada", objetivo=target)


def eval_llm(col):
    f = EVAL / "llm_stats.json"
    out = OUTPUT / "segmentos_scores.csv"
    if not out.exists():
        return
    seg = pd.read_csv(out)
    from dimmit.llm.describe import facts, faithfulness

    ok = []
    for _, r in seg.iterrows():
        row = r.to_dict()
        row["segment_id"] = row["segmento_id"]
        good, _ = faithfulness(str(r["descripcion"]), facts(row))
        ok.append(good)
    fuente = seg["fuente_descripcion"].value_counts().to_dict()
    col.add("lenguaje", "/".join(fuente), "fidelidad_numerica", float(np.mean(ok)), conjunto="test+despliegue", umbral="llm_fidelidad", n=len(seg), notas=f"fuentes: {fuente}")
    if f.exists():
        st = json.loads(f.read_text())
        if st.get("modo") == "claude":
            col.add("lenguaje", "claude", "json_valido", st["json_valido"] / max(st["exitosos"], 1), umbral="llm_json_valido", n=st["enviados"])
            col.add("lenguaje", "claude", "aceptadas_por_validador", st["fieles"] / max(st["enviados"], 1), n=st["enviados"])
        else:
            col.add("lenguaje", "plantilla", "modo", 0, notas=st.get("aviso", "sin clave de API: descripciones por plantilla"))


def eval_drift(col, frame, cfg):
    """PSI por variable: referencia F ∪ T vs lotes (control negativo y positivos)."""
    vars_ = [v for v in cfg["deriva"]["variables"] if v in frame.columns]
    ref = frame[frame["split"].isin(["fusion", "test"])]
    batches = {
        "despliegue_test1": frame[(frame["split"] == "despliegue") & frame["segment_id"].str.startswith("T1")],
        "despliegue_test2": frame[(frame["split"] == "despliegue") & frame["segment_id"].str.startswith("T2")],
        "control_positivo_solo_chequia": frame[(frame["split"] == "despliegue") & (frame["country"] == "Czech")],
    }
    ref_no_cz = ref[ref["country"] != "Czech"]
    deg = FEATURES / "degradado_frame.parquet"
    if deg.exists():
        batches["control_positivo_degradacion"] = pd.read_parquet(deg)
    rows = []
    emb = [c for c in frame.columns if c.startswith("emb_pc")][:32]
    for name, b in batches.items():
        r = ref_no_cz if name == "control_positivo_solo_chequia" else ref
        vals = {v: M.psi(r[v], b[v]) for v in vars_ if v in b}
        for v, p in vals.items():
            rows.append({"lote": name, "variable": v, "psi": round(p, 4), "estado": M.status(p, 0.10, 0.25, "menor")})
        mx = max([p for p in vals.values() if np.isfinite(p)], default=np.nan)
        col.add("deriva", "monitor PSI", "psi_maximo", mx, conjunto=name, nivel="fotograma", umbral="psi", n=len(b), notas=f"variable con mayor PSI: {max(vals, key=lambda k: vals[k] if np.isfinite(vals[k]) else -1)}")
        if emb and all(c in b for c in emb) and len(b) > 50:
            X = pd.concat([r[emb], b[emb]]).fillna(0).to_numpy()
            yy = np.r_[np.zeros(len(r)), np.ones(len(b))]
            p = cross_val_predict(LogisticRegression(max_iter=500, class_weight="balanced"), X, yy, cv=5, method="predict_proba")[:, 1]
            col.add("deriva", "clasificador de dominio (PCA embedding)", "auroc", roc_auc_score(yy, p), conjunto=name, nivel="fotograma", umbral="dominio_auroc", n=len(b))
    pd.DataFrame(rows).to_csv(EVAL / "drift_psi.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()
    cfg = load_yaml("configs/evaluation.yaml")
    cfg_lab = pp.load_cfg()
    seg_out = OUTPUT / "segmentos_scores.csv"
    run_id = args.run_id or (pd.read_csv(seg_out, usecols=["run_id"])["run_id"].iloc[0] if seg_out.exists() else "sin_run")
    col = Collector(run_id, cfg)
    frame = build_tables()
    splits = pd.read_csv(INDEX / "splits.csv")
    boxes = pd.read_parquet(INDEX / "boxes.parquet")
    truth = segment_truth(frame)
    blocks = splits.groupby("segment_id")["group_id"].first()
    conf_rows = []
    modelo_det = eval_detector(col, frame, splits, boxes)
    eval_vision_headline(col, frame, truth, blocks)
    eval_fusion(col, truth, blocks, cfg_lab, conf_rows)
    eval_sensors(col, truth)
    eval_context(col)
    eval_llm(col)
    eval_drift(col, frame, cfg)
    ensure_dir(EVAL)
    df = col.frame()
    df.to_csv(EVAL / "metricas_evaluacion.csv", index=False)
    pd.concat(conf_rows).to_csv(EVAL / "matrices_confusion.csv", index=False)
    # historial de corridas
    pick = lambda e, m, met, obj="", sub="todos": df.query("etapa==@e and modelo==@m and metrica==@met and objetivo==@obj and subgrupo==@sub")["valor"].squeeze() if len(df.query("etapa==@e and modelo==@m and metrica==@met and objetivo==@obj and subgrupo==@sub")) else None  # noqa: E731
    hist = {
        "run_id": run_id,
        "fecha": pd.Timestamp.now(tz="America/Bogota").strftime("%Y-%m-%d %H:%M"),
        "detector": modelo_det,
        "detector_map50": pick("detector", modelo_det, "map50"),
        "vision_pci_spearman": pick("vision", "PCI_det (reglas sobre detecciones)", "spearman", "pci_visual"),
        "icv_mae": pick("fusion", "principal", "mae", "icv"),
        "icv_spearman": pick("fusion", "principal", "spearman", "icv"),
        "estado_qwk": pick("fusion", "principal", "kappa_ponderada_cuadratica", "estado"),
        "estado_recall_malo_o_peor": pick("fusion", "principal", "recall_malo_o_peor", "estado"),
        "n_ok": int((df["estado"] == "OK").sum()),
        "n_alerta": int((df["estado"] == "ALERTA").sum()),
        "n_falla": int((df["estado"] == "FALLA").sum()),
    }
    hp = EVAL / "historial_corridas.csv"
    h = pd.read_csv(hp) if hp.exists() else pd.DataFrame()
    h = pd.concat([h[h.get("run_id", pd.Series(dtype=str)) != run_id] if len(h) else h, pd.DataFrame([hist])], ignore_index=True)
    h.to_csv(hp, index=False)
    print(df["estado"].value_counts().to_dict())
    print(df[df["umbral_ok"].notna()][["etapa", "modelo", "metrica", "objetivo", "conjunto", "valor", "estado"]].to_string(index=False))


if __name__ == "__main__":
    main()
