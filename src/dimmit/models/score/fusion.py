"""Red neuronal de fusión (condición vial): visión (YOLOv10) + embeddings + IMU/GPS.

Arquitectura (PyTorch):
  - un codificador por grupo de entrada (visión, embedding, IMU) con "modality dropout": en
    entrenamiento se apaga un grupo completo con p=0.15 y se avisa con una bandera, así el mismo
    modelo funciona solo con visión (sensores caídos) o solo con sensores;
  - tronco 128 -> 128 con conexión residual;
  - cabezas: log1p(densidad) de las 4 clases de daño y ln(IRI) -> con las fórmulas del pseudo-PCI
    dan los sub-scores, el PCI visual, el índice de rodadura y el ICV (interpretables y agregables
    al segmento de forma exacta); cabeza auxiliar de ICV directo y cabeza ordinal CORN del estado.
El estado entregado es bucket(ICV), así nunca contradice al score.
Entrenamiento en el conjunto F con 5 folds agrupados por bloque; el ensamble son los modelos de
los folds x semillas. Incertidumbre: dispersión del ensamble + intervalo conformal (residuos
fuera de fold en F) y P(estado) desde esa distribución.
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from sklearn.model_selection import GroupKFold
from torch import nn

from dimmit.labels import pseudo_pci as pp
from dimmit.models.score.dataset import build_tables, feature_groups, finite, segment_truth
from dimmit.utils.io import CLASS_KEYS, DATA, FEATURES, REPO_ROOT, ensure_dir, load_yaml

N_STATES = len(pp.STATE_ORDER)


class Preprocessor:
    """log1p en columnas no negativas muy asimétricas, imputación por mediana y estandarización."""

    def fit(self, df, cols):
        self.cols = list(cols)
        x = df[self.cols].astype(float)
        self.log_cols = [c for c in self.cols if x[c].min() >= 0 and x[c].skew() > 2]
        x = self._log(x)
        self.med = x.median()
        self.mu = x.fillna(self.med).mean()
        self.sd = x.fillna(self.med).std().replace(0, 1).fillna(1)
        return self

    def _log(self, x):
        x = x.copy()
        for c in self.log_cols:
            x[c] = np.log1p(np.maximum(x[c], 0))
        return x

    def transform(self, df):
        x = self._log(df[self.cols].astype(float)).fillna(self.med)
        return np.clip(finite((x - self.mu) / self.sd), -8, 8)

    def state(self):
        return {"cols": self.cols, "log_cols": self.log_cols, "med": self.med.to_dict(), "mu": self.mu.to_dict(), "sd": self.sd.to_dict()}


class FusionNet(nn.Module):
    def __init__(self, group_dims: dict, hidden=None, p_modal=0.15):
        super().__init__()
        hidden = hidden or {"vision": 64, "embedding": 32, "imu": 32, "context": 16, "oracle": 16}
        self.groups = [g for g, d in group_dims.items() if d > 0]
        self.p_modal = p_modal
        self.enc = nn.ModuleDict(
            {
                g: nn.Sequential(nn.Linear(group_dims[g], hidden[g]), nn.GELU(), nn.LayerNorm(hidden[g]), nn.Dropout(0.3 if g == "embedding" else 0.1))
                for g in self.groups
            }
        )
        width = sum(hidden[g] for g in self.groups) + len(self.groups)
        self.inp = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Dropout(0.2))
        self.res = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 128))
        self.head_rho = nn.Linear(128, 4)
        self.head_iri = nn.Linear(128, 1)
        self.head_icv = nn.Linear(128, 1)
        self.head_corn = nn.Linear(128, N_STATES - 1)

    def forward(self, xs: dict, drop: dict | None = None):
        parts, flags = [], []
        n = next(iter(xs.values())).shape[0]
        for g in self.groups:
            h = self.enc[g](xs[g])
            keep = torch.ones(n, 1)
            if drop is not None and g in drop:
                keep = drop[g]
            elif self.training and len(self.groups) > 1:
                keep = (torch.rand(n, 1) > self.p_modal).float()
            parts.append(h * keep)
            flags.append(1 - keep)
        z = self.inp(torch.cat(parts + flags, dim=1))
        z = z + self.res(z)
        return {"rho": self.head_rho(z), "iri": self.head_iri(z).squeeze(1), "icv": self.head_icv(z).squeeze(1), "corn": self.head_corn(z)}


def corn_loss(logits, y):
    """CORN (ordinal condicional): tarea k = P(y > k | y > k-1)."""
    loss, n = 0.0, 0
    for k in range(logits.shape[1]):
        m = y > (k - 1) if k > 0 else torch.ones_like(y, dtype=torch.bool)
        if m.any():
            loss = loss + nn.functional.binary_cross_entropy_with_logits(logits[m, k], (y[m] > k).float(), reduction="sum")
            n += int(m.sum())
    return loss / max(n, 1)


def corn_probs(logits):
    cond = torch.sigmoid(logits)
    gt = torch.cumprod(cond, dim=1)  # P(y > k)
    ones = torch.ones(gt.shape[0], 1)
    upper = torch.cat([ones, gt], dim=1)
    lower = torch.cat([gt, torch.zeros(gt.shape[0], 1)], dim=1)
    return (upper - lower).clamp(min=0)


class Targets:
    """Estandarización de objetivos de regresión."""

    def fit(self, frame):
        rho = np.log1p(frame[[f"rho_{c}" for c in CLASS_KEYS]].to_numpy(float))
        iri = np.log(frame["iri_real"].to_numpy(float))
        self.rho_mu, self.rho_sd = rho.mean(0), rho.std(0) + 1e-6
        self.iri_mu, self.iri_sd = iri.mean(), iri.std() + 1e-6
        return self

    def tensors(self, frame):
        rho = (np.log1p(frame[[f"rho_{c}" for c in CLASS_KEYS]].to_numpy(float)) - self.rho_mu) / self.rho_sd
        iri = (np.log(frame["iri_real"].to_numpy(float)) - self.iri_mu) / self.iri_sd
        icv = frame["icv"].to_numpy(float) / 100
        st = pp.state_index(frame["estado"].to_numpy())
        return (torch.tensor(rho, dtype=torch.float32), torch.tensor(iri, dtype=torch.float32), torch.tensor(icv, dtype=torch.float32), torch.tensor(st))

    def decode(self, out):
        rho = np.expm1(out["rho"].detach().numpy() * self.rho_sd + self.rho_mu).clip(min=0)
        iri = np.exp(out["iri"].detach().numpy() * self.iri_sd + self.iri_mu)
        return rho, iri, out["icv"].detach().numpy() * 100

    def state(self):
        return {"rho_mu": self.rho_mu.tolist(), "rho_sd": self.rho_sd.tolist(), "iri_mu": float(self.iri_mu), "iri_sd": float(self.iri_sd)}


def derive_scores(rho, iri, segment_ids, cfg):
    """De densidades e IRI por fotograma a sub-scores/PCI/ICV por fotograma y por segmento."""
    rho_df = pd.DataFrame(rho, columns=[f"rho_{c}" for c in CLASS_KEYS])
    fr = pp.pci_from_rho(rho_df, cfg)
    fr["iri"] = iri
    fr["ride"] = pp.ride_index(iri, cfg)
    fr["icv"] = pp.icv(fr["pci_vis"], fr["ride"], cfg)
    fr["segment_id"] = np.asarray(segment_ids)
    g = fr.groupby("segment_id")
    seg = pp.pci_from_rho(g[[f"rho_{c}" for c in CLASS_KEYS]].mean(), cfg)
    seg["iri"] = g["iri"].mean()
    seg["ride"] = pp.ride_index(seg["iri"], cfg)
    seg["icv"] = pp.icv(seg["pci_vis"], seg["ride"], cfg)
    return fr, seg


def train_one(xs_tr, y_tr, xs_va, y_va, seg_va, tgt, dims, cfg_f, cfg_lab, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = FusionNet(dims, p_modal=cfg_f["modality_dropout"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg_f["lr"], weight_decay=cfg_f["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)
    w = cfg_f["loss_weights"]
    rho_w = torch.tensor([1.0, 1.0, 1.0, w["d40"]])
    huber = nn.HuberLoss(reduction="none")
    n = y_tr[0].shape[0]
    best, best_state, bad = np.inf, None, 0
    seg_true = seg_va["icv"]
    for epoch in range(cfg_f["max_epochs"]):
        model.train()
        perm = torch.randperm(n)
        for s in range(0, n, cfg_f["batch"]):
            idx = perm[s : s + cfg_f["batch"]]
            out = model({g: x[idx] for g, x in xs_tr.items()})
            rho, iri, icv, st = (t[idx] for t in y_tr)
            loss = (
                w["rho"] * (huber(out["rho"], rho) * rho_w).mean()
                + w["iri"] * huber(out["iri"], iri).mean()
                + w["icv"] * huber(out["icv"] * 10, icv * 10).mean()
                + w["corn"] * corn_loss(out["corn"], st)
            )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            r, i, _ = tgt.decode(model(xs_va))
        _, seg = derive_scores(r, i, seg_va.attrs["frame_segments"], cfg_lab)
        mae = float(np.abs(seg["icv"].reindex(seg_true.index) - seg_true).mean())
        sched.step(mae)
        if mae < best - 1e-3:
            best, bad = mae, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg_f["patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best, epoch + 1


def to_tensors(pres, frame):
    return {g: torch.tensor(p.transform(frame), dtype=torch.float32) for g, p in pres.items()}


def fit_ensemble(frame_f, groups, cfg_f, cfg_lab, seeds, tag):
    """CV agrupada por bloque en F: devuelve modelos, predicciones fuera de fold y preprocesadores."""
    gkf = GroupKFold(n_splits=cfg_f["folds"])
    seg_truth_f = segment_truth(frame_f)
    members, oof_rho, oof_iri, oof_icv, oof_corn = [], {}, {}, {}, {}
    for fold, (tr, va) in enumerate(gkf.split(frame_f, groups=frame_f["block"])):
        ftr, fva = frame_f.iloc[tr], frame_f.iloc[va]
        pres = {g: Preprocessor().fit(ftr, cols) for g, cols in groups.items()}
        dims = {g: len(cols) for g, cols in groups.items()}
        tgt = Targets().fit(ftr)
        xs_tr, xs_va = to_tensors(pres, ftr), to_tensors(pres, fva)
        y_tr, y_va = tgt.tensors(ftr), tgt.tensors(fva)
        seg_va = seg_truth_f.loc[fva["segment_id"].unique()].copy()
        seg_va.attrs["frame_segments"] = fva["segment_id"].to_numpy()
        for seed in seeds:
            model, mae, ep = train_one(xs_tr, y_tr, xs_va, y_va, seg_va, tgt, dims, cfg_f, cfg_lab, seed)
            members.append({"model": model, "pres": pres, "tgt": tgt, "fold": fold, "seed": seed, "mae_val": mae, "epocas": ep})
            with torch.no_grad():
                out = model(xs_va)
            r, i, d = tgt.decode(out)
            for k, idx in enumerate(fva.index):
                oof_rho.setdefault(idx, []).append(r[k])
                oof_iri.setdefault(idx, []).append(i[k])
                oof_icv.setdefault(idx, []).append(d[k])
                oof_corn.setdefault(idx, []).append(corn_probs(out["corn"])[k].numpy())
        print(f"  [{tag}] fold {fold}: MAE ICV segmento (val) = {np.mean([m['mae_val'] for m in members if m['fold'] == fold]):.2f}", flush=True)
    idx = frame_f.index
    oof = {
        "rho": np.stack([np.mean(oof_rho[i], 0) for i in idx]),
        "iri": np.array([np.mean(oof_iri[i]) for i in idx]),
        "icv_directo": np.array([np.mean(oof_icv[i]) for i in idx]),
        "corn": np.stack([np.mean(oof_corn[i], 0) for i in idx]),
    }
    return members, oof


def predict_ensemble(members, frame, cfg_lab, drop_groups=()):
    rhos, iris, icvs, corns, seg_icvs = [], [], [], [], []
    for m in members:
        xs = to_tensors(m["pres"], frame)
        n = len(frame)
        drop = {g: torch.zeros(n, 1) for g in drop_groups if g in m["model"].groups}
        with torch.no_grad():
            out = m["model"](xs, drop=drop or None)
        r, i, d = m["tgt"].decode(out)
        rhos.append(r)
        iris.append(i)
        icvs.append(d)
        corns.append(corn_probs(out["corn"]).numpy())
        _, seg = derive_scores(r, i, frame["segment_id"].to_numpy(), cfg_lab)
        seg_icvs.append(seg["icv"])
    rho, iri = np.mean(rhos, 0), np.exp(np.mean(np.log(iris), 0))
    fr, seg = derive_scores(rho, iri, frame["segment_id"].to_numpy(), cfg_lab)
    fr["icv_directo"] = np.mean(icvs, 0)
    for k, s in enumerate(pp.STATE_ORDER):
        fr[f"p_corn_{s}"] = np.mean(corns, 0)[:, k]
    seg["icv_std_ensamble"] = pd.concat(seg_icvs, axis=1).std(axis=1)
    return fr, seg


def state_probs(icv, sigma, cfg_lab):
    """P(estado) bajo ICV ~ Normal(icv, sigma), con los límites de estados."""
    bounds = sorted(cfg_lab["states"].items(), key=lambda kv: kv[1])
    lows = [b for _, b in bounds]
    out = {}
    for k, (name, lo) in enumerate(bounds):
        hi = lows[k + 1] if k + 1 < len(lows) else np.inf
        lo_eff = -np.inf if k == 0 else lo
        out[f"p_{name}"] = norm.cdf((hi - icv) / sigma) - norm.cdf((lo_eff - icv) / sigma)
    return pd.DataFrame(out, index=icv.index)


def run_variant(frame, groups, cfg_f, cfg_lab, seeds, tag, drop_groups=()):
    f = frame[frame["split"] == "fusion"].copy()
    members, oof = fit_ensemble(f, groups, cfg_f, cfg_lab, seeds, tag)
    fr_oof, seg_oof = derive_scores(oof["rho"], oof["iri"], f["segment_id"].to_numpy(), cfg_lab)
    res = {"members": members, "seg_oof": seg_oof, "fr_oof": fr_oof.assign(image_id=f["image_id"].to_numpy())}
    for split in ("test", "despliegue"):
        sub = frame[frame["split"] == split]
        if len(sub) == 0 or any(sub[c].isna().all() for g in groups.values() for c in g[:1]):
            continue
        fr, seg = predict_ensemble(members, sub, cfg_lab, drop_groups)
        res[split] = (fr.assign(image_id=sub["image_id"].to_numpy()), seg)
    return res


def conformal(seg_oof, truth_f, alpha=0.10):
    """Cuantil conformal de |residuo| y escala para P(estado), con residuos fuera de fold en F."""
    res = (seg_oof["icv"] - truth_f.loc[seg_oof.index, "icv"]).abs()
    n = len(res)
    q = float(np.quantile(res, min(1, np.ceil((n + 1) * (1 - alpha)) / n)))
    sigma = float(np.sqrt(np.mean(res**2)))
    return q, sigma


def save_release(members, groups, q, sigma, path):
    ensure_dir(path)
    meta = {"grupos": groups, "q90": q, "sigma": sigma, "miembros": []}
    for k, m in enumerate(members):
        torch.save(m["model"].state_dict(), path / f"fusion_{k:02d}.pt")
        meta["miembros"].append(
            {"archivo": f"fusion_{k:02d}.pt", "fold": m["fold"], "seed": m["seed"], "mae_val": m["mae_val"], "epocas": m["epocas"],
             "pre": {g: p.state() for g, p in m["pres"].items()}, "objetivos": m["tgt"].state()}
        )
    (path / "fusion_meta.json").write_text(json.dumps(meta, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="1 semilla y menos épocas (prueba)")
    args = ap.parse_args()
    torch.set_num_threads(4)
    cfg_f = load_yaml("configs/fusion.yaml")
    cfg_lab = pp.load_cfg()
    if args.quick:
        cfg_f.update({"max_epochs": 30, "patience": 8})
    frame = build_tables()
    blocks = pd.read_csv(REPO_ROOT / "data/index/splits.csv")[["image_id", "group_id"]].rename(columns={"group_id": "block"})
    frame = frame.merge(blocks, on="image_id")
    groups_all = feature_groups(frame)
    truth = segment_truth(frame)
    truth_f = truth.loc[frame.loc[frame["split"] == "fusion", "segment_id"].unique()]
    seeds = cfg_f["seeds"][:1] if args.quick else cfg_f["seeds"]
    preds = {}

    # modelo principal
    main_res = run_variant(frame, groups_all, cfg_f, cfg_lab, seeds, "principal")
    q, sigma = conformal(main_res["seg_oof"], truth_f)
    print(f"[ok] principal: q90 conformal = {q:.2f}, sigma = {sigma:.2f}")
    save_release(main_res["members"], groups_all, q, sigma, REPO_ROOT / "models/release/fusion")
    preds["principal"] = main_res

    # ablaciones y controles (1 semilla)
    ab_seeds = seeds[:1]
    imu_null = FEATURES / "null_world"
    variants = {
        "solo_vision": {k: v for k, v in groups_all.items() if k != "imu"},
        "solo_sensores": {"imu": groups_all["imu"]},
        "sin_embedding": {k: v for k, v in groups_all.items() if k != "embedding"},
    }
    for name, grp in variants.items():
        preds[name] = run_variant(frame, grp, cfg_f, cfg_lab, ab_seeds, name)
    # canario de contexto: agregar contexto de Bogotá (no relacionado con las imágenes)
    ctx = pd.read_parquet(FEATURES / "context_segment.parquet")
    ctx_cols = ["dist_colegio_m", "colegio_300m", "dist_ips_m", "dist_hospital_m", "siniestros_250m", "lluvia_30d_mm", "pendiente_pct", "ancho_m", "dim_social", "dim_tecnica", "ip_umv"]
    fc = frame.merge(ctx[["segment_id", *ctx_cols]], on="segment_id", how="left")
    preds["con_contexto"] = run_variant(fc, {**groups_all, "context": ctx_cols}, cfg_f, cfg_lab, ab_seeds, "con_contexto")
    # oráculo: densidades reales (anotaciones) en lugar de detecciones -> techo atribuible al detector
    fo = frame.copy()
    oracle_cols = [f"oraculo_rho_{c}" for c in CLASS_KEYS]
    for c in CLASS_KEYS:
        fo[f"oraculo_rho_{c}"] = fo[f"rho_{c}"]
    preds["oraculo_vision"] = run_variant(fo[fo["split"] != "despliegue"], {"oracle": oracle_cols, "imu": groups_all["imu"]}, cfg_f, cfg_lab, ab_seeds, "oraculo")
    # etiquetas permutadas dentro de F (debe rendir como azar)
    fp = frame.copy()
    fm = fp["split"] == "fusion"
    perm_cols = [f"rho_{c}" for c in CLASS_KEYS] + ["iri_real", "icv", "estado", "segment_id"]
    rng = np.random.default_rng(0)
    seg_ids = fp.loc[fm, "segment_id"].unique()
    shuffled = dict(zip(seg_ids, rng.permutation(seg_ids)))
    donor = fp.loc[fm].set_index("segment_id")
    fp_f = fp.loc[fm].copy()
    for s_id, src in shuffled.items():
        rows_dst = fp_f.index[fp_f["segment_id"] == s_id]
        rows_src = donor.loc[[src]]
        k = min(len(rows_dst), len(rows_src))
        fp_f.loc[rows_dst[:k], perm_cols[:-1]] = rows_src[perm_cols[:-1]].to_numpy()[:k]
    fp.loc[fm] = fp_f
    preds["etiquetas_permutadas"] = run_variant(fp[fp["split"] != "despliegue"], groups_all, cfg_f, cfg_lab, ab_seeds, "permutadas")
    # mundo nulo: solo sensores (IMU sin relación con el daño) intentando predecir la condición visual
    fn = build_tables(sim_dir=DATA / "simulated_null", imu_dir=imu_null).merge(blocks, on="image_id")
    fn = fn[fn["split"] != "despliegue"]
    preds["mundo_nulo_solo_sensores"] = run_variant(fn, {"imu": groups_all["imu"]}, cfg_f, cfg_lab, ab_seeds, "mundo_nulo")

    # guardar predicciones para evaluación
    out = ensure_dir(FEATURES / "predicciones")
    for name, res in preds.items():
        res["seg_oof"].to_parquet(out / f"{name}__fusion_oof_segmento.parquet")
        res["fr_oof"].to_parquet(out / f"{name}__fusion_oof_fotograma.parquet", index=False)
        for split in ("test", "despliegue"):
            if split in res:
                fr, seg = res[split]
                if name == "principal":
                    seg = seg.join(state_probs(seg["icv"], np.sqrt(sigma**2 + seg["icv_std_ensamble"] ** 2), cfg_lab))
                    seg["icv_p05"] = (seg["icv"] - q).clip(0, 100)
                    seg["icv_p95"] = (seg["icv"] + q).clip(0, 100)
                seg.to_parquet(out / f"{name}__{split}_segmento.parquet")
                fr.to_parquet(out / f"{name}__{split}_fotograma.parquet", index=False)
    print(f"[ok] predicciones -> {out}")


if __name__ == "__main__":
    main()
