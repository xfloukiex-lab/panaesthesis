"""Relational field: watch EVERY internal site at once, not one early<->late pair.

`run_field` taps the whole model (all-sites: per-head attention + residual stream
+ MLP neurons + every module), collects a chosen family of taps across the
stimulus, and computes the Hodos relational field over ALL of them — a Diastema
distance matrix and a Symploke coupling matrix — rendered as two heatmaps.

Why a family at a time (site-class + cap): capture is exhaustive, but relations
are pairwise, so relating N taps is O(N^2). The instrument taps everything and
lets you point the relational readout at a coherent family (the residual stream
across depth, the heads within a layer, the MLP neuron blocks) or a capped span
of all of them. The cost is stated, not hidden.

Mechanism note (design-the-test-first): per-head Q/K/V/Z vectors and residual
sites are meaningful for any stimulus; attention *patterns* need S>1 and are a
separate readout (sites.attention_patterns).
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hodos_monitor import equations, sites, stimuli
from hodos_monitor.equations import symploke, NB, N_PAIR, NJ, K_SHUFFLE


# --- relational-field engine -------------------------------------------------
# Lives ABOVE the equations (FAMILY-DESIGN principle: equations.py stays v1
# verbatim — the four pure Hodos equations only; every field/matrix readout is
# built on top of them here). These reuse the v1 symploke() and its constants.
def _flat(a):
    a = np.asarray(a, dtype=float)
    return a.reshape(a.shape[0], -1)          # (T, W)


def field_distributions(acts, nb=NB):
    """Per-tap per-step distributions on ONE shared quantile grid.

    acts: {tap_name: (T, ...) array}. Same standardize -> active-units ->
    shared-edges recipe as layer_distributions, but pooled over EVERY tap so all
    taps live on one comparable outcome space. Returns (taps, P) with P shape
    (N_taps, T, nb).
    """
    taps = list(acts.keys())
    if not taps:
        raise ValueError("field_distributions: no taps")
    T = _flat(acts[taps[0]]).shape[0]
    z, act, pooled = {}, {}, []
    for tp in taps:
        a = _flat(acts[tp])
        if a.shape[0] != T:
            raise ValueError(f"tap {tp!r} has {a.shape[0]} steps, expected {T}")
        az = (a - a.mean()) / (a.std() + 1e-12)
        m = a != 0
        z[tp], act[tp] = az, m
        if m.any():
            pooled.append(az[m])
    pooled = np.concatenate(pooled) if pooled else np.zeros(1)
    edges = np.quantile(pooled, np.linspace(0, 1, nb + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    P = np.zeros((len(taps), T, nb))
    for n, tp in enumerate(taps):
        az, m = z[tp], act[tp]
        for t in range(T):
            h = np.histogram(az[t][m[t]], bins=edges)[0]
            P[n, t] = h / h.sum() if h.sum() else np.full(nb, 1.0 / nb)
    return taps, P


def diastema_matrix(P):
    """Mean per-step Fisher-Rao gap between EVERY tap pair. (N,T,nb) -> (N,N).

    This is the field's Diastema: the per-moment ground distance g(p_i, q_j)
    averaged over the run — NOT the path-normalized DTW form (that is O(T^2) per
    pair and is kept in diastema() for a single focus pair). A symmetric N x N
    map of how far apart every internal site lives from every other.
    """
    N, T, nb = P.shape
    sq = np.sqrt(P)
    D = np.zeros((N, N))
    for i in range(N):
        bc = np.clip(np.einsum("tb,jtb->jt", sq[i], sq), 0.0, 1.0)   # (N, T)
        D[i] = (2.0 * np.arccos(bc)).mean(axis=1)
    return D


def symploke_matrix(acts, taps, seed, n_pair=N_PAIR, nj=NJ, k=K_SHUFFLE):
    """Mean Symploke braid strength z between EVERY tap pair. -> (Z, meta).

    Z[i,j] = mean over the run of the per-step z-scored surplus of the joint of
    (tap_i, tap_j) over the product of their marginals — the same symploke() the
    two-layer portrait uses, computed for the whole field. Each pair gets its own
    seeded rng (from `seed`) so the matrix is deterministic and order-independent.
    Both directions are computed (the grouping is asymmetric between early/late,
    so Z is near- but not exactly symmetric — reported as measured, not forced).

    n_pair is clamped down to the narrowest tap's width (a head is head_dim wide);
    the effective value is returned in meta.
    """
    widths = [_flat(acts[t]).shape[1] for t in taps]
    npair = int(min(n_pair, min(widths)))
    if npair < 2:
        raise ValueError(
            f"symploke_matrix: narrowest tap is {min(widths)} wide — need >=2")
    N = len(taps)
    A = {t: _flat(acts[t]) for t in taps}
    Z = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            rng = np.random.default_rng((seed & 0xFFFFFFFF) ^ (i * 2654435761)
                                        ^ (j * 40503))
            _, _, _, z = symploke(A[taps[i]], A[taps[j]], rng,
                                  nj=nj, k=k, n_pair=npair)
            Z[i, j] = float(np.mean(z))
    meta = {"n_pair_effective": npair, "n_pair_requested": int(n_pair),
            "k_shuffle": int(k), "nj": int(nj)}
    return Z, meta


def relational_field(acts, seed, n_pair=N_PAIR):
    """The full relational field over every tapped internal site.

    acts: {tap_name: (T, ...) array} for as many taps as you want related.
    Returns a dict: taps, D-matrix (Diastema, per-step FR gap), Z-matrix
    (Symploke mean braid z), and honest summaries. This is the readout that
    watches EVERY internal relation, not one early<->late pair.
    """
    taps, P = field_distributions(acts)
    D = diastema_matrix(P)
    Z, zmeta = symploke_matrix(acts, taps, seed, n_pair=n_pair)
    iu = np.triu_indices(len(taps), k=1)
    return {
        "taps": taps,
        "n_taps": len(taps),
        "D_matrix": D,
        "Z_matrix": Z,
        "D_offdiag_mean": float(D[iu].mean()) if len(taps) > 1 else 0.0,
        "D_offdiag_max": float(D[iu].max()) if len(taps) > 1 else 0.0,
        "z_field_mean": float(Z[~np.eye(len(taps), dtype=bool)].mean())
        if len(taps) > 1 else 0.0,
        "z_field_max": float(Z.max()) if len(taps) > 1 else 0.0,
        "symploke_meta": zmeta,
    }


def _short(tap, prefix):
    s = tap[len(prefix):] if prefix and tap.startswith(prefix) else tap
    s = s.replace("model.", "").replace("layers.", "L").replace("self_attn", "attn")
    return s


def _common_prefix(taps):
    if not taps:
        return ""
    p = taps[0]
    for t in taps[1:]:
        while not t.startswith(p):
            p = p[:-1]
            if not p:
                return ""
    return p.rsplit(".", 1)[0] + "." if "." in p else ""


def select_taps(all_names, site_class=None, max_taps=48):
    """Pick which taps to relate. site_class in {resid,head,mlp,attn,module,all}."""
    inner = [t for t in all_names if t != "output"]
    if site_class in (None, "all"):
        chosen = inner
        # a readable default when relating 'all': the residual stream if present
        resid = [t for t in inner if sites.classify(t) == "resid"]
        if resid:
            chosen = resid
    else:
        chosen = [t for t in inner if sites.classify(t) == site_class]
    if not chosen:
        raise SystemExit(
            f"no taps for site-class {site_class!r}; available classes: "
            + ", ".join(sorted({sites.classify(t) for t in inner})))
    if len(chosen) > max_taps:
        idx = np.linspace(0, len(chosen) - 1, max_taps).round().astype(int)
        chosen = [chosen[i] for i in dict.fromkeys(idx)]
    return chosen


def collect_acts(model, stimulus, keep):
    """Forward the stimulus once; collect the kept taps as (T, W) arrays.

    Shared by the field readout, the family watcher, and the multi-level
    changer so the capture loop lives in exactly one place. Returns
    (acts, preds, ys):
      acts  : {tap: (T, W)} for the taps that fired on step 0
      preds : (T,) argmax of the model's "output" tap (or -1 if none)
      ys    : list of the stimulus labels (None where unlabeled)
    """
    keep = list(keep)
    T = len(stimulus)
    seqs = {k: [] for k in keep}
    preds, ys = [], []
    fired = None
    for t in range(T):
        x, y, _r = stimulus.step(t)
        acts = model.forward(np.asarray(x)[None])
        if fired is None:
            fired = [k for k in keep if k in acts]
            if not fired:
                raise SystemExit(
                    "none of the selected taps fired on step 0 — "
                    f"selected {keep[:3]}..., model produced {list(acts)[:3]}...")
            seqs = {k: [] for k in fired}
        for k in fired:
            seqs[k].append(np.asarray(acts[k]).ravel())
        out = acts.get("output")
        preds.append(int(np.asarray(out).ravel().argmax())
                     if out is not None else -1)
        ys.append(y)
    acts_field = {}
    for k, seq in seqs.items():
        w = min(len(v) for v in seq)
        acts_field[k] = np.stack([v[:w] for v in seq])   # (T, W)
    return acts_field, np.array(preds), ys


def run_field(model, stimulus, seed, out_dir, site_class=None, max_taps=48,
              n_pair=64):
    from hodos_monitor.adapters import load_model
    model = load_model(model, all_sites=True)
    if isinstance(stimulus, str):
        stimulus = stimuli.load_stimulus(stimulus, seed=seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    keep = select_taps(model.tap_names(), site_class, max_taps)
    acts_field, _preds, _ys = collect_acts(model, stimulus, keep)

    fld = relational_field(acts_field, seed=int(seed), n_pair=n_pair)

    prefix = _common_prefix(fld["taps"])
    labels = [_short(t, prefix) for t in fld["taps"]]
    desc = model.describe()
    title = (f"{desc.get('model_name','model')} — relational field over "
             f"{fld['n_taps']} internal sites "
             f"(class: {site_class or 'all/resid'})")
    _render_field(out_dir / "field.png", fld, labels, title, desc)

    out = {
        "model": desc,
        "site_class": site_class or "all/resid",
        "n_taps": fld["n_taps"],
        "taps": fld["taps"],
        "D_offdiag_mean": fld["D_offdiag_mean"],
        "D_offdiag_max": fld["D_offdiag_max"],
        "z_field_mean": fld["z_field_mean"],
        "z_field_max": fld["z_field_max"],
        "symploke_meta": fld["symploke_meta"],
        "D_matrix": fld["D_matrix"].tolist(),
        "Z_matrix": fld["Z_matrix"].tolist(),
        "provenance": {
            "stimulus": stimulus.describe().get("stimulus", "?"),
            "seed": str(seed),
            "n_pair_requested": int(n_pair),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "note": "Diastema matrix = mean per-step Fisher-Rao gap (not DTW); "
                    "Symploke matrix = mean braid z; per-head + residual-stream "
                    "taps derived by hodos_monitor.sites; white-box torch only.",
        },
    }
    with open(out_dir / "field_metrics.json", "w") as f:
        json.dump(out, f, indent=1)
    model.close()
    return out


def _render_field(path, fld, labels, title, desc):
    D, Z = fld["D_matrix"], fld["Z_matrix"]
    n = len(labels)
    fig, (axD, axZ) = plt.subplots(1, 2, figsize=(min(26, 6 + n * 0.34),
                                                  min(13, 4 + n * 0.17)))
    fig.suptitle(title, fontsize=13, weight="bold")

    imD = axD.imshow(D, cmap="magma", aspect="auto")
    axD.set_title("Diastema — how far apart sites live\n(Fisher-Rao gap, mean/step)",
                  fontsize=10)
    imZ = axZ.imshow(Z, cmap="viridis", aspect="auto")
    axZ.set_title("Symploke — how strongly sites braid\n(mean z vs shuffle null)",
                  fontsize=10)
    for ax, im in ((axD, imD), (axZ, imZ)):
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        fs = max(4, min(8, int(220 / max(1, n))))
        ax.set_xticklabels(labels, rotation=90, fontsize=fs)
        ax.set_yticklabels(labels, fontsize=fs)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.text(0.5, 0.005,
             "Every internal site tapped; the relational field computed over the "
             "whole family. White-box (torch hooks) only.",
             ha="center", fontsize=8, color="gray")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(path, dpi=110)
    plt.close(fig)
