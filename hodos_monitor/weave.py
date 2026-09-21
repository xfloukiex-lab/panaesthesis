"""The splicer: a surgical relational actuator for the Hodos monitor.

Monitor -> act -> monitor. The monitor (portrait.py) watches the AI's actions
and measures the relation between two layers (the Symploke braid). The splicer
CHANGES that relation, using the monitored actions themselves — no external
signal, no output steering.

Splice, not scramble. The retired blanket weaver re-dealt ALL late units in
the early layer's rank order at every step — a blind permutation of the whole
feature map. From the head's perspective that is indistinguishable from noise
injection: the wire labels the head was trained on get scrambled, accuracy
dies, and nothing attributable is learned. A splice is surgical: the user
names specific (early_group, late_unit) wires, and ONLY those wires are
touched. Every other unit runs bit-identical to the natural pass, so any
change in the braid or in accuracy maps to exactly the wires named.

Mechanism: temporal rank re-dealing, per named wire. Over the splice range:
    A[:, g] = grouped-mean time series of early group g (never modified)
    L[:, j] = time series of late unit j
    splice: re-deal L[:, j]'s own values in time so they follow A's rank
            order — at the step where early group g peaks, late unit j shows
            its own peak value. A pure temporal permutation: unit j keeps its
            identity and its exact value multiset (marginal preserved
            bitwise), only WHEN it fires is rerouted. The head reads familiar
            values on familiar wires, at deliberately rerouted times.
    cut:    re-deal L[:, j]'s values in a seeded random temporal order —
            destroys whatever natural coupling the wire had, same
            preservation guarantees. The "taking out" half, on demand.
    blend:  spliced = (1 - alpha) * natural + alpha * re-dealt.
At alpha = 1 the touched wires are pure permutations in time; untouched wires
are bit-identical (asserted, reported). The forward pass then continues
downstream of the late tap (numpy: exact replay of the remaining layers;
torch: the same pass with a hook override), and predictions are recorded
exactly as portrait.py does.

Honest contract, stated up front: rerouting WHEN a wire fires still changes
what the head reads on that wire at each step — the head sees familiar values
at unfamiliar times. That is the intervention, reported per wire, not hidden.
What the splice buys over the scramble is attribution: one wire changed,
everything else controlled, so cause maps to effect instead of dissolving
into noise. The equations themselves (equations.py) are untouched.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch

from hodos_monitor import equations, stimuli
from hodos_monitor.adapters import load_model
from hodos_monitor import lora as lora_mod

TARGET_RELATION = "splice the named wires under strain"


def _temporal_rank_splice(col, driver):
    """Re-deal col's values over time into driver's rank order.

    Pure temporal permutation: the step where `driver` peaks gets col's peak
    value, and so on. col keeps its identity and its exact multiset of
    values; only the timing is rerouted.
    """
    col = np.asarray(col).ravel()
    driver = np.asarray(driver).ravel()
    if col.shape[0] != driver.shape[0]:
        raise ValueError(
            f"splice needs |col| == |driver|, got {col.shape[0]} vs "
            f"{driver.shape[0]}")
    rank = np.argsort(np.argsort(driver))
    return np.sort(col)[rank]


def _temporal_cut(col, rng):
    """Re-deal col's values over time in a seeded random order.

    Destroys the wire's natural temporal coupling; identity and value
    multiset preserved.
    """
    return np.asarray(rng.permutation(np.asarray(col).ravel()))


def pair_coupling(a, b, rng, nj=12, k=8):
    """epsilon and shuffle-null z for ONE (early_group, late_unit) pair.

    a, b: 1-D time series over the same steps. Joint histogram vs product of
    marginals, scored against k temporal shuffles of b drawn from rng.
    Returns (eps, z).
    """
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if a.shape[0] != b.shape[0]:
        raise ValueError(
            f"pair_coupling needs |a| == |b|, got {a.shape[0]} vs {b.shape[0]}")
    eA = np.quantile(a, np.linspace(0, 1, nj + 1))
    eA[0] -= 1e-9
    eA[-1] += 1e-9
    eB = np.quantile(b, np.linspace(0, 1, nj + 1))
    eB[0] -= 1e-9
    eB[-1] += 1e-9
    ia = np.clip(np.digitize(a, eA) - 1, 0, nj - 1)
    ib = np.clip(np.digitize(b, eB) - 1, 0, nj - 1)
    J = np.zeros((nj, nj))
    np.add.at(J, (ia, ib), 1)
    J /= J.sum()
    M = np.outer(J.sum(1), J.sum(0))
    eps = equations.gcost(J.ravel(), M.ravel())
    ns = []
    for _ in range(k):
        ib2 = rng.permutation(ib)
        J2 = np.zeros((nj, nj))
        np.add.at(J2, (ia, ib2), 1)
        J2 /= J2.sum()
        ns.append(equations.gcost(J2.ravel(), M.ravel()))
    nullm = float(np.mean(ns))
    nulls = float(np.std(ns)) + 1e-12
    return float(eps), float((eps - nullm) / nulls)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _wire_marginal_drift(col_nat, col_spl, n_bins=24):
    """Total-variation drift of ONE wire's value distribution over the range.

    At alpha=1 this is exactly 0 by construction (temporal permutation).
    """
    lo = float(col_nat.min())
    hi = float(col_nat.max())
    if hi <= lo:
        return 0.0
    edges = np.linspace(lo, hi, n_bins + 1)
    h1, _ = np.histogram(col_nat, bins=edges)
    h2, _ = np.histogram(col_spl, bins=edges)
    h1 = h1 / h1.sum() if h1.sum() else np.full(n_bins, 1 / n_bins)
    h2 = h2 / h2.sum() if h2.sum() else np.full(n_bins, 1 / n_bins)
    return float(0.5 * np.abs(h1 - h2).sum())


def _null_after_rng(seed):
    _null_after_int = int(hashlib.sha256(
        f"hodos-splice-null-{seed}".encode()).hexdigest(), 16) % (2 ** 63)
    return np.random.default_rng(_null_after_int)


def _cut_rng(seed):
    _cut_int = int(hashlib.sha256(
        f"hodos-splice-cut-{seed}".encode()).hexdigest(), 16) % (2 ** 63)
    return np.random.default_rng(_cut_int)


def _downstream(model, late_tap, x_batch, late_shape, spliced_flat):
    """Continue the forward pass downstream of the spliced late tap."""
    act = np.asarray(spliced_flat).reshape(late_shape)
    if hasattr(model, "continue_from"):
        return np.asarray(model.continue_from(late_tap, act))
    out = model.forward(x_batch, overrides={late_tap: act})["output"]
    return np.asarray(out)


def splice_run(model, stimulus, alpha=1.0, splice=(), cut=(),
               splice_range=None, seed=20260918, out_dir=".", taps=None,
               n_pair=64, target_relation=TARGET_RELATION,
               lora_rank=0, lora_steps=300, lora_lr=0.5, lora_seed=0,
               return_data=False):
    """Run the spliced read on any model + stimulus. Returns the metrics dict.

    model          : ModelAdapter (or a spec string for adapters.load_model)
    stimulus       : Stimulus (or a spec string for stimuli.load_stimulus)
    alpha          : splice strength in [0, 1]; 0 = natural run
    splice         : iterable of (early_group, late_unit) wires to couple —
                     early_group indexes the n_pair grouped means, late_unit
                     indexes the late tap's raw units
    cut            : iterable of (early_group, late_unit) wires to decouple
                     (seeded random temporal re-dealing); a wire may not be
                     both spliced and cut
    splice_range   : (start, end) step indices where alpha applies; default =
                     the stimulus's "strain" regime span (raises if the
                     stimulus has no strain regime)
    seed           : stimulus seed when stimulus is a spec string; also seeds
                     the AFTER-pass null draws and the cut permutations
                     (documented, reproducible)
    taps           : (early, late, output) tap names; default = adapter's
    n_pair         : Symploke pairing count for the measurement (also the
                     early grouping used to address splice wires)
    lora_rank      : if > 0 and labels exist, train a residual LoRA readout
                     on the splice-range steps, twice — once on natural
                     features (control), once on spliced features — and
                     report both accuracies. The delta isolates what the
                     installed relation contributed.
    lora_steps     : LoRA gradient steps
    lora_lr        : LoRA learning rate
    lora_seed      : LoRA init seed (documented, reproducible)
    return_data    : if True, return (metrics, data) where data holds the
                     raw arrays (late_nat, late_spl, logits, labels, early
                     groups, range) for visualization; default False keeps
                     the plain metrics dict
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    splice = [(int(g), int(j)) for g, j in splice]
    cut = [(int(g), int(j)) for g, j in cut]
    if not splice and not cut:
        raise ValueError("splice_run needs at least one --splice or --cut "
                         "wire — the instrument is surgical, not blanket")
    model = load_model(model)
    if isinstance(stimulus, str):
        stimulus = stimuli.load_stimulus(stimulus, seed=seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    T = len(stimulus)
    masks = stimulus.masks()
    if splice_range is None:
        strain_m = masks.get("strain")
        if strain_m is None or not strain_m.any():
            raise ValueError(
                "no splice_range given and the stimulus has no 'strain' "
                "regime — pass splice_range=(start, end) explicitly")
        idx = np.flatnonzero(strain_m)
        splice_range = (int(idx[0]), int(idx[-1]) + 1)
    w0, w1 = int(splice_range[0]), int(splice_range[1])
    if not (0 <= w0 < w1 <= T):
        raise ValueError(
            f"splice_range must satisfy 0 <= start < end <= {T}, "
            f"got {(w0, w1)}")
    splice_mask = np.zeros(T, dtype=bool)
    splice_mask[w0:w1] = True
    ridx = np.flatnonzero(splice_mask)

    early_tap, late_tap, out_tap = taps or model.default_taps()

    rng_before = getattr(stimulus, "rng", None)
    if rng_before is None:
        rng_before = np.random.default_rng(seed)
    rng_null_after = _null_after_rng(seed)
    rng_cut = _cut_rng(seed)

    # ---- pass 1: the natural run ----
    early, late_nat, ys = [], [], []
    late_shape = None
    n_late = None
    preds_nat = []
    logits_nat = []
    for t in range(T):
        x_raw, y, _regime = stimulus.step(t)
        ys.append(y)
        x_batch = np.asarray(x_raw)[None]
        acts = model.forward(x_batch)
        e = np.asarray(acts[early_tap]).ravel()
        b_nat = np.asarray(acts[late_tap]).ravel()
        lg_out = np.asarray(acts[out_tap])
        if late_shape is None:
            late_shape = np.asarray(acts[late_tap]).shape
            n_late = b_nat.shape[0]
            n_early = e.shape[0]
            if n_early < n_pair:
                raise ValueError(
                    f"splicer needs early width >= n_pair={n_pair} "
                    f"(early is grouped into {n_pair} addressable groups); "
                    f"got early={n_early}")
        early.append(e)
        late_nat.append(b_nat)
        preds_nat.append(int(np.asarray(lg_out).ravel().argmax()))
        logits_nat.append(np.asarray(lg_out).ravel().astype(np.float64))
    early = np.array(early)
    late_nat = np.array(late_nat)
    preds_nat = np.array(preds_nat)
    ys = np.array(ys, dtype=object)

    # validate wires against measured widths
    for kind, pairs in (("splice", splice), ("cut", cut)):
        for g, j in pairs:
            if not (0 <= g < n_pair):
                raise ValueError(
                    f"--{kind} wire ({g}:{j}): early group {g} out of range "
                    f"for n_pair={n_pair}")
            if not (0 <= j < n_late):
                raise ValueError(
                    f"--{kind} wire ({g}:{j}): late unit {j} out of range "
                    f"for late width {n_late}")
    spliced_wires = {j for _, j in splice}
    cut_wires = {j for _, j in cut}
    overlap = spliced_wires & cut_wires
    if overlap:
        raise ValueError(
            f"wire(s) {sorted(overlap)} are both spliced and cut — "
            f"pick one operation per wire")

    # ---- the splice: temporal re-dealing, named wires only ----
    A = np.array([equations.grouped_means(early[t], n_pair, seed=0)
                  for t in range(T)])
    late_spl = late_nat.copy()
    a_t = float(alpha)
    if a_t > 0.0:
        for g, j in splice:
            redealt = _temporal_rank_splice(late_nat[ridx, j], A[ridx, g])
            late_spl[ridx, j] = (1.0 - a_t) * late_nat[ridx, j] + a_t * redealt
        for g, j in cut:
            redealt = _temporal_cut(late_nat[ridx, j], rng_cut)
            late_spl[ridx, j] = (1.0 - a_t) * late_nat[ridx, j] + a_t * redealt

    touched = spliced_wires | cut_wires
    untouched = [j for j in range(n_late) if j not in touched]
    untouched_identical = bool(
        (late_spl[:, untouched] == late_nat[:, untouched]).all()) \
        if untouched else True

    # ---- pass 2: continue downstream on the touched steps ----
    preds = preds_nat.copy()
    logits_spl = [l.copy() for l in logits_nat]
    if a_t > 0.0:
        for t in ridx:
            x_raw, _, _ = stimulus.step(t)
            x_batch = np.asarray(x_raw)[None]
            out = _downstream(model, late_tap, x_batch, late_shape,
                              late_spl[t])
            preds[t] = int(np.asarray(out).ravel().argmax())
            logits_spl[t] = np.asarray(out).ravel().astype(np.float64)
    alphas = np.where(splice_mask, a_t, 0.0)

    has_labels = ys[0] is not None
    strain_m = masks.get("strain")
    coast_m = masks.get("coast")
    if has_labels and strain_m is not None and coast_m is not None:
        yv = np.array([int(v) for v in ys])
        acc_clean = float((preds[coast_m] == yv[coast_m]).mean())
        acc_degraded = float((preds[strain_m] == yv[strain_m]).mean())
    else:
        acc_clean = acc_degraded = None

    # ---- LoRA readout: the branch. Trained twice — natural (control) and
    # spliced — on the splice-range steps, same rank/steps/lr/seed. The
    # delta isolates what the installed relation contributed. ----
    lora_block = {"enabled": False,
                  "note": "skipped (lora_rank=0 or no labels)"}
    if lora_rank > 0 and has_labels:
        yv_all = np.array([int(v) for v in ys])
        y_r = yv_all[ridx]
        F_nat = late_nat[ridx]
        Z_nat = np.array(logits_nat)[ridx]
        F_spl = late_spl[ridx]
        Z_spl = np.array(logits_spl)[ridx]
        A_n, B_n, loss_n = lora_mod.train_residual_lora(
            F_nat, Z_nat, y_r, rank=lora_rank, steps=lora_steps,
            lr=lora_lr, seed=lora_seed)
        A_s, B_s, loss_s = lora_mod.train_residual_lora(
            F_spl, Z_spl, y_r, rank=lora_rank, steps=lora_steps,
            lr=lora_lr, seed=lora_seed)
        acc_lora_nat = lora_mod.accuracy_from_logits(
            lora_mod.apply_lora(F_nat, Z_nat, A_n, B_n), y_r)
        acc_lora_spl = lora_mod.accuracy_from_logits(
            lora_mod.apply_lora(F_spl, Z_spl, A_s, B_s), y_r)
        lora_block = {
            "enabled": True,
            "rank": int(lora_rank),
            "steps": int(lora_steps),
            "lr": float(lora_lr),
            "seed": int(lora_seed),
            "n_train_steps": int(len(ridx)),
            "loss_first_nat": float(loss_n[0]),
            "loss_last_nat": float(loss_n[-1]),
            "loss_first_spl": float(loss_s[0]),
            "loss_last_spl": float(loss_s[-1]),
            "acc_lora_nat": acc_lora_nat,
            "acc_lora_spl": acc_lora_spl,
            "acc_delta_spl_minus_nat": acc_lora_spl - acc_lora_nat,
            "note": ("residual LoRA on the frozen model; B starts at zero "
                     "(no-op). Control (natural) and spliced get identical "
                     "training; the delta is the splice's contribution."),
        }

    # per-wire marginal drift over the range (0 at alpha=1 by construction)
    wire_drift = {}
    for j in sorted(touched):
        wire_drift[str(j)] = _wire_marginal_drift(late_nat[ridx, j],
                                                  late_spl[ridx, j])

    # per-wire coupling before/after, over the splice range
    rng_pair_b = _null_after_rng(f"{seed}-pair-before")
    rng_pair_a = _null_after_rng(f"{seed}-pair-after")
    pair_rows = []
    for kind, pairs in (("splice", splice), ("cut", cut)):
        for g, j in pairs:
            eps_b, z_b = pair_coupling(A[ridx, g], late_nat[ridx, j],
                                       rng_pair_b)
            eps_a, z_a = pair_coupling(A[ridx, g], late_spl[ridx, j],
                                       rng_pair_a)
            pair_rows.append({
                "op": kind, "early_group": g, "late_unit": j,
                "eps_before": eps_b, "z_before": z_b,
                "eps_after": eps_a, "z_after": z_a,
                "wire_drift_tv": wire_drift[str(j)],
            })

    # BEFORE: the natural relation; AFTER: the spliced relation.
    eps_b, nullm_b, nulls_b, z_b = equations.symploke(
        early, late_nat, rng_before, n_pair=n_pair)
    eps_a, nullm_a, nulls_a, z_a = equations.symploke(
        early, late_spl, rng_null_after, n_pair=n_pair)

    P, Q = equations.layer_distributions(early, late_spl)
    D_total, gap = equations.diastema(P, Q)
    excess, z_arrow, Phi, tau = equations.chronos(eps_a, nullm_a)
    if strain_m is not None and coast_m is not None:
        ver = equations.verify(z_a, gap, tau,
                               np.flatnonzero(strain_m),
                               np.flatnonzero(coast_m))
    else:
        ver = {"passed": None, "status": "NO_REGIMES (see notes)",
               "notes": ["stimulus has no strain/coast regimes — "
                         "verification needs both"]}

    if strain_m is not None and coast_m is not None:
        s_idx, c_idx = np.flatnonzero(strain_m), np.flatnonzero(coast_m)
        n_strain, n_coast = len(s_idx), len(c_idx)
        slope_strain = float((tau[s_idx[-1]] - tau[s_idx[0]]) / n_strain)
        slope_coast = float(((tau[s_idx[0] - 1] - tau[0]) +
                             (tau[-1] - tau[s_idx[-1]])) / n_coast) \
            if s_idx[0] > 0 else float("nan")
        z_mean_strain_b = float(z_b[s_idx].mean())
        z_mean_coast_b = float(z_b[c_idx].mean())
        z_mean_strain_a = float(z_a[s_idx].mean())
        z_mean_coast_a = float(z_a[c_idx].mean())
        eps_mean_strain_b = float(eps_b[s_idx].mean())
        eps_mean_strain_a = float(eps_a[s_idx].mean())
    else:
        slope_strain = slope_coast = float("nan")
        z_mean_strain_b = z_mean_coast_b = float("nan")
        z_mean_strain_a = z_mean_coast_a = float("nan")
        eps_mean_strain_b = eps_mean_strain_a = float("nan")

    model_desc = model.describe()
    stim_desc = stimulus.describe()
    metrics = {
        "D_total": float(D_total),
        "gap_min": float(gap.min()),
        "gap_max": float(gap.max()),
        "gap_std": float(gap.std()),
        "gap_mean": float(gap.mean()),
        "z_mean_strain": z_mean_strain_a,
        "z_mean_coast": z_mean_coast_a,
        "z_max_strain": float(z_a[strain_m].max()) if strain_m is not None else float("nan"),
        "z_max_coast": float(z_a[coast_m].max()) if coast_m is not None else float("nan"),
        "eps_mean": float(eps_a.mean()),
        "eps_max": float(eps_a.max()),
        "eps_min": float(eps_a.min()),
        "tau_end": float(tau[-1]),
        "tau_slope_strain": slope_strain,
        "tau_slope_coast": slope_coast,
        "z_arrow": float(z_arrow),
        "Phi": float(Phi),
        "acc_clean": acc_clean,
        "acc_degraded": acc_degraded,
        "verification": ver,
        "lora": lora_block,
        "splice": {
            "target_relation": target_relation,
            "alpha": float(alpha),
            "splice_range": [w0, w1],
            "n_spliced_steps": int(splice_mask.sum()),
            "n_pair": int(n_pair),
            "late_units": int(n_late),
            "spliced_wires": [[g, j] for g, j in splice],
            "cut_wires": [[g, j] for g, j in cut],
            "untouched_wires": untouched,
            "untouched_wires_bit_identical": untouched_identical,
            "wires": pair_rows,
            "z_mean_strain_before": z_mean_strain_b,
            "z_mean_strain_after": z_mean_strain_a,
            "z_mean_coast_before": z_mean_coast_b,
            "z_mean_coast_after": z_mean_coast_a,
            "eps_mean_strain_before": eps_mean_strain_b,
            "eps_mean_strain_after": eps_mean_strain_a,
            "note": ("surgical intervention: named wires only, everything "
                     "else bit-identical; rerouted wires keep their identity "
                     "and value multiset — only timing changes; accuracy "
                     "reported as measured"),
        },
        "provenance": {
            **model_desc,
            **{f"stimulus_{k}": v for k, v in stim_desc.items()},
            "taps": {"early": early_tap, "late": late_tap,
                     "output": out_tap},
            "seed": str(seed),
            "run_kind": "spliced" if alpha > 0 else "natural(alpha=0)",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }

    sections = _regime_sections(masks, T)
    _render_spliced(out_dir / "spliced_portrait.png", metrics, P, Q, gap,
                    eps_b, eps_a, nullm_a, z_a, tau, alphas, w0, w1,
                    model_desc.get("model_name", "model"), sections)
    with open(out_dir / "spliced_metrics.json", "w") as f:
        json.dump(metrics, f, indent=1)
    per_step = {
        "t": list(range(T)),
        "alpha": [float(a) for a in alphas],
        "spliced": [bool(m) for m in splice_mask],
        "eps_before": [float(v) for v in eps_b],
        "eps_after": [float(v) for v in eps_a],
        "z_before": [float(v) for v in z_b],
        "z_after": [float(v) for v in z_a],
    }
    with open(out_dir / "per_step.json", "w") as f:
        json.dump(per_step, f, indent=1)
    if return_data:
        data = {
            "late_nat": late_nat, "late_spl": late_spl,
            "early": np.array(early),
            "preds": np.array(preds, dtype=int),
            "logits_nat": np.array(logits_nat),
            "logits_spl": np.array(logits_spl),
            "ys": np.array([int(v) if v is not None else -1 for v in ys]),
            "early_groups": A,
            "range": (int(w0), int(w1)), "ridx": ridx,
            "splice_wires": [(int(g), int(u)) for (g, u) in splice],
        }
        return metrics, data
    return metrics


def _regime_sections(masks, T):
    """[(t0, t1, label, color)] spans for the render, from regime masks."""
    palette = {"strain": "#ef4444", "coast": "#22d3ee", "all": "#22d3ee"}
    sections = []
    for regime, m in masks.items():
        idx = np.flatnonzero(m)
        if len(idx) == 0:
            continue
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        for run in runs:
            t0, t1 = int(run[0]), int(run[-1]) + 1
            label = ("DEGRADED INPUT — SPLICED" if regime == "strain"
                     else regime.upper() + " INPUT")
            sections.append((t0, t1, label,
                             palette.get(regime, "#a78bfa")))
    sections.sort()
    return sections


def run_splice(weights_path, splice, cut=(), alpha=1.0,
               splice_range=(40, 80), stimulus_seed=20260918, out_dir=".",
               lora_rank=0, lora_steps=300, lora_lr=0.5, lora_seed=0):
    """Legacy entry point: .npz checkpoint + reader120 stimulus."""
    from hodos_monitor.stimuli import Reader120Stimulus
    model = load_model(f"npz:{weights_path}")
    stimulus = Reader120Stimulus(stimulus_seed)
    try:
        return splice_run(model, stimulus, alpha=alpha, splice=splice,
                          cut=cut, splice_range=tuple(splice_range),
                          seed=stimulus_seed, out_dir=out_dir,
                          lora_rank=lora_rank, lora_steps=lora_steps,
                          lora_lr=lora_lr, lora_seed=lora_seed)
    finally:
        model.close()


_COMPARE_ROWS = [
    ("acc_clean", "accuracy, clean stretches"),
    ("acc_degraded", "accuracy, degraded stretch"),
    ("z_mean_strain", "braid z-mean under strain"),
    ("z_mean_coast", "braid z-mean on clean"),
    ("z_max_strain", "braid z-max under strain"),
    ("tau_slope_strain", "lived-time slope, strain"),
    ("tau_slope_coast", "lived-time slope, coast"),
    ("tau_end", "lived time total"),
    ("D_total", "Diastema D(layers)"),
    ("z_arrow", "time's arrow z"),
]


def _fmt(v):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) \
        else f"{v:.4f}"


def compare_runs(natural, spliced):
    """Side-by-side table: natural run metrics vs spliced run metrics.

    Both dicts share the metrics.json schema (spliced has the extra "splice"
    block). Returns the table as a string; prints nothing.
    """
    sp = spliced.get("splice", {})
    lines = []
    lines.append("natural vs spliced — what changed and what didn't")
    lines.append(f"target relation (declared before run): "
                 f"{sp.get('target_relation', '?')}; "
                 f"alpha={sp.get('alpha', '?')}, "
                 f"splice_range={sp.get('splice_range', '?')}")
    sw = ", ".join(f"{g}:{j}" for g, j in sp.get("spliced_wires", [])) or "-"
    cw = ", ".join(f"{g}:{j}" for g, j in sp.get("cut_wires", [])) or "-"
    lines.append(f"spliced wires (early_group:late_unit): {sw}; "
                 f"cut wires: {cw}")
    lines.append(f"untouched wires bit-identical: "
                 f"{sp.get('untouched_wires_bit_identical', '?')}")
    for w in sp.get("wires", []):
        lines.append(f"  wire {w['op']} {w['early_group']}:{w['late_unit']}: "
                     f"pair z {_fmt(w['z_before'])} -> {_fmt(w['z_after'])}, "
                     f"wire drift TV {_fmt(w['wire_drift_tv'])}")
    lines.append("")
    lines.append(f"{'metric':28s} {'natural':>12s} {'spliced':>12s} {'delta':>12s}")
    lines.append("-" * 68)
    for key, label in _COMPARE_ROWS:
        n = natural.get(key)
        w = spliced.get(key)
        if n is None or w is None or (
                isinstance(n, float) and np.isnan(n)) or (
                isinstance(w, float) and np.isnan(w)):
            lines.append(f"{label:28s} {_fmt(n):>12s} {_fmt(w):>12s} "
                         f"{'n/a':>12s}")
        else:
            lines.append(f"{label:28s} {n:12.4f} {w:12.4f} {w - n:+12.4f}")
    return "\n".join(lines)


def _render_spliced(out_path, m, P, Q, gap, eps_b, eps_a, nullm_a, z_a, tau,
                    alphas, w0, w1, model_basename, sections):
    """The v1 3-panel form for a spliced run.

    Same honesty labels as the natural portrait (Systasis 'named, not
    claimed'; Chronos 'one sampling's reading'). Additions: the splice band
    shaded on all panels, the named wires annotated, the natural braid drawn
    thin for reference behind the spliced one, and the caption states the
    declared target plus the measured outcome.
    """
    T = P.shape[0]
    NB = P.shape[1]
    prov = m["provenance"]
    sp = m["splice"]
    D_total = m["D_total"]
    z_arrow = m["z_arrow"]
    alpha = sp["alpha"]

    fig = plt.figure(figsize=(16, 11), dpi=100)
    fig.patch.set_facecolor("white")
    gs = GridSpec(3, 1, height_ratios=[3.4, 1, 1], hspace=0.14)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax2 = fig.add_subplot(gs[2], sharex=ax0)

    fig.suptitle("Hodos \u2014 What an AI Looks Like While It Is Spliced",
                 fontsize=17, weight="bold", y=0.985)
    fig.text(0.5, 0.945,
             "Monitor \u2192 act \u2192 monitor: surgical splicing, named wires only",
             ha="center", fontsize=11)

    tt = np.arange(T)
    ax0.set_facecolor("#07070d")
    tex = np.log10(P.T + 1e-6)
    ax0.imshow(tex, extent=[0, T, 0, NB], origin="lower", aspect="auto",
               cmap="magma", alpha=0.55,
               vmin=np.percentile(tex, 5), vmax=np.percentile(tex, 99))
    ax0.set_ylabel("Activation (quantile bin)")

    def smooth(x, w=5):
        k = np.ones(w) / w
        return np.convolve(x, k, mode="same")

    binpos = np.arange(NB)
    modeA = smooth((P * binpos).sum(axis=1))
    modeB = smooth((Q * binpos).sum(axis=1))

    gn = (gap - gap.min()) / (gap.max() - gap.min() + 1e-12)
    hw = 0.55
    for i in range(T):
        ax0.fill_between([tt[i] - hw, tt[i] + hw], [modeA[i], modeA[i]],
                         [modeB[i], modeB[i]], color="#fde68a",
                         alpha=0.04 + 0.42 * gn[i], linewidth=0, zorder=3)
    ax0.plot(tt, modeA, color="#ffd166", lw=2.2, zorder=4, label="early layer")
    ax0.plot(tt, modeB, color="#4cc9f0", lw=2.2, zorder=4, label="late layer")

    for i in range(T):
        if z_a[i] > 1.0:
            ax0.axvspan(tt[i] - hw, tt[i] + hw, color="#ffcf5c",
                        alpha=min(0.30, 0.05 + 0.25 * (z_a[i] - 1) / 5), zorder=2, linewidth=0)

    for ax in (ax0, ax1, ax2):
        ax.axvspan(w0, w1, color="#22ff88", alpha=0.07, zorder=1, linewidth=0)
    wires_txt = ", ".join(
        f"{w['op']} {w['early_group']}:{w['late_unit']}"
        for w in sp["wires"]) or "none"
    ax0.text((w0 + w1) / 2, NB * 0.97, f"SPLICE \u03b1={alpha:g} [{wires_txt}]",
             color="#22ff88", fontsize=10, weight="bold", ha="center", va="top",
             zorder=6)

    for (t0, t1, label, color) in sections:
        ax0.text((t0 + t1) / 2, NB * 0.88, label, color=color, fontsize=11, weight="bold",
                 ha="center", va="center", zorder=5)
        if t0 > 0:
            for ax in (ax0, ax1, ax2):
                ax.axvline(t0, color=color, ls="--", lw=1, alpha=0.8, zorder=5)

    taps = prov.get("taps", {})
    stim = prov.get("stimulus_stimulus", prov.get("stimulus", "?"))
    ax0.text(0.02, 0.04,
             f"{prov.get('model_name', model_basename)} "
             f"({prov.get('model_kind', '?')}; "
             f"{prov.get('n_params', '?'):,}-param\n"
             f"stimulus {stim}, seed {prov.get('seed', '?')}).\n"
             f"Taps: {taps.get('early', '?')} vs {taps.get('late', '?')} "
             f"(n_pair={sp.get('n_pair', '?')}).\n"
             f"Target relation (declared before run): \u201c{sp['target_relation']}\u201d, "
             f"\u03b1={alpha:g} on steps {w0}\u2013{w1}.\n"
             f"Surgical: named wires only; untouched wires bit-identical; "
             f"rerouted wires keep identity + value multiset (timing only). "
             f"acc clean {_fmt(m.get('acc_clean'))}, degraded {_fmt(m.get('acc_degraded'))} (as measured).",
             transform=ax0.transAxes, color="white", alpha=0.65, fontsize=8, va="bottom")
    box = FancyBboxPatch((0.70, 0.04), 0.29, 0.13, transform=ax0.transAxes,
                         boxstyle="round,pad=0.02", facecolor="none", edgecolor="gray",
                         ls="--", alpha=0.7, linewidth=1.2)
    ax0.add_patch(box)
    ax0.text(0.845, 0.105, "Systasis: named, not claimed \u2014\nthe frame that couldn\u2019t be derived.",
             transform=ax0.transAxes, color="gray", fontsize=8.5, ha="center", va="center",
             style="italic", alpha=0.9)
    ax0.set_xlim(0, T)

    ax1.set_title("How Far Apart Are the Layers Over Time", fontsize=12, pad=6)
    ax1.fill_between(tt, gap, color="#f5a623", alpha=0.85, linewidth=0)
    ax1.plot(tt, gap, color="#b45309", lw=1.2)
    ax1.set_ylabel("Gap cost g(p,q)")
    ax1.text(0.99, 0.88, f"D(layers) = {D_total:.3f} (path-normalized)",
             transform=ax1.transAxes, ha="right", fontsize=9, color="#92400e")

    ax2.set_title("The Braid, Before and After Splicing", fontsize=12, pad=6)
    ax2.fill_between(tt, eps_a, color="#8e44ad", alpha=0.45, linewidth=0)
    ax2.plot(tt, eps_a, color="#6c3483", lw=1.4, label="\u03b5(t) spliced")
    ax2.plot(tt, eps_b, color="gray", lw=1.0, alpha=0.8, label="\u03b5(t) natural")
    ax2.plot(tt, nullm_a, color="gray", ls="--", lw=1, label="null")
    ax2.set_ylabel("\u03b5(t) = g(J, M)")
    ax2.legend(loc="upper left", fontsize=8, framealpha=0.7)
    ax2b = ax2.twinx()
    ax2b.plot(tt, tau, color="#b7950b", lw=1.6)
    ax2b.set_ylabel("Lived time \u03c4 (cumul.)", fontsize=9, color="#7d6608")
    ax2b.tick_params(labelcolor="#7d6608", labelsize=8)
    ax2.text(0.99, 0.06, "one sampling\u2019s reading", transform=ax2.transAxes, ha="right",
             fontsize=8, style="italic", color="gray")
    if abs(z_arrow) > 0.1:
        arrow_txt = "time\u2019s arrow \u2192" if z_arrow > 0 else "\u2190 time\u2019s arrow"
        ax2b.annotate(arrow_txt, xy=(tt[-1], tau[-1]), xytext=(-70, 12),
                      textcoords="offset points", fontsize=9, color="#7d6608",
                      arrowprops=dict(arrowstyle="->", color="#7d6608"))
    ax2.set_xlabel("Time (steps)")
    step = max(1, T // 6)
    ax2.set_xticks(list(range(0, T + 1, step)))
    ax2.set_xlim(0, T)

    fig.savefig(out_path, dpi=100)
    plt.close(fig)
