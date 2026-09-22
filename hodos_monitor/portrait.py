"""One portrait: stimulus -> activations -> four equations -> PNG + metrics.json.

run(weights_path, stimulus_seed, out_dir):
  - loads the checkpoint (arch inferred from weight shapes, layers tapped by kind)
  - renders the reader120 stimulus with a single rng (v1 consumption order:
    stimulus first, then the Symploke null shuffles — required for v1 parity)
  - runs the four equations (verbatim), verification (report, never force)
  - writes portrait.png (same 3-panel form as v1) and metrics.json
  - returns the metrics dict
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
from hodos_monitor.models import load_checkpoint, param_count
from hodos_monitor.stimuli import HARD_START, HARD_END, T_READER120


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_model(model, stimulus, seed, out_dir, taps=None, n_pair=64):
    """Render the portrait for ANY model + stimulus. Returns metrics dict.

    model    : ModelAdapter, or a spec string for adapters.load_model
               ("npz:<path>", "torch:<file.py>:<attr>", or a bare .npz path).
    stimulus : Stimulus, or a spec string for stimuli.load_stimulus
               ("reader120", "array:<path>.npy", "array:<path>.npz").
    seed     : stimulus seed (used when stimulus is a spec string); also
               seeds the Symploke null draws when the stimulus has no rng.
    taps     : (early, late, output) tap names; default = adapter's choice.
    n_pair   : Symploke pairing count (default 64 = v1).
    """
    from hodos_monitor.adapters import load_model
    model = load_model(model)
    if isinstance(stimulus, str):
        stimulus = stimuli.load_stimulus(stimulus, seed=seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    T = len(stimulus)
    masks = stimulus.masks()
    early_tap, late_tap, out_tap = taps or model.default_taps()
    rng = getattr(stimulus, "rng", None)
    if rng is None:
        rng = np.random.default_rng(seed)

    early, late, preds, ys = [], [], [], []
    for t in range(T):
        x, y, _regime = stimulus.step(t)
        acts = model.forward(np.asarray(x)[None])
        early.append(np.asarray(acts[early_tap]).ravel())
        late.append(np.asarray(acts[late_tap]).ravel())
        preds.append(np.asarray(acts[out_tap]).ravel().argmax())
        ys.append(y)
    early = np.array(early)
    late = np.array(late)
    preds = np.array(preds)

    has_labels = ys[0] is not None
    strain_m = masks.get("strain")
    coast_m = masks.get("coast")
    strain_idx = np.flatnonzero(strain_m) if strain_m is not None else None
    coast_idx = np.flatnonzero(coast_m) if coast_m is not None else None
    if has_labels and strain_idx is not None and coast_idx is not None:
        yv = np.array([int(v) for v in ys])
        acc_clean = float((preds[coast_idx] == yv[coast_idx]).mean())
        acc_degraded = float((preds[strain_idx] == yv[strain_idx]).mean())
    else:
        acc_clean = acc_degraded = None

    P, Q = equations.layer_distributions(early, late)
    D_total, gap = equations.diastema(P, Q)
    eps, nullm, nulls, z = equations.symploke(early, late, rng, n_pair=n_pair)
    excess, z_arrow, Phi, tau = equations.chronos(eps, nullm)
    if strain_idx is not None and coast_idx is not None:
        ver = equations.verify(z, gap, tau, strain_idx, coast_idx)
        slope_strain = float(
            (tau[strain_idx[-1]] - tau[strain_idx[0]]) / len(strain_idx))
        slope_coast = float(
            ((tau[strain_idx[0] - 1] - tau[0] if strain_idx[0] > 0 else 0.0)
             + (tau[-1] - tau[strain_idx[-1] + 1]
                if strain_idx[-1] + 1 < T else 0.0)) / len(coast_idx))
    else:
        ver = {"passed": None, "status": "NO_REGIMES (see notes)",
               "notes": ["stimulus has no strain/coast regimes — "
                         "verification needs both"]}
        slope_strain = slope_coast = float("nan")

    model_desc = model.describe()
    stim_desc = stimulus.describe()
    metrics = {
        "D_total": float(D_total),
        "gap_min": float(gap.min()),
        "gap_max": float(gap.max()),
        "gap_std": float(gap.std()),
        "gap_mean": float(gap.mean()),
        "z_mean_strain": float(z[strain_idx].mean())
        if strain_idx is not None else float("nan"),
        "z_mean_coast": float(z[coast_idx].mean())
        if coast_idx is not None else float("nan"),
        "z_max_strain": float(z[strain_idx].max())
        if strain_idx is not None else float("nan"),
        "z_max_coast": float(z[coast_idx].max())
        if coast_idx is not None else float("nan"),
        "eps_mean": float(eps.mean()),
        "eps_max": float(eps.max()),
        "eps_min": float(eps.min()),
        "tau_end": float(tau[-1]),
        "tau_slope_strain": slope_strain,
        "tau_slope_coast": slope_coast,
        "z_arrow": float(z_arrow),
        "Phi": float(Phi),
        "acc_clean": acc_clean,
        "acc_degraded": acc_degraded,
        "verification": ver,
        "provenance": {
            **model_desc,
            **{f"stimulus_{k}": v for k, v in stim_desc.items()},
            "stimulus": stim_desc.get("stimulus"),
            "taps": {"early": early_tap, "late": late_tap,
                     "output": out_tap},
            "n_pair": int(n_pair),
            "seed": str(seed),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }

    sections = _regime_sections(masks, T)
    _render(out_dir / "portrait.png", metrics, P, Q, gap, eps, nullm, z, tau,
            model_desc.get("model_name", "model"), sections=sections)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=1)
    return metrics


def _regime_sections(masks, T):
    """[(t0, t1, label, color)] spans for the render, from regime masks."""
    palette = {"strain": "#ef4444", "coast": "#22d3ee"}
    sections = []
    for regime, m in masks.items():
        idx = np.flatnonzero(m)
        if len(idx) == 0:
            continue
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        for run in runs:
            t0, t1 = float(run[0]), float(run[-1]) + 1
            sections.append((t0, t1, f"{regime.upper()} INPUT",
                             palette.get(regime, "#a78bfa")))
    sections.sort()
    return sections


def run(weights_path, stimulus_seed, out_dir):
    """Legacy entry point: .npz checkpoint + reader120 stimulus.

    Bit-identical to the pre-adapter implementation on the same inputs:
    the stimulus rng is consumed in v1 order and the Symploke null draws
    come from it afterwards.
    """
    from hodos_monitor.adapters import load_model
    from hodos_monitor.stimuli import Reader120Stimulus
    model = load_model(f"npz:{weights_path}")
    stimulus = Reader120Stimulus(stimulus_seed)
    try:
        return run_model(model, stimulus, seed=stimulus_seed, out_dir=out_dir)
    finally:
        model.close()


def _fmt(v):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) \
        else f"{v:.3f}"


def _caption_text(m, weights_basename, note=None):
    prov = m["provenance"]
    n_params = prov.get("n_params")
    n_params_s = f"{n_params:,}-param" if isinstance(n_params, int) else "?"
    model_kind = prov.get("model_kind", "?")
    stim = prov.get("stimulus", "?")
    seed = prov.get("seed", prov.get("stimulus_seed", "?"))
    taps = prov.get("taps", {})
    early_s = taps.get("early", "?")
    late_s = taps.get("late", "?")
    n_pair = prov.get("n_pair", 64)
    sha = prov.get("weights_sha256", "")
    sha_s = f"sha256 {sha[:12]}…" if sha else "no stored weights"
    # A reader-demo / regime run keeps the caption verbatim; a connected dataset with
    # no planted caps gets a caption in its own terms, not the reader's.
    if str(stim).startswith("reader") or m.get("acc_clean") is not None:
        last = (f"Caps mark the INPUT (planted). acc clean {_fmt(m.get('acc_clean'))} "
                f"→ degraded {_fmt(m.get('acc_degraded'))}; "
                f"z strain {_fmt(m.get('z_mean_strain'))} vs coast {_fmt(m.get('z_mean_coast'))}.")
    else:
        last = "Every tapped part tracked over the connected data; nulls reported as loudly as positives."
    return (f"{prov.get('model_name', weights_basename)}: {n_params_s} {model_kind} "
            f"(weights: {weights_basename};\n"
            f"{sha_s}; stimulus {stim}, seed {seed}).\n"
            f"Taps: {early_s} vs {late_s}. "
            f"Joint: {n_pair} paired obs/step, 12×12, 8-shuffle null.\n"
            f"{last}"
            + (f"\n{note}" if note else ""))


def _draw_panels(fig, gs, m, P, Q, gap, eps, nullm, z, tau, weights_basename,
                 note=None, inpanel_caption=True, sections=None):
    """Draw the v1 3-panel portrait into an existing figure/GridSpec.

    Extracted from _render so the live twin view can embed the same panels
    beside its NOW column. Batch behavior is unchanged: _render builds the
    standard figure and calls this.

    Honesty labels kept verbatim: Systasis 'named, not claimed'; Chronos 'one
    sampling's reading'; section caps labeled as PLANTED input regimes; accuracy
    numbers as measured; verification reported, never forced.

    note: optional extra honesty line appended to the caption (used by the
    live instrument to document its streaming amendments). Batch runs pass
    None and render exactly the v1 form.

    inpanel_caption: when True (batch default) the provenance caption sits
    inside the heatmap panel as in v1. The live twin passes False and draws
    it at the figure bottom instead, where the narrower panels would overlap it.

    sections: [(t0, t1, label, color)] regime spans for the caps; default is
    the v1 reader120 spans.
    """
    T = P.shape[0]
    NB = P.shape[1]
    prov = m["provenance"]
    D_total = m["D_total"]
    z_arrow = m["z_arrow"]

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax2 = fig.add_subplot(gs[2], sharex=ax0)

    _is_reader = str(prov.get("stimulus", "")).startswith("reader")
    _mname = prov.get("model_name", weights_basename)
    if _is_reader:
        _title = "Hodos \u2014 What an AI Looks Like While It Reads"
        _sub = "Two layers, one mind: Time (left\u2192right) \u00d7 Activation (bottom\u2192top)"
    else:
        _title = f"Panaesthesis \u2014 {_mname}: a relational portrait"
        _sub = "Two tracked parts: Time (left\u2192right) \u00d7 activity (bottom\u2192top)"
    fig.suptitle(_title, fontsize=17, weight="bold", y=0.985)
    fig.text(0.5, 0.945, _sub, ha="center", fontsize=11)

    tt = np.arange(T)
    # ---- top panel: the portrait
    ax0.set_facecolor("#07070d")
    tex = np.log10(P.T + 1e-6)                       # early-layer activation mass over time
    ax0.imshow(tex, extent=[0, T, 0, NB], origin="lower", aspect="auto",
               cmap="magma", alpha=0.55,
               vmin=np.percentile(tex, 5), vmax=np.percentile(tex, 99))
    ax0.set_ylabel("Activation (quantile bin)")

    def smooth(x, w=5):
        k = np.ones(w) / w
        return np.convolve(x, k, mode="same")

    binpos = np.arange(NB)                          # y-axis is bin index: plot expected bin
    modeA = smooth((P * binpos).sum(axis=1))        # early-layer activation centroid
    modeB = smooth((Q * binpos).sum(axis=1))        # late-layer activation centroid

    gn = (gap - gap.min()) / (gap.max() - gap.min() + 1e-12)
    hw = 0.55
    for i in range(T):                               # Diastema gap band between the layers
        ax0.fill_between([tt[i] - hw, tt[i] + hw], [modeA[i], modeA[i]],
                         [modeB[i], modeB[i]], color="#fde68a",
                         alpha=0.04 + 0.42 * gn[i], linewidth=0, zorder=3)
    ax0.plot(tt, modeA, color="#ffd166", lw=2.2, zorder=4, label="early layer")
    ax0.plot(tt, modeB, color="#4cc9f0", lw=2.2, zorder=4, label="late layer")

    for i in range(T):                               # Symploke glow where the braid is high
        if z[i] > 1.0:
            ax0.axvspan(tt[i] - hw, tt[i] + hw, color="#ffcf5c",
                        alpha=min(0.30, 0.05 + 0.25 * (z[i] - 1) / 5), zorder=2, linewidth=0)

    # Section caps: the INPUT regimes (planted, ground truth of the demo) —
    # labeled as input, because the computed braid may NOT follow them.
    if sections is None:
        sections = [(0.0, float(HARD_START), "CLEAN INPUT", "#22d3ee"),
                    (float(HARD_START), float(HARD_END), "DEGRADED INPUT", "#ef4444"),
                    (float(HARD_END), float(T), "CLEAN INPUT", "#22d3ee")]
    for (t0, t1, label, color) in sections:
        ax0.text((t0 + t1) / 2, NB * 0.88, label, color=color, fontsize=11, weight="bold",
                 ha="center", va="center", zorder=5)
        if t0 > 0:
            for ax in (ax0, ax1, ax2):
                ax.axvline(t0, color=color, ls="--", lw=1, alpha=0.8, zorder=5)

    if inpanel_caption:
        ax0.text(0.02, 0.04, _caption_text(m, weights_basename, note),
                 transform=ax0.transAxes, color="white", alpha=0.65, fontsize=8,
                 va="bottom")
    box = FancyBboxPatch((0.70, 0.04), 0.29, 0.13, transform=ax0.transAxes,
                         boxstyle="round,pad=0.02", facecolor="none", edgecolor="gray",
                         ls="--", alpha=0.7, linewidth=1.2)
    ax0.add_patch(box)
    ax0.text(0.845, 0.105, "Systasis: named, not claimed \u2014\nthe frame that couldn\u2019t be derived.",
             transform=ax0.transAxes, color="gray", fontsize=8.5, ha="center", va="center",
             style="italic", alpha=0.9)
    ax0.set_xlim(0, T)

    # ---- middle panel: Diastema gap over time
    ax1.set_title("How Far Apart Are the Layers Over Time" if _is_reader
                  else "How Far Apart Are the Parts Over Time", fontsize=12, pad=6)
    ax1.fill_between(tt, gap, color="#f5a623", alpha=0.85, linewidth=0)
    ax1.plot(tt, gap, color="#b45309", lw=1.2)
    ax1.set_ylabel("Gap cost g(p,q)")
    ax1.text(0.99, 0.88, f"D(layers) = {D_total:.3f} (path-normalized)",
             transform=ax1.transAxes, ha="right", fontsize=9, color="#92400e")

    # ---- bottom panel: Symploke epsilon(t) + Chronos tau
    ax2.set_title("Do the Layers Come Together Under Strain" if _is_reader
                  else "Do the Parts Come Together Under Strain", fontsize=12, pad=6)
    ax2.fill_between(tt, eps, color="#8e44ad", alpha=0.45, linewidth=0)
    ax2.plot(tt, eps, color="#6c3483", lw=1.4, label="\u03b5(t)")
    ax2.plot(tt, nullm, color="gray", ls="--", lw=1, label="null")
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
    ax2.set_xlabel("Time (glyphs read)" if _is_reader else "Time (steps)")
    if _is_reader:
        ax2.set_xticks([0, 20, 40, 60, 80, 100, 120])
    ax2.set_xlim(0, T)


def _render(out_path, m, P, Q, gap, eps, nullm, z, tau, weights_basename,
            note=None, sections=None):
    """The v1 3-panel form, parameterized by measured metrics + provenance."""
    fig = plt.figure(figsize=(16, 11), dpi=100)
    fig.patch.set_facecolor("white")
    gs = GridSpec(3, 1, height_ratios=[3.4, 1, 1], hspace=0.14)
    _draw_panels(fig, gs, m, P, Q, gap, eps, nullm, z, tau, weights_basename,
                 note=note, sections=sections)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
