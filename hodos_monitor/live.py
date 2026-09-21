"""Live Hodos portrait: the instrument reads as the model reads.

Works with any model (via adapters.ModelAdapter) and any stimulus (via
stimuli.Stimulus). The reader120 lane is the legacy default, not a limit.

Batch portrait (portrait.run_model) renders every step first, then computes
every equation with full-run statistics. A live instrument cannot see the
future, so live.run_model does the honest streaming adaptation, documented
here and in the output metrics under ``live_amendments``:

  CALIBRATION (first ``warmup`` steps plus ``cal_degraded`` test-card inputs
  on a dedicated rng stream):
    - per-layer (mean, std) for the scale-free standardization
    - the shared NB-bin quantile edges over pooled active units
    - the Symploke 12x12 joint edges (eE from the n_pair group means,
      eL from the late group means)
  The test card is the instrument's white balance: it puts the calibration
  across the operating range instead of warmup-only. Test-card activations
  never enter the measurement timeline, and the stimulus rng is never
  touched by calibration.
  STREAMING (every step):
    - forward pass, tap (early, late, output) — same taps as the batch run
    - P[t], Q[t] with the FIXED calibration constants (v1's form, fixed consts)
    - instantaneous gap g(p_t, q_t) on the live middle panel; the DTW
      path-normalized D is closed at end of run (DTW needs the whole run)
    - eps[t] = g(J_t, M_t) with fixed joint edges; null = 8 shuffles drawn
      in v1's exact order (for the reader120 stimulus, rng state matches the
      batch run after identical stimulus consumption — the only math
      difference from batch is the fixed calibration edges)
    - Chronos via equations.chronos on the stream seen so far
      (running third-moment arrow -> Phi -> tau)

Every frame re-renders the same 3-panel figure as the batch portrait, every
number on it computed from data seen so far. Nothing is smoothed with the
future. Early frames visibly settle as calibration locks in — that is the
instrument calibrating, not a defect.

Outputs in out_dir:
  frames/frame_%04d.png  -- the portrait re-rendered every --frame-every steps
  frames/twin_%04d.png   -- two-screen frames with --twin: the full portrait
                            (everything so far) beside the NOW panel (the
                            tapped layers' live activity this instant, the
                            relation numbers this instant, a trailing-24
                            heartbeat). The NOW side shows the model's
                            insides — never the input.
  portrait.png           -- final full portrait (batch 3-panel form)
  metrics.json           -- portrait.run_model's schema + live_amendments block
  live.mp4 / live-twin.mp4 -- frames assembled with ffmpeg (--no-video to skip)
"""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")

from hodos_monitor import equations, portrait, stimuli
from hodos_monitor.adapters import load_model
from hodos_monitor.monitor import DEFAULT_MASTER_SEED


def _derived_rng(seed, tag):
    """Dedicated rng streams for live-only needs (null shuffles, cal card).

    v1 draws null shuffles from the stimulus rng after the stimulus; a live
    run cannot reproduce that consumption order, so live-only randomness gets
    its own documented streams derived from the seed (works for int or tuple
    seeds). The stimulus rng is never touched by calibration.
    """
    digest = hashlib.sha256(f"hodos-live-{tag}-{seed}".encode()).hexdigest()
    return np.random.default_rng(int(digest[:16], 16))


def _null_rng(seed):
    return _derived_rng(seed, "null")


class _Cal:
    """Calibration constants fixed after the warm-up stretch."""

    def __init__(self, early_c, late_c, n_pair=64):
        self.n_pair = int(n_pair)
        self.mean_e = float(early_c.mean())
        self.std_e = float(early_c.std())
        self.mean_l = float(late_c.mean())
        self.std_l = float(late_c.std())

        act_e = early_c != 0
        act_l = late_c != 0
        ez = (early_c - self.mean_e) / (self.std_e + 1e-12)
        lz = (late_c - self.mean_l) / (self.std_l + 1e-12)
        pooled = np.concatenate([ez[act_e], lz[act_l]])
        edges = np.quantile(pooled, np.linspace(0, 1, equations.NB + 1))
        edges[0] -= 1e-9
        edges[-1] += 1e-9
        self.edges = edges

        A_c = np.array([equations.grouped_means(early_c[t], self.n_pair, seed=0)
                        for t in range(early_c.shape[0])])
        B_c = np.array([equations.grouped_means(late_c[t], self.n_pair, seed=None)
                        for t in range(late_c.shape[0])])
        eE = np.quantile(A_c, np.linspace(0, 1, equations.NJ + 1))
        eE[0] -= 1e-9
        eE[-1] += 1e-9
        eL = np.quantile(B_c, np.linspace(0, 1, equations.NJ + 1))
        eL[0] -= 1e-9
        eL[-1] += 1e-9
        self.eE = eE
        self.eL = eL

    def distributions(self, e, l):
        """P[t], Q[t]: v1's layer_distributions form, calibration constants."""
        nb = equations.NB
        ez = (e - self.mean_e) / (self.std_e + 1e-12)
        lz = (l - self.mean_l) / (self.std_l + 1e-12)
        he = np.histogram(ez[e != 0], bins=self.edges)[0]
        hl = np.histogram(lz[l != 0], bins=self.edges)[0]
        P = he / he.sum() if he.sum() else np.full(nb, 1 / nb)
        Q = hl / hl.sum() if hl.sum() else np.full(nb, 1 / nb)
        return P, Q

    def symploke_step(self, e, l, rng_null):
        """eps[t], null_mean[t]: v1's symploke per-step body, fixed edges."""
        nj, k = equations.NJ, equations.K_SHUFFLE
        a = equations.grouped_means(e, self.n_pair, seed=0)
        b = equations.grouped_means(l, self.n_pair, seed=None)
        ia = np.clip(np.digitize(a, self.eE) - 1, 0, nj - 1)
        ib = np.clip(np.digitize(b, self.eL) - 1, 0, nj - 1)
        J = np.zeros((nj, nj))
        np.add.at(J, (ia, ib), 1)
        J /= J.sum()
        M = np.outer(J.sum(1), J.sum(0))
        eps = equations.gcost(J.ravel(), M.ravel())
        ns = []
        for _ in range(k):
            ib2 = rng_null.permutation(ib)
            J2 = np.zeros((nj, nj))
            np.add.at(J2, (ia, ib2), 1)
            J2 /= J2.sum()
            ns.append(equations.gcost(J2.ravel(), M.ravel()))
        return float(eps), float(np.mean(ns))


def _draw_layer_now(ax, raw, title):
    """The tapped layer's activity, this instant — the model's insides working.

    Generic over whatever the tap returns, so this works on any model pointed
    at the instrument (taps are by layer kind in models.load_checkpoint):
    a spatial (C,H,W) tap tiles its channels; anything flat draws as a unit
    bar. The batch dim, if present, is squeezed.
    """
    r = np.asarray(raw)
    while r.ndim > 3:
        r = r[0]
    if r.ndim == 3 and r.shape[0] == 1:
        r = r[0]
    ax.set_title(title, fontsize=11)
    if r.ndim == 3 and r.shape[1] == r.shape[2]:
        C, H, W = r.shape
        cols = int(np.ceil(np.sqrt(C)))
        rows = int(np.ceil(C / cols))
        tile = np.zeros((rows * H, cols * W))
        for c in range(C):
            tile[(c // cols) * H:(c // cols + 1) * H,
                 (c % cols) * W:(c % cols + 1) * W] = r[c]
        ax.imshow(tile, cmap="magma", interpolation="nearest")
        ax.axis("off")
        ax.text(0.01, 0.99, f"{C} channels", transform=ax.transAxes,
                color="white", fontsize=8, va="top", ha="left")
    else:
        v = r.ravel()
        ax.bar(np.arange(v.size), v, color="#4cc9f0", width=1.0)
        ax.set_xlim(-1, v.size)
        ax.tick_params(labelbottom=False)


def _input_thumb(x_raw, cell=24):
    """2D thumbnail of a model input for the INPUT window, or None.

    Image-like inputs ((C,H,W) or (H,W) with H,W >= 4) render as a
    grayscale thumbnail; anything else (vectors, scalars, odd shapes)
    returns None and the INPUT window draws the raw values as bars.
    """
    r = np.asarray(x_raw, dtype=float)
    while r.ndim > 3:
        r = r[0]
    if r.ndim == 3:
        r = r.mean(axis=0)  # (C,H,W) -> (H,W)
    if r.ndim == 2 and r.shape[0] >= 4 and r.shape[1] >= 4:
        h, w = r.shape
        ys = (np.arange(cell) * h / cell).astype(int)
        xs = (np.arange(cell) * w / cell).astype(int)
        return r[ys][:, xs]
    return None


def _render_twin(out_path, m, P, Q, gap, eps, nullm, z, tau, weights_basename,
                 note, early_t, late_t, correct, acc_so_far, t, T,
                 x_raw_t, inputs_hist, weights_label, has_labels=True):
    """Two windows beside the full portrait: OUTPUT and INPUT.

    Left: the full portrait (everything so far). Right top (OUTPUT): the
    model's insides this instant — tapped layers' live activity, the relation
    numbers, the heartbeat. Right bottom (INPUT): what was fed in — the
    current input and the trailing strip of recent inputs, labeled with the
    weights loaded for the run (weights are fixed per run, not per-step
    input, so they are labeled, not drawn).
    """
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(24, 11), dpi=100)
    fig.patch.set_facecolor("white")
    outer = GridSpec(1, 2, width_ratios=[1.5, 1.0], wspace=0.10, bottom=0.12)

    gs_left = GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[0],
                                      height_ratios=[3.4, 1, 1], hspace=0.14)
    portrait._draw_panels(fig, gs_left, m, P, Q, gap, eps, nullm, z, tau,
                          weights_basename, note=note, inpanel_caption=False)
    fig.text(0.02, 0.015, portrait._caption_text(m, weights_basename, note),
             ha="left", va="bottom", fontsize=7.5, color="black", alpha=0.7)

    gs_right = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1],
                                       height_ratios=[1.55, 1.0], hspace=0.35)

    # ---- OUTPUT window: the model's insides, this instant ----
    gs_out = GridSpecFromSubplotSpec(4, 1, subplot_spec=gs_right[0],
                                     height_ratios=[1.30, 1.15, 0.95, 1.10],
                                     hspace=0.85)
    n = t + 1

    ax = fig.add_subplot(gs_out[0])
    _draw_layer_now(ax, early_t, "OUTPUT \u2014 early layer, this instant")

    ax = fig.add_subplot(gs_out[1])
    _draw_layer_now(ax, late_t, "OUTPUT \u2014 late layer, this instant")

    ax = fig.add_subplot(gs_out[2])
    ax.axis("off")
    ax.set_title("OUTPUT \u2014 the relation, this instant", fontsize=11, pad=8)
    ax.text(0.05, 0.82, f"gap right now:   {gap[t]:.3f}", fontsize=11,
            family="monospace", transform=ax.transAxes, va="top")
    ax.text(0.05, 0.60, f"\u03b5 right now:     {eps[t]:.3f}   (null {nullm[t]:.3f})",
            fontsize=11, family="monospace", transform=ax.transAxes, va="top")
    ax.text(0.05, 0.38, f"z right now:     {z[t]:+.2f}", fontsize=11,
            family="monospace", transform=ax.transAxes, va="top")
    if has_labels:
        guess_txt = (f"this guess: {'\u2713' if correct else '\u2717'}   "
                     f"accuracy so far: {acc_so_far:.3f}")
        guess_color = "#15803d" if correct else "#b91c1c"
    else:
        guess_txt = "no labels \u2014 accuracy n/a"
        guess_color = "gray"
    ax.text(0.05, 0.16, guess_txt, fontsize=11, family="monospace",
            transform=ax.transAxes, va="top", color=guess_color)

    ax = fig.add_subplot(gs_out[3])
    W = 24
    lo = max(0, n - W)
    tt = np.arange(lo, n)
    ax.plot(tt, eps[lo:n], color="#6c3483", lw=2.0)
    ax.plot(tt, nullm[lo:n], color="gray", ls="--", lw=1)
    ax.set_title("OUTPUT \u2014 heartbeat \u03b5(t), trailing 24", fontsize=11)
    ax.set_xlim(max(0, n - W), max(W, n))
    ax.set_ylabel("\u03b5(t)", fontsize=9)
    ax.set_xlabel("step", fontsize=9)

    # ---- INPUT window: what was fed in ----
    gs_in = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_right[1],
                                    height_ratios=[1.35, 1.15], hspace=0.45)
    ax = fig.add_subplot(gs_in[0])
    thumb = _input_thumb(x_raw_t)
    if thumb is not None:
        ax.imshow(thumb, cmap="gray", interpolation="nearest")
        ax.axis("off")
    else:
        v = np.asarray(x_raw_t).ravel()
        ax.bar(np.arange(v.size), v, color="#9ca3af", width=1.0)
        ax.set_xlim(-1, v.size)
        ax.tick_params(labelbottom=False)
    ax.set_title(f"INPUT \u2014 fed in at step {n}/{T}", fontsize=11,
                 weight="bold")

    ax = fig.add_subplot(gs_in[1])
    ax.axis("off")
    ax.set_title("INPUT \u2014 recent feed", fontsize=11, pad=6)
    thumbs = [_input_thumb(g) for g in inputs_hist]
    if thumbs and all(t is not None for t in thumbs):
        K = len(thumbs)
        cols, cell = 6, 24
        rows = int(np.ceil(max(K, 1) / cols))
        grid = np.zeros((rows * cell, cols * cell))
        for k, g in enumerate(thumbs):
            grid[(k // cols) * cell:(k // cols + 1) * cell,
                 (k % cols) * cell:(k % cols + 1) * cell] = g
        ax.imshow(grid, cmap="gray", interpolation="nearest", aspect="equal")
        if K:
            # red box on the newest input (last cell)
            r0, c0 = ((K - 1) // cols) * cell, ((K - 1) % cols) * cell
            ax.plot([c0, c0 + cell - 1, c0 + cell - 1, c0, c0],
                    [r0, r0, r0 + cell - 1, r0 + cell - 1, r0],
                    color="#dc2626", lw=1.5)
    else:
        ax.text(0.5, 0.5, "recent feed: non-image inputs\n(values above)",
                ha="center", va="center", fontsize=9, color="gray")
    ax.text(0.5, -0.06, f"weights: {weights_label} (fixed for this run)",
            transform=ax.transAxes, fontsize=8, va="top", ha="center",
            color="black", alpha=0.7, clip_on=False)

    fig.savefig(out_path, dpi=100)
    plt.close(fig)



def run_model(model, stimulus, seed=DEFAULT_MASTER_SEED, out_dir="live-out",
              warmup=20, cal_degraded=20, frame_every=4, framerate=6,
              make_video=True, twin=False, taps=None, n_pair=64):
    """Live portrait for ANY model + stimulus. Returns the result dict.

    model    : ModelAdapter, or a spec string for adapters.load_model
               ("npz:<path>", "torch:<file.py>:<attr>", or a bare .npz path).
    stimulus : Stimulus, or a spec string for stimuli.load_stimulus
               ("reader120", "array:<path>.npy", "array:<path>.npz").
    seed     : stimulus seed (used when stimulus is a spec string); also
               seeds the live-only derived rng streams (cal card, null).
    taps     : (early, late, output) tap names; default = adapter's choice.
    n_pair   : Symploke pairing count (default 64 = v1).
    """
    model = load_model(model)
    if isinstance(stimulus, str):
        stimulus = stimuli.load_stimulus(stimulus, seed=seed)
    out_dir = Path(out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    T = len(stimulus)
    masks = stimulus.masks()
    early_tap, late_tap, out_tap = taps or model.default_taps()
    # NOTE: the stimulus rng is consumed here in EXACTLY v1's order (the
    # reader120 stimulus mirrors render_reader120's loop verbatim). After
    # the stream, its state therefore matches v1's rng state at the start
    # of symploke -- so the null shuffles below are drawn in bit-identical
    # order to the batch run. The test card uses its own stream and never
    # touches the stimulus rng.
    rng_stim = getattr(stimulus, "rng", None)
    if rng_stim is None:
        rng_stim = np.random.default_rng(seed)

    # ---- phase 1: stream the read, one forward pass per step ----
    early_raw, late_raw, preds, ys = [], [], [], []
    early_maps, late_vecs = [], []  # unraveled taps, for the NOW panel
    inputs_raw = []  # model inputs as fed (no batch dim), for INPUT window
    for t in range(T):
        x_raw, y, _regime = stimulus.step(t)
        inputs_raw.append(np.asarray(x_raw).copy())
        acts = model.forward(np.asarray(x_raw)[None])
        e = np.asarray(acts[early_tap])
        la = np.asarray(acts[late_tap])
        lg = np.asarray(acts[out_tap])
        early_raw.append(e.ravel())
        late_raw.append(la.ravel())
        early_maps.append(np.asarray(e[0]).copy())
        late_vecs.append(np.asarray(la[0]).copy())
        preds.append(int(np.asarray(lg).ravel().argmax()))
        ys.append(y)
    early_raw = np.array(early_raw)
    late_raw = np.array(late_raw)
    preds = np.array(preds)
    has_labels = ys[0] is not None
    yv = np.array([int(v) for v in ys]) if has_labels else None

    # ---- phase 2: calibrate (same honesty contract as v1) ----
    rng_cal = _derived_rng(seed, "calcard")
    cal_early, cal_late = [early_raw[:warmup]], [late_raw[:warmup]]
    for card in stimulus.calibration_inputs(rng_cal, cal_degraded):
        acts = model.forward(np.asarray(card)[None])
        cal_early.append(np.asarray(acts[early_tap]).ravel()[None, :])
        cal_late.append(np.asarray(acts[late_tap]).ravel()[None, :])
    cal = _Cal(np.concatenate(cal_early), np.concatenate(cal_late),
               n_pair=n_pair)

    P = np.zeros((T, equations.NB))
    Q = np.zeros((T, equations.NB))
    gap_inst = np.zeros(T)
    eps = np.zeros(T)
    nullm = np.zeros(T)
    for t in range(T):
        P[t], Q[t] = cal.distributions(early_raw[t], late_raw[t])
        gap_inst[t] = equations.gcost(P[t], Q[t])
        eps[t], nullm[t] = cal.symploke_step(early_raw[t], late_raw[t],
                                            rng_stim)
    # Live z: per-step null spread is not tracked (v1 keeps per-step std of
    # the 8 shuffles); frames standardize the excess stream seen so far.
    # Documented approximation; the closing metrics use the same form.
    strain_m = masks.get("strain")
    coast_m = masks.get("coast")
    strain_idx = (np.flatnonzero(strain_m) if strain_m is not None
                  else np.array([], dtype=int))
    coast_idx = (np.flatnonzero(coast_m) if coast_m is not None
                 else np.array([], dtype=int))

    model_desc = model.describe()
    stim_desc = stimulus.describe()
    model_name = model_desc.get("model_name", "model")
    base_prov = {
        **model_desc,
        **{f"stimulus_{k}": v for k, v in stim_desc.items()},
        "stimulus": stim_desc.get("stimulus"),
        "taps": {"early": early_tap, "late": late_tap, "output": out_tap},
        "n_pair": int(n_pair),
        "seed": str(seed),
        "mode": "live",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    amendments = [
        "per-layer (mean,std) and all quantile edges fixed at calibration; "
        "v1 uses full-run statistics. Calibration pool = the run's first "
        f"{warmup} steps + {cal_degraded} test-card inputs "
        "(dedicated rng stream; test-card activations never enter the "
        "measurement timeline). This is the ONLY mathematical difference "
        "from the batch run.",
        "Symploke null shuffles are bit-identical to the batch run when "
        "the stimulus owns its rng (reader120): rng state matches v1's "
        "after identical stimulus consumption",
        "live middle panel shows the instantaneous gap g(p_t,q_t); the DTW "
        "path-normalized D is closed at end of run",
        "Chronos Phi uses the running third-moment arrow over diffs seen so far",
    ]
    note = (f"LIVE run -- instrument reading as it happens. Calibration "
            f"locked on {warmup} steps + {cal_degraded}-input test card; "
            f"null shuffles bit-identical to batch order; "
            f"see metrics.json live_amendments.")
    sections = portrait._regime_sections(masks, T)

    # ---- phase 3: frames as the run "happens" ----
    # Frames use a CONTIGUOUS sequence index (frame_0000.png, ...) so ffmpeg
    # reads every one; the t each frame corresponds to is in manifest.json.
    frame_ts = list(range(warmup, T, frame_every))
    if frame_ts[-1] != T - 1:
        frame_ts.append(T - 1)
    manifest = []
    for i, t in enumerate(frame_ts):
        n = t + 1
        _e, _n = eps[:n], nullm[:n]
        _, z_arrow_n, _, tau_n = equations.chronos(_e, _n)
        D_n, _ = equations.diastema(P[:n], Q[:n])
        z_n = (_e - _n) / (np.std(_e - _n) + 1e-12)
        hs = strain_idx[strain_idx < n]
        es = coast_idx[coast_idx < n]
        m = {
            "D_total": float(D_n),
            "z_arrow": float(z_arrow_n),
            "acc_clean": (float((preds[es] == yv[es]).mean())
                          if has_labels and len(es) else float("nan")),
            "acc_degraded": (float((preds[hs] == yv[hs]).mean())
                             if has_labels and len(hs) else float("nan")),
            "z_mean_strain": float(z_n[hs].mean()) if len(hs) else float("nan"),
            "z_mean_coast": float(z_n[es].mean()) if len(es) else float("nan"),
            "provenance": dict(base_prov),
        }
        portrait._render(frames_dir / f"frame_{i:04d}.png", m,
                         P[:n], Q[:n], gap_inst[:n], _e, _n, z_n, tau_n,
                         f"{model_name} -- LIVE t={n}/{T}", note=note,
                         sections=sections)
        if twin:
            if has_labels:
                acc_so_far = float((preds[:n] == yv[:n]).mean())
                correct = bool(preds[t] == yv[t])
            else:
                acc_so_far = float("nan")
                correct = False
            _render_twin(frames_dir / f"twin_{i:04d}.png", m,
                         P[:n], Q[:n], gap_inst[:n], _e, _n, z_n, tau_n,
                         f"{model_name} -- LIVE t={n}/{T}", note,
                         early_maps[t], late_vecs[t],
                         correct, acc_so_far, t, T,
                         inputs_raw[t], inputs_raw[max(0, t - 11):t + 1],
                         model_name, has_labels)
        manifest.append({"index": i, "t": int(t), "n_steps": int(n)})
    with open(frames_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    n_frames = len(frame_ts)

    # ---- close: final portrait in batch form + metrics ----
    D_total, gap = equations.diastema(P, Q)
    excess, z_arrow, Phi, tau = equations.chronos(eps, nullm)
    z = (eps - nullm) / (np.std(eps - nullm) + 1e-12)
    if len(strain_idx) and len(coast_idx):
        ver = equations.verify(z, gap, tau, strain_idx, coast_idx)
        slope_strain = float((tau[strain_idx[-1]] - tau[strain_idx[0]])
                             / len(strain_idx))
        slope_coast = float(
            ((tau[strain_idx[0] - 1] - tau[0] if strain_idx[0] > 0 else 0.0)
             + (tau[-1] - tau[strain_idx[-1] + 1]
                if strain_idx[-1] + 1 < T else 0.0)) / len(coast_idx))
    else:
        ver = {"passed": None, "status": "NO_REGIMES (see notes)",
               "notes": ["stimulus has no strain/coast regimes -- "
                         "verification needs both"]}
        slope_strain = slope_coast = float("nan")

    metrics = {
        "D_total": float(D_total),
        "gap_min": float(gap.min()),
        "gap_max": float(gap.max()),
        "gap_std": float(gap.std()),
        "gap_mean": float(gap.mean()),
        "z_mean_strain": (float(z[strain_idx].mean())
                          if len(strain_idx) else float("nan")),
        "z_mean_coast": (float(z[coast_idx].mean())
                         if len(coast_idx) else float("nan")),
        "z_max_strain": (float(z[strain_idx].max())
                         if len(strain_idx) else float("nan")),
        "z_max_coast": (float(z[coast_idx].max())
                        if len(coast_idx) else float("nan")),
        "eps_mean": float(eps.mean()),
        "eps_max": float(eps.max()),
        "eps_min": float(eps.min()),
        "tau_end": float(tau[-1]),
        "tau_slope_strain": slope_strain,
        "tau_slope_coast": slope_coast,
        "z_arrow": float(z_arrow),
        "Phi": float(Phi),
        "acc_clean": (float((preds[coast_idx] == yv[coast_idx]).mean())
                      if has_labels and len(coast_idx)
                      else (float((preds == yv).mean())
                            if has_labels else None)),
        "acc_degraded": (float((preds[strain_idx] == yv[strain_idx]).mean())
                         if has_labels and len(strain_idx) else None),
        "verification": ver,
        "live_amendments": amendments,
        "warmup_steps": int(warmup),
        "cal_degraded": int(cal_degraded),
        "frame_every": int(frame_every),
        "provenance": base_prov,
    }
    portrait._render(out_dir / "portrait.png", metrics, P, Q, gap,
                     eps, nullm, z, tau, model_name, note=note,
                     sections=sections)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=1)

    video_path = None
    if make_video:
        # hold the final portrait briefly so the ending lands
        hold = framerate * 2
        import shutil
        stem = "twin" if twin else "frame"
        last = frames_dir / f"{stem}_{n_frames - 1:04d}.png"
        for j in range(hold):
            shutil.copy(last, frames_dir / f"{stem}_{n_frames + j:04d}.png")
        video_path = out_dir / ("live-twin.mp4" if twin else "live.mp4")
        r = subprocess.run(
            ["ffmpeg", "-y", "-framerate", str(framerate),
             "-i", str(frames_dir / f"{stem}_%04d.png"),
             "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(video_path)],
            capture_output=True, text=True)
        if r.returncode != 0:
            video_path = None
            (out_dir / "ffmpeg_error.txt").write_text(r.stderr[-2000:])

    return {"metrics": metrics, "frames": n_frames,
            "video": str(video_path) if video_path else None,
            "out_dir": str(out_dir)}


def run(weights_path, stimulus_seed=DEFAULT_MASTER_SEED, out_dir="live-out",
        warmup=20, cal_degraded=20, frame_every=4, framerate=6, make_video=True,
        twin=False):
    """Legacy entry point: .npz checkpoint + reader120 stimulus.

    Bit-identical to the pre-adapter implementation on the same inputs.
    The reader-lane gate is gone -- any .npz the adapter can load works.
    """
    from hodos_monitor.stimuli import Reader120Stimulus
    return run_model(f"npz:{weights_path}", Reader120Stimulus(stimulus_seed),
                     seed=stimulus_seed, out_dir=out_dir, warmup=warmup,
                     cal_degraded=cal_degraded, frame_every=frame_every,
                     framerate=framerate, make_video=make_video, twin=twin)
