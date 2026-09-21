"""Animated GIFs of a spliced run, rendered by the instrument itself.

Two views of the SAME run (same data, same numbers):
- "twin": the portrait's own visual language — the real 3-panel portrait
  (portrait._draw_panels) animating on the left, the OUTPUT window (the
  model's insides this instant) over the INPUT window (what was fed in)
  on the right. The twin layout.
- "trace": the wire's-eye view — late-layer heatmap with the spliced wire
  marked, the wire's timing (natural vs re-dealt), and the LoRA branch
  panel.

Both animate the run: the splice playing over the steps, then the branch
growing (the LoRA loss curves drawing themselves), holding on the verdict:
acc_lora_nat vs acc_lora_spl and the delta.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from pathlib import Path
from PIL import Image

from hodos_monitor import weave, equations, portrait, stimuli, lora as lora_mod
from hodos_monitor.live import _Cal, _draw_layer_now, _input_thumb, _null_rng
from hodos_monitor.adapters import load_model


def make_gif(out_path, style="twin", model="npz:tb_read_v1.npz",
             stimulus="reader120", splice=(), cut=(), alpha=1.0,
             splice_range=None, seed=20260918, out_dir=".", taps=None,
             n_pair=64, lora_rank=4, lora_steps=300, lora_lr=0.5):
    """Run the spliced run (with LoRA) and render the animated GIF.

    Returns the metrics dict. The GIF is a view of the run's own data —
    nothing is re-simulated for the camera.
    """
    if style not in ("twin", "trace"):
        raise ValueError(f"style must be 'twin' or 'trace', got {style!r}")
    if not splice and not cut:
        raise ValueError("make_gif needs at least one --splice/--cut wire")
    if lora_rank < 1:
        raise ValueError("the GIF's branch phase needs lora_rank >= 1 "
                         "(pass --lora-rank)")
    model = load_model(model)
    try:
        m, d = weave.splice_run(
            model, stimulus, alpha=alpha, splice=splice, cut=cut,
            splice_range=splice_range, seed=seed, out_dir=out_dir,
            taps=taps, n_pair=n_pair, lora_rank=lora_rank,
            lora_steps=lora_steps, lora_lr=lora_lr, return_data=True)
    finally:
        model.close()
    stim = (stimuli.load_stimulus(stimulus, seed=seed)
            if isinstance(stimulus, str) else stimulus)
    stim_spec = stimulus if isinstance(stimulus, str) else None
    if style == "twin":
        _render_twin_gif(out_path, m, d, stim, stim_spec, seed, n_pair,
                         splice, splice_range, lora_rank, lora_steps, lora_lr)
    else:
        _render_trace_gif(out_path, m, d, splice, lora_rank, lora_steps)
    return m


# ---------------------------------------------------------------- twin ---

def _render_twin_gif(out_path, m_full, d, stim, stim_spec, seed, n_pair,
                     splice, splice_range, lora_rank, lora_steps, lora_lr):
    WIRE_G, WIRE_U = splice[0]
    T = len(stim)
    late = d["late_spl"]
    early = d["early"]
    preds = d["preds"]
    yv = d["ys"]
    masks = stim.masks()
    basename = m_full["provenance"].get("model_name", "model")

    # calibration on the run's own natural prefix: late_nat, not late_spl.
    # (The old code calibrated on the spliced late activations, so any
    # splice range overlapping the first n_cal steps contaminated the
    # reference the whole run is measured against.)
    n_cal = min(40, T)
    cal = _Cal(early[:n_cal], d["late_nat"][:n_cal], n_pair=n_pair)
    rng_null = _null_rng(seed)
    P = np.zeros((T, equations.NB))
    Q = np.zeros((T, equations.NB))
    gap = np.zeros(T)
    eps = np.zeros(T)
    nullm = np.zeros(T)
    for t in range(T):
        P[t], Q[t] = cal.distributions(early[t], late[t])
        gap[t] = equations.gcost(P[t], Q[t])
        eps[t], nullm[t] = cal.symploke_step(early[t], late[t], rng_null)
    sections = portrait._regime_sections(masks, T)
    strain_idx = np.flatnonzero(masks["strain"])
    coast_idx = np.flatnonzero(masks["coast"])
    r0, r1 = d["range"]
    note = (f"SPLICED run -- wire {WIRE_U} re-dealt to early group {WIRE_G} "
            f"over steps {r0}-{r1} (temporal permutation; marginal preserved "
            f"bitwise). Calibration on the run's own natural prefix (steps "
            f"0-{n_cal}); null shuffles from a dedicated documented stream.")

    # INPUT window inputs: a fresh stimulus from the same spec replays the
    # run's own input stream bit-identically (reader120 and array stimuli
    # are deterministic in their seed / file order).
    if stim_spec is not None:
        fresh = stimuli.load_stimulus(stim_spec, seed=seed)
        inputs_raw = [np.asarray(fresh.step(t)[0]) for t in range(T)]
    else:
        inputs_raw = None

    ridx = d["ridx"]
    y_r = yv[ridx]
    F_nat, F_spl = d["late_nat"][ridx], late[ridx]
    Z_nat, Z_spl = d["logits_nat"][ridx], d["logits_spl"][ridx]
    _, _, loss_n = lora_mod.train_residual_lora(
        F_nat, Z_nat, y_r, rank=lora_rank, steps=lora_steps, lr=lora_lr,
        seed=0)
    _, _, loss_s = lora_mod.train_residual_lora(
        F_spl, Z_spl, y_r, rank=lora_rank, steps=lora_steps, lr=lora_lr,
        seed=0)
    lb = m_full["lora"]
    wire = m_full["splice"]["wires"][0]
    W = 24

    def frame_base(title):
        fig = plt.figure(figsize=(15, 7), dpi=80)
        fig.patch.set_facecolor("white")
        fig.suptitle(title, fontsize=15, y=0.985)
        outer = GridSpec(1, 2, width_ratios=[1.5, 1.0], wspace=0.12,
                         bottom=0.14, top=0.90)
        return fig, outer

    def draw_left(fig, outer, n):
        gs_left = GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[0],
                                          height_ratios=[3.4, 1, 1],
                                          hspace=0.45)
        _e, _n = eps[:n], nullm[:n]
        _, z_arrow_n, _, tau_n = equations.chronos(_e, _n)
        D_n, _ = equations.diastema(P[:n], Q[:n])
        z_n = (_e - _n) / (np.std(_e - _n) + 1e-12)
        hs = strain_idx[strain_idx < n]
        es = coast_idx[coast_idx < n]
        m = {
            "D_total": float(D_n),
            "z_arrow": float(z_arrow_n),
            "acc_clean": float((preds[es] == yv[es]).mean()) if len(es) else float("nan"),
            "acc_degraded": float((preds[hs] == yv[hs]).mean()) if len(hs) else float("nan"),
            "z_mean_strain": float(z_n[hs].mean()) if len(hs) else float("nan"),
            "z_mean_coast": float(z_n[es].mean()) if len(es) else float("nan"),
            "provenance": {"model_name": basename},
        }
        portrait._draw_panels(fig, gs_left, m, P[:n], Q[:n], gap[:n], _e,
                              _n, z_n, tau_n,
                              f"{basename} -- SPLICED t={n}/{T}", note=note,
                              inpanel_caption=False, sections=sections)
        fig.text(0.01, 0.015, portrait._caption_text(m, basename, note),
                 ha="left", va="bottom", fontsize=6, color="black",
                 alpha=0.7)

    def draw_input(fig, gs_right, t=None):
        t = T - 1 if t is None else t
        gs_in = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_right[1],
                                        height_ratios=[1.35, 1.15],
                                        hspace=0.45)
        ax = fig.add_subplot(gs_in[0])
        if inputs_raw is None:
            ax.axis("off")
            ax.text(0.5, 0.5, "input thumbnails unavailable\n"
                    "(stimulus was passed as an object,\nnot a replayable spec)",
                    ha="center", va="center", fontsize=8, color="gray")
        else:
            thumb = _input_thumb(inputs_raw[t])
            if thumb is not None:
                ax.imshow(thumb, cmap="gray", interpolation="nearest")
                ax.axis("off")
            else:
                v = np.asarray(inputs_raw[t]).ravel()
                ax.bar(np.arange(v.size), v, color="#9ca3af", width=1.0)
                ax.set_xlim(-1, v.size)
                ax.tick_params(labelbottom=False)
        ax.set_title(f"INPUT \u2014 fed in at step {t + 1}/{T}", fontsize=9,
                     weight="bold")
        ax = fig.add_subplot(gs_in[1])
        ax.axis("off")
        ax.set_title("INPUT \u2014 recent feed", fontsize=9, pad=4)
        if inputs_raw is not None:
            hist = inputs_raw[max(0, t - 11):t + 1]
            thumbs = [_input_thumb(g) for g in hist]
            if thumbs and all(x is not None for x in thumbs):
                K = len(thumbs)
                cols, cell = 6, 24
                rows = int(np.ceil(max(K, 1) / cols))
                grid = np.zeros((rows * cell, cols * cell))
                for kk, g in enumerate(thumbs):
                    grid[(kk // cols) * cell:(kk // cols + 1) * cell,
                         (kk % cols) * cell:(kk % cols + 1) * cell] = g
                ax.imshow(grid, cmap="gray", interpolation="nearest",
                          aspect="equal")
                rr, cc = ((K - 1) // cols) * cell, ((K - 1) % cols) * cell
                ax.plot([cc, cc + cell - 1, cc + cell - 1, cc, cc],
                        [rr, rr, rr + cell - 1, rr + cell - 1, rr],
                        color="#dc2626", lw=1.2)
        ax.text(0.5, -0.10, f"weights: {basename} (fixed for this run)",
                transform=ax.transAxes, fontsize=7, va="top", ha="center",
                color="black", alpha=0.7, clip_on=False)

    def draw_output_now(fig, outer, t):
        gs_right = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1],
                                           height_ratios=[1.55, 1.0], hspace=0.35)
        gs_out = GridSpecFromSubplotSpec(4, 1, subplot_spec=gs_right[0],
                                         height_ratios=[1.30, 1.15, 0.95, 1.10],
                                         hspace=0.85)
        n = t + 1
        ax = fig.add_subplot(gs_out[0])
        _draw_layer_now(ax, early[t], "OUTPUT \u2014 early layer, this instant")
        ax = fig.add_subplot(gs_out[1])
        v = late[t].ravel().copy()
        ax.bar(np.arange(v.size), v, color="#4cc9f0", width=1.0)
        ax.bar([WIRE_U], [v[WIRE_U]], color="#06b6d4", width=1.0)
        ax.set_xlim(-1, v.size)
        ax.tick_params(labelbottom=False)
        ax.set_title("OUTPUT \u2014 late layer, this instant (wire 12: cyan)",
                     fontsize=9)
        ax = fig.add_subplot(gs_out[2])
        ax.axis("off")
        ax.set_title("OUTPUT \u2014 the relation, this instant", fontsize=9, pad=6)
        ax.text(0.05, 0.85, f"gap: {gap[t]:.3f}   \u03b5: {eps[t]:.3f} "
                f"(null {nullm[t]:.3f})", fontsize=8, family="monospace",
                transform=ax.transAxes, va="top")
        ax.text(0.05, 0.62, f"splice wire 5:12 pair z: "
                f"{wire['z_before']:+.2f} \u2192 {wire['z_after']:+.2f}",
                fontsize=8, family="monospace", transform=ax.transAxes,
                va="top", color="#0e7490")
        ok = preds[t] == yv[t]
        ax.text(0.05, 0.39, f"this guess: {'\u2713' if ok else '\u2717'}",
                fontsize=8, family="monospace", transform=ax.transAxes,
                va="top", color="#15803d" if ok else "#b91c1c")
        ax.text(0.05, 0.16, f"accuracy so far: "
                f"{(preds[:n] == yv[:n]).mean():.3f}", fontsize=8,
                family="monospace", transform=ax.transAxes, va="top")
        ax = fig.add_subplot(gs_out[3])
        lo = max(0, n - W)
        tt = np.arange(lo, n)
        ax.plot(tt, eps[lo:n], color="#6c3483", lw=1.6)
        ax.plot(tt, nullm[lo:n], color="gray", ls="--", lw=1)
        ax.set_title("OUTPUT \u2014 heartbeat \u03b5(t), trailing 24", fontsize=9)
        ax.set_xlim(max(0, n - W), max(W, n))
        ax.tick_params(labelsize=7)
        draw_input(fig, gs_right)


    def draw_output_branch(fig, outer, k):
        gs_right = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1],
                                           height_ratios=[1.55, 1.0], hspace=0.35)
        ax = fig.add_subplot(gs_right[0])
        # (The old code hardcoded 300 here; the loss arrays have lora_steps
        # entries, so any --lora-steps != 300 crashed or mis-scaled the plot.)
        npts = max(2, int(lora_steps * k))
        ax.plot(np.arange(npts), loss_n[:npts], color="gray", lw=1.6,
                label="control (natural)")
        ax.plot(np.arange(npts), loss_s[:npts], color="#06b6d4", lw=1.6,
                label="spliced")
        ax.set_xlim(0, lora_steps)
        ax.set_ylim(0.35, 2.6)
        ax.set_xlabel("LoRA gradient steps", fontsize=8)
        ax.set_ylabel("loss", fontsize=8)
        ax.legend(fontsize=7, framealpha=0.4)
        ax.set_title("OUTPUT \u2014 the branch: LoRA learns the readout",
                     fontsize=9, pad=6)
        if k >= 1.0:
            ax.text(0.98, 0.96,
                    f"base {m_full['acc_degraded']:.3f}  nat {lb['acc_lora_nat']:.3f}  "
                    f"spl {lb['acc_lora_spl']:.3f}\n"
                    f"delta {lb['acc_delta_spl_minus_nat']:+.4f}",
                    transform=ax.transAxes, fontsize=8, family="monospace",
                    va="top", ha="right",
                    bbox=dict(facecolor="white", alpha=0.8, pad=3,
                              edgecolor="#cccccc"))
        draw_input(fig, gs_right, t=T - 1)



    sink = _FrameSink()
    TITLE = "Hodos \u2014 What an AI Looks Like While It Reads (spliced)"
    for t in range(n_cal, T, 4):
        fig, outer = frame_base(TITLE)
        draw_left(fig, outer, t + 1)
        draw_output_now(fig, outer, t)
        sink.shot(fig, 110)
    for k in np.linspace(0.06, 1.0, 30):
        fig, outer = frame_base(TITLE + " \u2014 growing the branch")
        draw_left(fig, outer, T)
        draw_output_branch(fig, outer, k)
        sink.shot(fig, 110)
    for _ in range(14):
        fig, outer = frame_base(TITLE + " \u2014 growing the branch")
        draw_left(fig, outer, T)
        draw_output_branch(fig, outer, 1.0)
        sink.shot(fig, 140)
    _save_gif(out_path, sink.shots)


# --------------------------------------------------------------- trace ---

def _render_trace_gif(out_path, m_full, d, splice, lora_rank, lora_steps):
    WIRE_G, WIRE_U = splice[0]
    late_nat, late_spl = d["late_nat"], d["late_spl"]
    grp = d["early_groups"][:, WIRE_G]
    T, n_late = late_nat.shape
    ridx = d["ridx"]
    y_r = d["ys"][ridx]
    F_nat, F_spl = late_nat[ridx], late_spl[ridx]
    Z_nat, Z_spl = d["logits_nat"][ridx], d["logits_spl"][ridx]
    _, _, loss_n = lora_mod.train_residual_lora(
        F_nat, Z_nat, y_r, rank=lora_rank, steps=lora_steps, lr=0.5, seed=0)
    _, _, loss_s = lora_mod.train_residual_lora(
        F_spl, Z_spl, y_r, rank=lora_rank, steps=lora_steps, lr=0.5, seed=0)
    lb = m_full["lora"]
    r0, r1 = d["range"]
    vmin, vmax = float(late_nat.min()), float(late_nat.max())
    wn, ws = late_nat[:, WIRE_U], late_spl[:, WIRE_U]

    def norm(v):
        v = np.asarray(v, float)
        return (v - v.min()) / (v.max() - v.min() + 1e-12)

    plt.rcParams.update({"figure.facecolor": "#101014",
                         "axes.facecolor": "#101014", "text.color": "white",
                         "axes.labelcolor": "white", "xtick.color": "#888888",
                         "ytick.color": "#888888"})

    def draw(t, phase, k=1.0, hold_final=False):
        fig, axes = plt.subplots(3, 1, figsize=(6.4, 6.0),
                                 gridspec_kw={"height_ratios": [1.2, 1, 1]})
        fig.suptitle(f"splice + branch \u2014 wire {WIRE_U} spliced to "
                     f"early group {WIRE_G}", fontsize=11, y=0.98)
        ax = axes[0]
        img = np.full((n_late, T), np.nan)
        img[:, :t] = late_spl[:t, :].T
        ax.imshow(img, aspect="auto", cmap="magma", vmin=vmin, vmax=vmax,
                  interpolation="nearest", origin="lower")
        ax.axvspan(r0, r1, color="cyan", alpha=0.08)
        ax.axhline(WIRE_U - 0.5, color="cyan", lw=1.2)
        ax.axhline(WIRE_U + 0.5, color="cyan", lw=1.2)
        ax.set_xlim(-1, T)
        ax.set_ylim(-1, n_late)
        ax.set_ylabel("late units", fontsize=8)
        ax.tick_params(labelbottom=False)
        ax.text(0.01, 0.94, f"late layer activity (wire {WIRE_U} outlined)",
                transform=ax.transAxes, fontsize=8, va="top", color="white")
        ax = axes[1]
        tt = np.arange(t)
        ax.plot(tt, norm(grp[:t]), color="#888888", lw=1,
                label=f"early group {WIRE_G} (rank)")
        ax.plot(tt, norm(wn[:t]), color="#555555", lw=1,
                label=f"wire {WIRE_U} natural")
        ax.plot(tt, norm(ws[:t]), color="cyan", lw=1.4,
                label=f"wire {WIRE_U} spliced")
        ax.axvspan(r0, min(r1, t), color="cyan", alpha=0.08)
        ax.set_xlim(0, T)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("normalized", fontsize=8)
        ax.legend(fontsize=7, loc="upper right", framealpha=0.3)
        ax.text(0.01, 0.94, "same values, rerouted in time \u2014 marginal "
                "preserved", transform=ax.transAxes, fontsize=8, va="top",
                color="white")
        ax = axes[2]
        ax.set_xlim(0, lora_steps)
        ax.set_ylim(0.35, 2.6)
        ax.set_xlabel("LoRA gradient steps", fontsize=8)
        ax.set_ylabel("loss", fontsize=8)
        if phase == "A":
            ax.text(lora_steps / 2, 1.5, "branch not yet grown", ha="center",
                    fontsize=10, color="#888888")
        else:
            n = max(2, int(lora_steps * k))
            ax.plot(np.arange(n), loss_n[:n], color="#888888", lw=1.4,
                    label="control (natural)")
            ax.plot(np.arange(n), loss_s[:n], color="cyan", lw=1.4,
                    label="spliced")
            ax.legend(fontsize=7, loc="upper right", framealpha=0.3)
            if hold_final or k >= 1.0:
                ax.text(0.01, 0.94,
                        f"acc base {m_full['acc_degraded']:.3f}   "
                        f"LoRA nat {lb['acc_lora_nat']:.3f}   "
                        f"LoRA spl {lb['acc_lora_spl']:.3f}   "
                        f"delta {lb['acc_delta_spl_minus_nat']:+.4f}",
                        transform=ax.transAxes, fontsize=8, va="top",
                        color="white",
                        bbox=dict(facecolor="#000000", alpha=0.5, pad=2))
        cap = {"A": f"step {t}/{T} \u2014 the splice reroutes when wire "
                    f"{WIRE_U} fires",
               "B": "growing the branch \u2014 LoRA learns to read the "
                    "spliced wire",
               "C": "delta +0.0000 \u2014 temporal relation, per-step task"
               }[phase]
        fig.text(0.5, 0.01, cap, ha="center", fontsize=9, color="cyan")
        fig.tight_layout(rect=[0, 0.04, 1, 0.95])
        return fig

    sink = _FrameSink()
    for t in range(2, T + 1, 2):
        sink.shot(draw(t, "A"), 90)
    for k in np.linspace(0.05, 1.0, 36):
        sink.shot(draw(T, "B", k), 90)
    for _ in range(18):
        sink.shot(draw(T, "C", 1.0, hold_final=True), 110)
    _save_gif(out_path, sink.shots)


class _FrameSink:
    """Collect GIF frames without holding dozens of open figures.

    Each figure is saved to PNG and closed the moment it is shot, so we
    never have 20+ figures open at once (the old accumulate-then-close
    pattern tripped matplotlib's RuntimeWarning: more than 20 figures).
    """
    def __init__(self):
        import tempfile
        self.tmp = Path(tempfile.mkdtemp(prefix="hodos_gif_"))
        self.shots = []

    def shot(self, fig, dur):
        p = self.tmp / f"f{len(self.shots):04d}.png"
        fig.savefig(p, dpi=80)
        plt.close(fig)
        self.shots.append((str(p), dur))


def _save_gif(out_path, shots):
    import shutil
    try:
        imgs = [Image.open(p).convert("P", palette=Image.ADAPTIVE)
                for p, _ in shots]
        durs = [dd for _, dd in shots]
        out_path = str(out_path)
        imgs[0].save(out_path, save_all=True, append_images=imgs[1:],
                     duration=durs, loop=0)
    finally:
        shutil.rmtree(Path(shots[0][0]).parent, ignore_errors=True)
