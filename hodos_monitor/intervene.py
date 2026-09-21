"""Multi-level relational actuator: change relations at ANY level, and watch the
WHOLE field reorganize before vs after.

The splicer (weave.py) changes ONE named wire between one layer pair and measures
that one pair. This module generalizes both halves:

  CHANGE at any level -- a single head, a family/layer of heads, a whole module,
  a whole site-class, or the entire field of attention projections. The same
  marginal-preserving actuator as the splicer (temporal re-deal) is applied to
  the targeted columns only; every other column AT THE INJECTION POINT runs
  bit-identical, so what you changed is exactly what you named. Everything
  DOWNSTREAM is free to move -- that motion is the measurement.

  WATCH the whole field, before AND after. Every change is paired with the full
  relational field (Diastema + Symploke over every internal site) on the natural
  pass and again on the changed pass. This is the fix for the neuron-rat trap:
  cut one relation and the OUTPUT may hold while the REST of the relations
  reroute to hold it up. Measuring only the output, or only the touched pair,
  hides that; measuring the whole field shows it.

Value-neutral by design: 'cut' decouples (random temporal re-deal -- degrade a
relation); 'couple' imposes one toward a driver (strengthen). Both preserve each
unit's exact value multiset -- only WHEN a unit fires changes. Better or worse is
the operator's call, reported as measured, never forced.

Level coverage (this build): head (q/k/v role), family / layer (all of a layer's
q/k/v heads), module (any hooked submodule output), class (every module whose
name ends with a suffix), field (all attention projections across all layers).
z-head and residual-stream injection are pre-hook targets and raise a clear
pointer to the module-level alternative -- they are the next addition, not a
silent gap.
"""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from hodos_monitor import field, sites, stimuli, weave

# a per-head tap:  <attn>.q.h3 / .k.h1 / .v.h0   (z.h* is a pre-hook target)
_HEAD_RE = re.compile(r"^(?P<attn>.+)\.(?P<role>[qkv])\.h(?P<h>\d+)$")
_HEADZ_RE = re.compile(r"^(?P<attn>.+)\.z\.h(?P<h>\d+)$")
_RESID_RE = re.compile(r"\.resid_(?:pre|mid|post)$")
_ROLE_MOD = {"q": "q_proj", "k": "k_proj", "v": "v_proj"}
_LAYER_RE = re.compile(r"(?:layers|blocks|h)[._]?(\d+)(?:\.|$)")


HONESTY = [
    "The actuator preserves each targeted unit's exact value multiset; only its "
    "timing is re-dealt (cut = random order, couple = a driver's rank order).",
    "At the injection point every NON-targeted column runs bit-identical; "
    "everything downstream is free to move, and that motion is the measurement.",
    "A whole-module / whole-field change is a GROSS intervention, not a surgical "
    "one -- it cannot be attributed to a single wire. The before/after field is "
    "what makes it readable.",
    "A relation that does not move is BELOW THIS INSTRUMENT'S RESOLUTION, never "
    "'nothing there' (separability rule).",
    "Behavior holding while the field reorganizes is the degeneracy result, not "
    "'the change did nothing' -- it is reported explicitly.",
]


def _module_taps(model):
    """The overrideable module taps (real hooked submodules, not derived/output)."""
    return [t for t in model.tap_names()
            if t != "output" and sites.classify(t) not in ("head", "resid")]


def _layer_of(tap):
    m = _LAYER_RE.search(tap)
    return int(m.group(1)) if m else None


def parse_target(level, select):
    """Light validation used by the CLI before a model is loaded.

    Returns a normalized (level, select) or raises SystemExit with a usage hint.
    Full resolution (which modules, which columns) happens in resolve_targets
    once the model's tap names + attention config are known.
    """
    if level in ("head", "module", "resid") and not select:
        raise SystemExit(f"--level {level} needs --select <tap name>")
    if level in ("family", "layer") and select is None:
        raise SystemExit(f"--level {level} needs --select <layer index>")
    if level == "class" and not select:
        raise SystemExit("--level class needs --select <module-name suffix>, "
                         "e.g. o_proj or mlp.down_proj")
    if level in ("family", "layer"):
        try:
            int(select)
        except (TypeError, ValueError):
            raise SystemExit(f"--level {level} --select must be a layer index, "
                             f"got {select!r}")
    return {"level": level, "select": select}


def resolve_targets(model, level, select):
    """Resolve (level, select) into a list of injection targets.

    Each target: {"module": tap, "inject": "out"|"prein", "cols": ndarray|None,
    "src": ("tap", name) | ("concat", [names]), "label": str}. "out" overrides
    the module OUTPUT (a forward hook); "prein" overrides its INPUT (a pre-hook)
    -- the route for residual-stream and z-head sites, which are module inputs,
    not outputs. "cols" selects flat columns of the source to re-deal
    (None = every column). "src" says how to read the natural source from a
    forward's acts (a single tap, or several concatenated along the last axis).
    """
    mods = _module_taps(model)
    modset = set(mods)
    attn_cfg = model.describe().get("attn_config")

    def _need(m):
        if m not in modset:
            raise SystemExit(f"{m!r} is not an overrideable module tap on this "
                             f"model; run `taps` to list them")
        return m

    if level == "head":
        mz = _HEADZ_RE.match(select or "")
        if mz:                                    # z-head: o_proj INPUT slice
            if attn_cfg is None:
                raise SystemExit("model exposes no attention config")
            n_heads, _, head_dim = attn_cfg
            attn, h = mz.group("attn"), int(mz.group("h"))
            module = _need(f"{attn}.o_proj")
            cols = np.arange(h * head_dim, (h + 1) * head_dim)
            return [{"module": module, "inject": "prein", "cols": cols,
                     "src": ("concat", [f"{attn}.z.h{i}" for i in range(n_heads)]),
                     "label": f"z-head {h} of {attn} (o_proj input)"}]
        m = _HEAD_RE.match(select or "")          # q/k/v head: proj OUTPUT slice
        if not m:
            raise SystemExit("--level head wants a q/k/v/z head tap like "
                             f"'model.layers.2.self_attn.q.h3', got {select!r}")
        if attn_cfg is None:
            raise SystemExit("model exposes no attention config")
        _, _, head_dim = attn_cfg
        role, h = m.group("role"), int(m.group("h"))
        module = _need(f"{m.group('attn')}.{_ROLE_MOD[role]}")
        cols = np.arange(h * head_dim, (h + 1) * head_dim)
        return [{"module": module, "inject": "out", "cols": cols,
                 "src": ("tap", module),
                 "label": f"{role}-head {h} of {m.group('attn')}"}]

    if level == "resid":
        if not (select and _RESID_RE.search(select)):
            raise SystemExit("--level resid wants a residual tap like "
                             "'layers.2.resid_pre' or '...resid_post'")
        block = _RESID_RE.sub("", select)
        _need(block)
        if select.endswith(".resid_mid"):
            raise SystemExit("resid_mid is an intra-block sum, not a module "
                             "boundary; use resid_pre (block input) or "
                             "resid_post (block output)")
        if select.endswith(".resid_pre"):
            return [{"module": block, "inject": "prein", "cols": None,
                     "src": ("tap", select),
                     "label": f"residual stream IN @ {block}"}]
        return [{"module": block, "inject": "out", "cols": None,
                 "src": ("tap", block),
                 "label": f"residual stream OUT @ {block}"}]

    if level == "module":
        _need(select)
        return [{"module": select, "inject": "out", "cols": None,
                 "src": ("tap", select), "label": f"module {select}"}]

    if level in ("family", "layer"):
        L = int(select)
        proj = [t for t in mods if _layer_of(t) == L
                and any(t.endswith("." + p) for p in _ROLE_MOD.values())]
        if not proj:
            raise SystemExit(f"no q/k/v projection modules found for layer {L}")
        return [{"module": t, "inject": "out", "cols": None, "src": ("tap", t),
                 "label": f"layer {L} :: {t.rsplit('.', 1)[-1]}"} for t in proj]

    if level == "class":
        hit = [t for t in mods if t.endswith(select)]
        if not hit:
            raise SystemExit(f"no module tap ends with {select!r}; "
                             f"try o_proj, q_proj, mlp.down_proj, ...")
        return [{"module": t, "inject": "out", "cols": None, "src": ("tap", t),
                 "label": f"class {select} :: {t}"} for t in hit]

    if level == "field":
        proj = [t for t in mods
                if any(t.endswith("." + p) for p in
                       ("q_proj", "k_proj", "v_proj", "o_proj"))]
        if not proj:
            raise SystemExit("no attention projection modules to relate as a "
                             "field on this model")
        return [{"module": t, "inject": "out", "cols": None, "src": ("tap", t),
                 "label": f"field :: {t}"} for t in proj]

    raise SystemExit(f"unknown level {level!r}")


def _redeal_columns(nat_flat, cols, ridx, op, alpha, driver_series, rng):
    """Return a copy of nat_flat (T, W) with `cols` re-dealt over `ridx`.

    op == 'cut'    : random temporal re-deal of each column over the range.
    op == 'couple' : re-deal each column into driver_series' rank order.
    alpha blends re-dealt with natural; at alpha=1 the marginal is exact.
    """
    spl = nat_flat.copy()
    for j in cols:
        natcol = nat_flat[ridx, j]
        if op == "cut":
            redealt = weave._temporal_cut(natcol, rng)
        else:  # couple
            redealt = weave._temporal_rank_splice(natcol, driver_series)
        spl[ridx, j] = (1.0 - alpha) * natcol + alpha * redealt
    return spl


def _acc(preds, ys, mask):
    if mask is None or not mask.any():
        return None
    yv = np.array([int(v) if v is not None else -1 for v in ys])
    m = mask & (yv >= 0)
    if not m.any():
        return None
    return float((np.asarray(preds)[m] == yv[m]).mean())


def _field_delta(before, after):
    """Compare two relational_field dicts computed on the SAME tap set/order."""
    Zb = np.asarray(before["Z_matrix"])
    Za = np.asarray(after["Z_matrix"])
    taps = before["taps"]
    delta = {
        "D_offdiag_mean": after["D_offdiag_mean"] - before["D_offdiag_mean"],
        "z_field_mean": after["z_field_mean"] - before["z_field_mean"],
        "z_field_max": after["z_field_max"] - before["z_field_max"],
    }
    if Zb.shape == Za.shape and Zb.size and Zb.shape[0] == len(taps):
        diff = np.abs(Za - Zb)
        np.fill_diagonal(diff, 0.0)
        i, j = np.unravel_index(int(np.argmax(diff)), diff.shape)
        delta["z_pair_max_abs"] = float(diff[i, j])
        delta["z_pair_max_at"] = f"{taps[i]} <-> {taps[j]}"
    else:
        delta["z_pair_max_abs"] = float("nan")
        delta["z_pair_max_at"] = "n/a (tap sets differ)"
    return delta


def _reading(behavior, field_delta, field_before, field_after, alpha):
    """Honest one-line interpretation -- the rat readout.

    'Field moved' is judged RELATIVE to the field's own magnitude, not an
    absolute cutoff: the same 0.443 that is nothing on a toy whose couplings run
    to ~1e10 is a large move on a real model whose couplings run to ~0.5. So the
    largest single-relation shift is compared to the field's own coupling scale,
    the mean-coupling change to its own mean, and the Diastema change to its own
    distance. (Absolute cutoffs, calibrated once on the toy, under-reported real
    models -- found on Qwen2.5-0.5B, 2026-09-20.)
    """
    if alpha == 0.0:
        return "alpha=0: natural run, no change applied."
    acc_moved = False
    for k in ("acc_clean", "acc_degraded"):
        b, a = behavior[f"{k}_before"], behavior[f"{k}_after"]
        if b is not None and a is not None and abs(a - b) > 0.05:
            acc_moved = True
    fb = field_before or {}
    z_scale = max(abs(fb.get("z_field_max", 0.0)),
                  abs(fb.get("z_field_mean", 0.0)) * 3.0, 1e-6)
    rel_pair = abs(field_delta.get("z_pair_max_abs", 0.0)) / z_scale
    rel_zmean = (abs(field_delta.get("z_field_mean", 0.0))
                 / max(abs(fb.get("z_field_mean", 0.0)), 1e-6))
    rel_D = (abs(field_delta.get("D_offdiag_mean", 0.0))
             / max(abs(fb.get("D_offdiag_mean", 0.0)), 1e-6))
    field_moved = rel_pair >= 0.5 or rel_zmean >= 0.3 or rel_D >= 0.05
    if field_moved and not acc_moved:
        return ("behavior HELD while the field REORGANIZED -- the degeneracy "
                "result: relations rerouted to hold the output up.")
    if field_moved and acc_moved:
        return "the change moved BOTH the field and behavior."
    if acc_moved and not field_moved:
        return ("behavior moved but the whole-field maps did not -- the effect "
                "is below this field's resolution (try a finer site-class).")
    return ("neither behavior nor the field moved beyond resolution -- BELOW "
            "THIS INSTRUMENT'S RESOLUTION, not 'nothing there'.")


def intervene_run(model, stimulus, level, select, op="cut", driver=None,
                  alpha=1.0, splice_range=None, seed=20260918, out_dir=".",
                  site_class="head", max_taps=48, n_pair=64):
    """Change relations at `level`/`select`, watching the whole field before/after.

    Returns the metrics dict (also written to out_dir/intervene_metrics.json)
    and renders field_before.png + field_after.png.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if op == "couple" and not driver:
        raise SystemExit("--op couple needs --driver <tap name> (the relation "
                         "to impose toward)")
    model = _load(model)
    if isinstance(stimulus, str):
        stimulus = stimuli.load_stimulus(stimulus, seed=seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = resolve_targets(model, level, select)
    field_keep = field.select_taps(model.tap_names(), site_class, max_taps)
    src_taps = set()
    for tg in targets:
        kind, ref = tg["src"]
        src_taps.update([ref] if kind == "tap" else ref)

    T = len(stimulus)
    masks = stimulus.masks()
    if splice_range is None:
        strain = masks.get("strain")
        if strain is None or not strain.any():
            raise SystemExit("no --range given and the stimulus has no strain "
                             "regime; pass --range START END")
        idx = np.flatnonzero(strain)
        splice_range = (int(idx[0]), int(idx[-1]) + 1)
    w0, w1 = int(splice_range[0]), int(splice_range[1])
    if not (0 <= w0 < w1 <= T):
        raise SystemExit(f"--range must be 0 <= start < end <= {T}, got {(w0, w1)}")
    ridx = np.arange(w0, w1)

    # ---- pass 1: natural. Capture the field sites (for field_before) + every
    # tap a target's source needs + (if coupling) the driver, in one loop. ----
    cap = {}
    preds_nat, ys, drv_seq = [], [], []
    fired = None
    for t in range(T):
        x, y, _r = stimulus.step(t)
        acts = model.forward(np.asarray(x)[None])
        if fired is None:
            fired = [k for k in field_keep if k in acts]
            if not fired:
                raise SystemExit("no field taps fired on step 0")
            for s in src_taps:
                if s not in acts:
                    raise SystemExit(f"source tap {s!r} did not fire on step 0")
            cap = {k: [] for k in (set(fired) | src_taps)}
        for k in cap:
            cap[k].append(np.asarray(acts[k]))
        if driver:
            if driver not in acts:
                raise SystemExit(f"driver tap {driver!r} did not fire")
            drv_seq.append(float(np.asarray(acts[driver]).mean()))
        out = acts.get("output")
        preds_nat.append(int(np.asarray(out).ravel().argmax())
                         if out is not None else -1)
        ys.append(y)

    field_before = field.relational_field(
        {k: np.stack([a.ravel() for a in cap[k]]) for k in fired},
        seed=int(seed), n_pair=n_pair)

    def _src_full(tg):
        kind, ref = tg["src"]
        if kind == "tap":
            flat = np.stack([a.ravel() for a in cap[ref]])
            natshape = np.asarray(cap[ref][0]).shape
        else:  # concat several taps along the last axis, per step (z-heads)
            flat = np.stack([np.concatenate([np.asarray(cap[r][i]).ravel()
                                             for r in ref]) for i in range(T)])
            natshape = np.concatenate([np.asarray(cap[r][0]) for r in ref],
                                      axis=-1).shape
        return flat, natshape

    # ---- build the changed per-step overrides for each target ----
    driver_series = np.asarray(drv_seq)[ridx] if driver else None
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    n_units = 0
    overrides_by_step = {int(t): {} for t in ridx}
    injection_ok = True
    for tg in targets:
        flat, natshape = _src_full(tg)
        W = flat.shape[1]
        cols = np.arange(W) if tg["cols"] is None else np.asarray(tg["cols"])
        cols = cols[(cols >= 0) & (cols < W)]
        n_units += len(cols)
        spl = (_redeal_columns(flat, cols, ridx, op, alpha, driver_series, rng)
               if (alpha > 0.0 and len(cols)) else flat)
        untouched = np.setdiff1d(np.arange(W), cols)
        if untouched.size and not np.array_equal(spl[:, untouched],
                                                 flat[:, untouched]):
            injection_ok = False
        # "out" -> override the module output; "prein" -> override its input
        key = tg["module"] if tg["inject"] == "out" else "pre:" + tg["module"]
        for t in ridx:
            overrides_by_step[int(t)][key] = spl[t].reshape(natshape)

    target_mods = sorted({t["module"] for t in targets})

    # ---- pass 2: the changed run. Override on the range steps; collect the
    # whole field again (the injected value shows through the derived taps). ----
    fld_seq2 = {k: [] for k in fired}
    preds_aft = []
    for t in range(T):
        x, _y, _r = stimulus.step(t)
        ov = overrides_by_step.get(int(t)) if alpha > 0.0 else None
        acts = model.forward(np.asarray(x)[None], overrides=ov or None)
        for k in fired:
            fld_seq2[k].append(np.asarray(acts[k]).ravel())
        out = acts.get("output")
        preds_aft.append(int(np.asarray(out).ravel().argmax())
                         if out is not None else -1)

    field_after = field.relational_field(
        {k: np.stack(v) for k, v in fld_seq2.items()},
        seed=int(seed), n_pair=n_pair)
    delta = _field_delta(field_before, field_after)

    strain_m, coast_m = masks.get("strain"), masks.get("coast")
    behavior = {
        "acc_clean_before": _acc(preds_nat, ys, coast_m),
        "acc_clean_after": _acc(preds_aft, ys, coast_m),
        "acc_degraded_before": _acc(preds_nat, ys, strain_m),
        "acc_degraded_after": _acc(preds_aft, ys, strain_m),
    }
    behavior["reading"] = _reading(behavior, delta, field_before, field_after,
                                   float(alpha))

    desc = model.describe()

    def _summ(fld):
        return {k: fld[k] for k in ("n_taps", "D_offdiag_mean", "D_offdiag_max",
                                    "z_field_mean", "z_field_max")}

    metrics = {
        "change": {
            "op": op, "level": level, "select": select, "alpha": float(alpha),
            "driver": driver, "range": [w0, w1],
            "targeted_taps": target_mods,
            "targets": [{"module": t["module"], "inject": t["inject"],
                         "cols": ("all" if t["cols"] is None
                                  else [int(c) for c in t["cols"]]),
                         "label": t["label"]} for t in targets],
            "n_targeted_units": int(n_units),
            "untouched_bit_identical": bool(injection_ok),
            "site_class_watched": site_class,
        },
        "field_before": _summ(field_before),
        "field_after": _summ(field_after),
        "field_delta": delta,
        "behavior": behavior,
        "honesty": HONESTY,
        "provenance": {
            **desc,
            "stimulus": stimulus.describe().get("stimulus", "?"),
            "seed": str(seed), "n_pair": int(n_pair),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }
    with open(out_dir / "intervene_metrics.json", "w") as f:
        json.dump(metrics, f, indent=1)

    _render(out_dir / "field_before.png", field_before,
            f"{desc.get('model_name','model')} -- field BEFORE change")
    _render(out_dir / "field_after.png", field_after,
            f"{desc.get('model_name','model')} -- field AFTER "
            f"{op} @ {level}:{select}")
    model.close()
    return metrics


def _render(path, fld, title):
    prefix = field._common_prefix(fld["taps"])
    labels = [field._short(t, prefix) for t in fld["taps"]]
    field._render_field(path, fld, labels, title, {})


def _load(spec):
    from hodos_monitor.adapters import load_model
    return load_model(spec, all_sites=True)
