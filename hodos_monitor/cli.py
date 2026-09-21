"""CLI.

Any model + any stimulus (v0.9.0):

python -m hodos_monitor portrait --model npz:X.npz [--replicates 3] [--master-seed S] --out DIR
python -m hodos_monitor portrait --model torch:mymodel.py:net --stimulus array:inputs.npz --taps h1,h2,out --out DIR
python -m hodos_monitor watch --incoming DIR --out DIR [--every 600] [--once] [--replicates 3] [--master-seed S]
python -m hodos_monitor compare --history history.jsonl --out DIR
python -m hodos_monitor splice --model npz:X.npz --splice 5:12 [--cut 3:44] [--alpha 1.0] [--splice-range 40 80] [--master-seed S] --out DIR [--compare-with NATURAL_METRICS_JSON]
python -m hodos_monitor live --model npz:X.npz [--twin] [--no-video] --out DIR

Model specs (--model):
  npz:<path>              numpy checkpoint (any Conv/Affine/MLP arch in the family)
  torch:<file.py>:<attr>  torch.nn.Module loaded from a python file
  <path>.npz              bare path = npz: (legacy --weights behavior)

Stimulus specs (--stimulus):
  reader120               the v1 120-glyph protocol (default)
  array:<path>.npy        inputs array, shape (N, ...) — no batch dim
  array:<path>.npz        inputs=..., optional labels=..., regime=...

--weights is kept as a legacy alias for --model npz:<weights>.
"""
import argparse
import json
from pathlib import Path

from hodos_monitor import (compare, family, field, intervene, live, monitor,
                           portrait, sites, stimuli, weave)
from hodos_monitor.monitor import DEFAULT_MASTER_SEED


def _fmt(v, prec=3):
    if v is None:
        return "n/a"
    try:
        if v != v:  # NaN
            return "n/a"
    except TypeError:
        pass
    return f"{v:.{prec}f}"


def _resolve_model(a):
    if a.model:
        return a.model
    if a.weights:
        return f"npz:{a.weights}"
    raise SystemExit("need --model or --weights")


def _parse_taps(s):
    if not s:
        return None
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3 or not all(parts):
        raise SystemExit("--taps takes three comma-separated tap names: "
                         "early,late,output")
    return tuple(parts)


def _add_model_args(p):
    p.add_argument("--model", default=None,
                   help="model spec: npz:<path> | torch:<file.py>:<attr> | "
                        "<path>.npz")
    p.add_argument("--weights", default=None,
                   help="legacy alias for --model npz:<weights>")
    p.add_argument("--stimulus", default="reader120",
                   help="stimulus spec: reader120 | array:<path>.npy | "
                        "array:<path>.npz")
    p.add_argument("--taps", default=None,
                   help="three comma-separated tap names: early,late,output "
                        "(default: the adapter's choice)")
    p.add_argument("--n-pair", type=int, default=64,
                   help="Symploke pairing count (default 64 = v1)")


def _cmd_portrait(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model_spec = _resolve_model(a)
    taps = _parse_taps(a.taps)
    rep_metrics = []
    for r in range(a.replicates):
        seed = stimuli.replicate_seed(a.master_seed, r)
        m = portrait.run_model(model_spec, a.stimulus, seed,
                               out / f"rep{r}", taps=taps, n_pair=a.n_pair)
        rep_metrics.append(m)
        print(f"rep{r}: D={m['D_total']:.4f} acc_clean={_fmt(m['acc_clean'])} "
              f"acc_degraded={_fmt(m['acc_degraded'])} "
              f"z_strain={_fmt(m['z_mean_strain'])} z_coast={_fmt(m['z_mean_coast'])} "
              f"tau_end={m['tau_end']:.4f} z_arrow={m['z_arrow']:+.3f} "
              f"ver={m['verification']['status']}")
    summary = {
        "model": model_spec,
        "stimulus": a.stimulus,
        "taps": list(taps) if taps else None,
        "n_pair": a.n_pair,
        "master_seed": str(a.master_seed),
        "replicates": a.replicates,
        "metrics": rep_metrics,
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(f"wrote {out}/rep*/portrait.png + metrics.json, summary.json")


def _cmd_watch(a):
    monitor.watch(a.incoming, a.out, every_s=a.every, once=a.once,
                  replicates=a.replicates, master_seed=a.master_seed)


def _cmd_compare(a):
    rows = []
    with open(a.history) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    compare.regenerate(rows, out)
    print(f"wrote {out}/TRENDS.md + trends.png from {len(rows)} checkpoints")


def _parse_wires(s):
    """Parse 'g:u,g:u' wire specs into [(g, u), ...]."""
    pairs = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            g, u = tok.split(":")
            pairs.append((int(g), int(u)))
        except ValueError:
            raise SystemExit(
                f"bad wire spec '{tok}' — use early_group:late_unit, e.g. 5:12")
    return pairs


def _cmd_splice(a, legacy=False):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if legacy:
        print("note: 'weave' is deprecated — the blanket rank-coupling weave "
              "was retired (from the head's perspective it was noise, not a "
              "relation). Use 'splice' with named wires instead.")
    model_spec = _resolve_model(a)
    taps = _parse_taps(a.taps)
    splice_range = tuple(a.splice_range) if a.splice_range else None
    splice = _parse_wires(a.splice) if a.splice else []
    cut = _parse_wires(a.cut) if a.cut else []
    # The master seed is used DIRECTLY as the stimulus seed, so a splice run
    # shares its stimulus bit-for-bit with replicate 0 of a natural portrait
    # run on the same master seed.
    if a.gif:
        # The GIF renderer runs the spliced run itself (it needs the raw
        # arrays), so this replaces the plain run above instead of doubling it.
        from hodos_monitor import splice_gif
        m = splice_gif.make_gif(a.gif, style=a.gif_style, model=model_spec,
                                stimulus=a.stimulus, splice=splice, cut=cut,
                                alpha=a.alpha, splice_range=splice_range,
                                seed=a.master_seed, out_dir=out, taps=taps,
                                n_pair=a.n_pair, lora_rank=a.lora_rank,
                                lora_steps=a.lora_steps, lora_lr=a.lora_lr)
        print(f"wrote animated GIF ({a.gif_style} view): {a.gif}")
    else:
        m = weave.splice_run(model_spec, a.stimulus, alpha=a.alpha,
                             splice=splice, cut=cut,
                             splice_range=splice_range, seed=a.master_seed,
                             out_dir=out, taps=taps, n_pair=a.n_pair,
                             target_relation=a.target_relation,
                             lora_rank=a.lora_rank, lora_steps=a.lora_steps,
                             lora_lr=a.lora_lr)
    sp = m["splice"]
    print(f"spliced run: target '{sp['target_relation']}', alpha={sp['alpha']}, "
          f"range={sp['splice_range']}")
    for w in sp["wires"]:
        print(f"  wire {w['op']} {w['early_group']}:{w['late_unit']}: "
              f"pair z {_fmt(w['z_before'])} -> {_fmt(w['z_after'])}, "
              f"wire drift TV {w['wire_drift_tv']:.6f}")
    print(f"  untouched wires bit-identical: {sp['untouched_wires_bit_identical']}")
    print(f"  z_mean_strain: {_fmt(sp['z_mean_strain_before'])} -> "
          f"{_fmt(sp['z_mean_strain_after'])} "
          f"(coast {_fmt(sp['z_mean_coast_before'])} -> {_fmt(sp['z_mean_coast_after'])})")
    lb = m.get("lora", {})
    if lb.get("enabled"):
        print(f"  LoRA readout (rank {lb['rank']}, {lb['steps']} steps): "
              f"loss {lb['loss_first_spl']:.3f} -> {lb['loss_last_spl']:.3f}")
        print(f"    acc_lora_nat={lb['acc_lora_nat']:.4f} "
              f"acc_lora_spl={lb['acc_lora_spl']:.4f} "
              f"delta={lb['acc_delta_spl_minus_nat']:+.4f}")
    print(f"  acc_clean={_fmt(m['acc_clean'])} acc_degraded={_fmt(m['acc_degraded'])} "
          f"D={m['D_total']:.4f} tau_end={m['tau_end']:.4f} "
          f"ver={m['verification']['status']}")
    if a.compare_with:
        with open(a.compare_with) as f:
            natural = json.load(f)
        print()
        print(weave.compare_runs(natural, m))
    print(f"wrote {out}/spliced_portrait.png + spliced_metrics.json + per_step.json")


def _cmd_field(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model_spec = _resolve_model(a)
    m = field.run_field(model_spec, a.stimulus, seed=a.master_seed, out_dir=out,
                        site_class=a.site_class, max_taps=a.max_taps,
                        n_pair=a.n_pair)
    print(f"relational field: {m['n_taps']} sites (class {m['site_class']})")
    print(f"  Diastema off-diagonal: mean {m['D_offdiag_mean']:.3f} "
          f"max {m['D_offdiag_max']:.3f}")
    print(f"  Symploke braid z: field mean {m['z_field_mean']:+.3f} "
          f"max {m['z_field_max']:+.3f} "
          f"(n_pair {m['symploke_meta']['n_pair_effective']})")
    print(f"wrote {out}/field.png + field_metrics.json")


def _cmd_family(a):
    import numpy as np
    from hodos_monitor.adapters import load_model
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model = load_model(_resolve_model(a), all_sites=True)
    names = [t for t in model.tap_names() if t != "output"]
    keep = [t for t in names if sites.classify(t) == a.site_class]
    if not keep:
        avail = ", ".join(sorted({sites.classify(t) for t in names}))
        raise SystemExit(f"no taps of class {a.site_class!r} on this model "
                         f"(available classes: {avail})")
    stim = (stimuli.load_stimulus(a.stimulus, seed=a.master_seed)
            if isinstance(a.stimulus, str) else a.stimulus)
    acts, _preds, _ys = field.collect_acts(model, stim, keep)
    fams, left_out = family.build_families(list(acts.keys()), a.families)
    rng = np.random.default_rng(a.master_seed)
    m = family.run_auto(acts, fams, rng, out, hist_path=a.hist,
                        n_pair=a.n_pair, pool=a.pool, seed=a.master_seed)
    model.close()
    n_l0 = sum(len(v) for v in m["L0_pairs"].values())
    print(f"family watcher: {len(fams)} families (class {a.site_class}), "
          f"pool={a.pool}, n_pair={m['n_pair_effective']}, {n_l0} pairs scored")
    if m["L1_lit"]:
        for row in m["L1_lit"]:
            print(f"  LIT {row['pair']}: {'; '.join(row['reasons'])}")
        if m["L2"]:
            for pair, d in m["L2"].items():
                top = d["top"][0] if d["top"] else "-"
                tz = d["top_z"][0] if d["top_z"] else float("nan")
                print(f"    drill {pair}: top {top} z={_fmt(tz)}")
    else:
        print("  no family pair lit — BELOW THE SCREEN'S RESOLUTION, "
              "not 'nothing there' (pooling can hide opposed relations)")
    if left_out:
        print(f"  {len(left_out)} tap(s) left out of all families")
    print(f"wrote {out}/metrics.json")


def _cmd_intervene(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model_spec = _resolve_model(a)
    ops = intervene.parse_target(a.level, a.select)
    m = intervene.intervene_run(
        model_spec, a.stimulus, level=a.level, select=a.select, op=a.op,
        driver=a.driver, alpha=a.alpha, splice_range=(tuple(a.range)
        if a.range else None), seed=a.master_seed, out_dir=out,
        site_class=a.site_class, max_taps=a.max_taps, n_pair=a.n_pair)
    ch = m["change"]
    print(f"intervene: op={ch['op']} level={ch['level']} select={ch['select']!r} "
          f"alpha={ch['alpha']} range={ch['range']}")
    print(f"  targeted {ch['n_targeted_units']} unit(s) across "
          f"{len(ch['targeted_taps'])} site(s); untouched sites bit-identical: "
          f"{ch['untouched_bit_identical']}")
    fb = m["field_before"]
    fa = m["field_after"]
    print(f"  WHOLE FIELD over {fb['n_taps']} sites — reorganization:")
    print(f"    Diastema off-diag mean {_fmt(fb['D_offdiag_mean'])} -> "
          f"{_fmt(fa['D_offdiag_mean'])} (delta {_fmt(m['field_delta']['D_offdiag_mean'])})")
    print(f"    Symploke field-z mean {_fmt(fb['z_field_mean'])} -> "
          f"{_fmt(fa['z_field_mean'])} (delta {_fmt(m['field_delta']['z_field_mean'])})")
    print(f"    largest single-relation shift: {_fmt(m['field_delta']['z_pair_max_abs'])} "
          f"at {m['field_delta']['z_pair_max_at']}")
    b = m["behavior"]
    print(f"  BEHAVIOR: acc_clean {_fmt(b['acc_clean_before'])} -> {_fmt(b['acc_clean_after'])}, "
          f"acc_degraded {_fmt(b['acc_degraded_before'])} -> {_fmt(b['acc_degraded_after'])}")
    print(f"    {b['reading']}")
    print(f"wrote {out}/intervene_metrics.json + field_before.png + field_after.png")


def _cmd_taps(a):
    from hodos_monitor.adapters import load_model
    model = load_model(_resolve_model(a), all_sites=not a.modules_only)
    names = model.tap_names()
    by_class = {}
    for t in names:
        if t == "output":
            continue
        by_class.setdefault(sites.classify(t), []).append(t)
    desc = model.describe()
    print(f"{desc.get('model_name','model')}: {desc.get('n_params','?'):,} params, "
          f"{len(names)} taps (all_sites={desc.get('all_sites', False)})")
    if desc.get("attn_config"):
        nh, nkv, hd = desc["attn_config"]
        print(f"  attention: {nh} heads, {nkv} kv-heads, head_dim {hd}")
    for cls in sorted(by_class):
        ex = by_class[cls][:3]
        print(f"  [{cls:>6}] {len(by_class[cls]):>4}  e.g. {', '.join(ex)}")
    model.close()


def _cmd_live(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model_spec = _resolve_model(a)
    taps = _parse_taps(a.taps)
    r = live.run_model(model_spec, a.stimulus, seed=a.master_seed,
                       out_dir=out, warmup=a.warmup,
                       cal_degraded=a.cal_degraded, frame_every=a.frame_every,
                       framerate=a.framerate, make_video=not a.no_video,
                       twin=a.twin, taps=taps, n_pair=a.n_pair)
    m = r["metrics"]
    print(f"live run: {r['frames']} frames in {out}/frames/")
    print(f"  D={m['D_total']:.4f} acc_clean={_fmt(m['acc_clean'])} "
          f"acc_degraded={_fmt(m['acc_degraded'])} "
          f"z_strain={_fmt(m['z_mean_strain'])} z_coast={_fmt(m['z_mean_coast'])} "
          f"tau_end={m['tau_end']:.4f} z_arrow={m['z_arrow']:+.3f} "
          f"ver={m['verification']['status']}")
    if r["video"]:
        print(f"  video: {r['video']}")
    else:
        print("  video: skipped (ffmpeg unavailable or --no-video)")
    print(f"wrote {out}/portrait.png + metrics.json (+ live_amendments)")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="hodos_monitor",
                                 description="Hodos model monitor: portrait any model, keep history, flag changes, weave relations.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("portrait", help="portrait one model (N replicates)")
    _add_model_args(p)
    p.add_argument("--replicates", type=int, default=3)
    p.add_argument("--master-seed", type=int, default=DEFAULT_MASTER_SEED)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=_cmd_portrait)

    w = sub.add_parser("watch", help="watch a directory for new .npz checkpoints")
    w.add_argument("--incoming", required=True)
    w.add_argument("--out", required=True)
    w.add_argument("--every", type=int, default=600)
    w.add_argument("--once", action="store_true",
                   help="single poll pass (for tests / dry runs)")
    w.add_argument("--replicates", type=int, default=3)
    w.add_argument("--master-seed", type=int, default=DEFAULT_MASTER_SEED)
    w.set_defaults(fn=_cmd_watch)

    c = sub.add_parser("compare", help="regenerate TRENDS.md + trends.png from a history file")
    c.add_argument("--history", required=True)
    c.add_argument("--out", required=True)
    c.set_defaults(fn=_cmd_compare)

    s = sub.add_parser("splice", help="splice a surgical relational intervention into a read: name wires, couple or cut them (monitor -> act -> monitor)")
    _add_model_args(s)
    s.add_argument("--splice", default=None, metavar="G:U,G:U",
                   help="wires to couple, as early_group:late_unit pairs "
                        "(e.g. --splice 5:12,3:44)")
    s.add_argument("--cut", default=None, metavar="G:U,G:U",
                   help="wires to decouple (seeded random re-dealing), as "
                        "early_group:late_unit pairs")
    s.add_argument("--alpha", type=float, default=1.0,
                   help="splice strength in [0, 1]; 0 = natural run")
    s.add_argument("--splice-range", nargs=2, type=int, default=None,
                   metavar=("START", "END"),
                   help="step range where alpha applies "
                        "(default: the stimulus's strain regime)")
    s.add_argument("--master-seed", type=int, default=DEFAULT_MASTER_SEED,
                   help="used directly as the stimulus seed")
    s.add_argument("--target-relation", default=weave.TARGET_RELATION,
                   help="target relation, declared before the run")
    s.add_argument("--lora-rank", type=int, default=0,
                   help="if > 0, train a residual LoRA readout on the "
                        "splice-range steps — once on natural features "
                        "(control), once on spliced — and report both "
                        "accuracies (default 0 = off)")
    s.add_argument("--lora-steps", type=int, default=300,
                   help="LoRA gradient steps")
    s.add_argument("--lora-lr", type=float, default=0.5,
                   help="LoRA learning rate")
    s.add_argument("--gif", default=None, metavar="PATH",
                   help="render an animated GIF of this run (same software, "
                        "two views via --gif-style)")
    s.add_argument("--gif-style", default="twin", choices=["twin", "trace"],
                   help="'twin': the portrait's own layout (3-panel "
                        "portrait + OUTPUT/INPUT); 'trace': the wire's-eye "
                        "view (heatmap, wire timing, branch panel). "
                        "Needs --lora-rank >= 1.")
    s.add_argument("--out", required=True)
    s.add_argument("--compare-with", default=None,
                   help="natural-run metrics.json for a side-by-side table")
    s.set_defaults(fn=_cmd_splice)

    # Deprecated alias: the blanket rank-coupling weave was retired — the
    # scramble read as noise downstream, not a relation. Maps to splice.
    v = sub.add_parser("weave", help="(deprecated: use splice) blanket weave, retired")
    _add_model_args(v)
    v.add_argument("--splice", default=None, metavar="G:U,G:U")
    v.add_argument("--cut", default=None, metavar="G:U,G:U")
    v.add_argument("--alpha", type=float, default=1.0)
    v.add_argument("--splice-range", nargs=2, type=int, default=None,
                   metavar=("START", "END"))
    v.add_argument("--master-seed", type=int, default=DEFAULT_MASTER_SEED)
    v.add_argument("--target-relation", default=weave.TARGET_RELATION)
    v.add_argument("--lora-rank", type=int, default=0)
    v.add_argument("--lora-steps", type=int, default=300)
    v.add_argument("--lora-lr", type=float, default=0.5)
    v.add_argument("--gif", default=None, metavar="PATH")
    v.add_argument("--gif-style", default="twin", choices=["twin", "trace"])
    v.add_argument("--out", required=True)
    v.add_argument("--compare-with", default=None)
    v.set_defaults(fn=lambda a: _cmd_splice(a, legacy=True))

    fp = sub.add_parser("field", help="relational field over EVERY internal site (per-head attention + residual stream + MLP neurons), not one early<->late pair")
    _add_model_args(fp)
    fp.add_argument("--master-seed", type=int, default=DEFAULT_MASTER_SEED)
    fp.add_argument("--site-class", default=None,
                    choices=["resid", "head", "mlp", "attn", "module", "all"],
                    help="which family of internal sites to relate "
                         "(default: residual stream if present, else all)")
    fp.add_argument("--max-taps", type=int, default=48,
                    help="cap on the number of sites related (relations are "
                         "O(N^2); evenly subsampled if the family is larger)")
    fp.add_argument("--out", required=True)
    fp.set_defaults(fn=_cmd_field)

    fm = sub.add_parser("family", help="family-scale watcher (Law of Two across two scales): group a layer's heads into families, watch family<->family (cheap, always-on), drill into member heads only where a family pair lights up")
    _add_model_args(fm)
    fm.add_argument("--master-seed", type=int, default=DEFAULT_MASTER_SEED)
    fm.add_argument("--site-class", default="head",
                    choices=["resid", "head", "mlp", "attn", "module"],
                    help="which sites become family members (default: attention heads)")
    fm.add_argument("--families", default="auto",
                    help="'auto' (group by layer index) or an explicit spec "
                         "'fam0=tapA,tapB;fam1=tapC,tapD'")
    fm.add_argument("--pool", default="concat", choices=["concat", "mean"],
                    help="how to pool a family's members into one family tap "
                         "(concat preserves most; mean can cancel opposed "
                         "relations — measured non-neutral)")
    fm.add_argument("--hist", default=None,
                    help="history .jsonl for the L1 band trigger (optional; "
                         "created/appended)")
    fm.add_argument("--out", required=True)
    fm.set_defaults(fn=_cmd_family)

    iv = sub.add_parser("intervene", help="change relations at ANY level (head/family/layer/whole) and watch the WHOLE field reorganize before vs after — the general multi-level actuator")
    _add_model_args(iv)
    iv.add_argument("--level", required=True,
                    choices=["head", "resid", "family", "layer", "module",
                             "class", "field"],
                    help="the level to change: one head (q/k/v OR z), a "
                         "residual stream (resid_pre/resid_post), a family/layer "
                         "of heads, a whole module, a whole site-class, or the "
                         "entire field")
    iv.add_argument("--select", default=None,
                    help="what to target: a head tap (…self_attn.q.h3 or "
                         "…z.h1), a residual tap (layers.2.resid_pre), a module "
                         "tap, a layer index, or a module-name suffix; omit at "
                         "--level field to take all attention projections")
    iv.add_argument("--op", default="cut", choices=["cut", "couple"],
                    help="'cut' decouples (random temporal re-deal — degrade); "
                         "'couple' imposes a relation toward --driver "
                         "(strengthen). Both preserve each unit's value "
                         "multiset; value-neutral by design")
    iv.add_argument("--driver", default=None,
                    help="for --op couple: the tap whose rank order the target "
                         "is coupled toward")
    iv.add_argument("--alpha", type=float, default=1.0,
                    help="change strength in [0,1]; 0 = natural run")
    iv.add_argument("--range", nargs=2, type=int, default=None,
                    metavar=("START", "END"),
                    help="step range where the change applies (default: the "
                         "stimulus's strain regime)")
    iv.add_argument("--site-class", default="head",
                    choices=["resid", "head", "mlp", "attn", "module", "all"],
                    help="which sites the WHOLE-FIELD before/after map relates "
                         "(default: attention heads)")
    iv.add_argument("--max-taps", type=int, default=48,
                    help="cap on sites in the before/after field map (O(N^2))")
    iv.add_argument("--master-seed", type=int, default=DEFAULT_MASTER_SEED,
                    help="used directly as the stimulus seed")
    iv.add_argument("--out", required=True)
    iv.set_defaults(fn=_cmd_intervene)

    tp = sub.add_parser("taps", help="list the tap taxonomy — proof of which internal sites are tapped")
    _add_model_args(tp)
    tp.add_argument("--modules-only", action="store_true",
                    help="show only raw module taps (no per-head/residual enrichment)")
    tp.set_defaults(fn=_cmd_taps)

    lv = sub.add_parser("live", help="live portrait: watch the relations as the model reads")
    _add_model_args(lv)
    lv.add_argument("--master-seed", type=int, default=DEFAULT_MASTER_SEED,
                    help="used directly as the stimulus seed")
    lv.add_argument("--out", required=True)
    lv.add_argument("--warmup", type=int, default=20,
                    help="calibration steps before frames start")
    lv.add_argument("--cal-degraded", type=int, default=20,
                    help="test-card inputs in the calibration pool")
    lv.add_argument("--frame-every", type=int, default=4,
                    help="re-render the portrait every N steps")
    lv.add_argument("--framerate", type=int, default=6,
                    help="video framerate for live.mp4")
    lv.add_argument("--no-video", action="store_true",
                    help="skip ffmpeg assembly, keep frames only")
    lv.add_argument("--twin", action="store_true",
                    help="two-screen frames: full portrait beside the NOW "
                         "panel (this step, this instant's relation, "
                         "trailing-24 heartbeat); video is live-twin.mp4")
    lv.set_defaults(fn=_cmd_live)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
