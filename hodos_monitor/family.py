"""Family-scale watcher (Hodoscope, L0/L1/L2). New in 0.12.0.

Implements FAMILY-DESIGN.md: the Law of Two across two scales. Groups
attention heads into natural families (one layer's heads = one family),
watches family-to-family always on (cheap), and drills into member heads
only when a family pair lights up.

The Hodos equations are NOT touched — family taps are just wide taps fed
through equations.symploke / equations.diastema unchanged. All the new
code lives here, in the tap/adapter layer above the equations.

Honesty labels (carried from the portrait, plus family-specific ones):
- A family-scale z is the z of the AGGREGATE, not of any head pair inside.
- Pooling can hide a relation: opposite relations cancel under AVERAGING
  (mean pooling) — measured, not assumed, in test_family.py — so the
  family clock has false negatives BY CONSTRUCTION. It is a screen that
  says where to look, never a proof that a quiet pair is empty.
  "Did not light up" is reported as BELOW THIS SCREEN'S RESOLUTION.
- The pooling method (concat vs mean) is a measurement decision recorded
  in metrics.json, never silently made. Do not assume a pooling is
  neutral until tested — see test_family.py.
- Directionality (deliberate): L0 computes unordered family pairs
  (fi < fj in family order), not the full F×F matrix. For layer
  families the order is causal (lower layer = early), which is the
  meaningful direction; the L0 claim is "this pair lights up", and L2
  resolves which heads carry it. Full F×F is deferred until a use
  case needs the reverse direction.
- Measured sight line (test_family.py): the family screen sees
  pool-level relations aligned with its pairing; a sharp head-to-head
  relation that the head-scale equation catches can sit below the
  family screen's resolution. Pinned by test_exhaustive_agreement.
- Systasis stays ghosted (named, not claimed). No anchor approximation.
"""

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from hodos_monitor import equations

DEFAULT_Z_ABS = 1.0      # absolute |z| bar for L1 light-up
DEFAULT_SIGMA = 2.0      # history-band width (in std) for L1 light-up
MIN_HISTORY = 5          # polls of history before the band rule applies
DEFAULT_MAX_DRILL = 8    # cap on lit pairs drilled per poll (L2 budget)

_LAYER_PATTERNS = [
    re.compile(r"[Ll]ayers?[._-]?(\d+)"),
    re.compile(r"[Bb]locks?[._-]?(\d+)"),
    re.compile(r"[Ll]ayer(\d+)"),
    re.compile(r"^h(\d+)[._-]"),
]


# ---------------------------------------------------------------------------
# Family definition
# ---------------------------------------------------------------------------

def parse_families_spec(spec, tap_names):
    """Parse an explicit family spec: "fam0=a,b;fam1=c,d".

    Every named tap must exist; every tap should belong to exactly one
    family (unassigned taps are left out and reported, never silently
    bucketed).
    """
    families = {}
    seen = set()
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, _, members = chunk.partition("=")
        name = name.strip()
        members = [m.strip() for m in members.split(",") if m.strip()]
        if not name or not members:
            raise ValueError(f"bad family chunk: {chunk!r} "
                             f"(want name=tap1,tap2,...)")
        for m in members:
            if m not in tap_names:
                raise ValueError(f"family {name!r}: unknown tap {m!r}")
            if m in seen:
                raise ValueError(f"tap {m!r} assigned to two families")
            seen.add(m)
        families[name] = members
    left_out = [t for t in tap_names if t not in seen]
    return families, left_out


def auto_layer_families(tap_names):
    """Group taps into families by layer index parsed from tap names.

    One layer's taps = one family, ordered by layer number. Raises if the
    tap names carry no detectable layer structure — an arbitrary grouping
    would measure the binning, not a relation, so we refuse to guess.
    """
    buckets = {}
    for tap in tap_names:
        layer = None
        for pat in _LAYER_PATTERNS:
            m = pat.search(tap)
            if m:
                layer = int(m.group(1))
                break
        if layer is None:
            raise ValueError(
                f"tap {tap!r} carries no layer index — cannot auto-group. "
                f"Pass explicit --families instead of 'auto'.")
        buckets.setdefault(layer, []).append(tap)
    families = {f"layer{layer}": buckets[layer]
                for layer in sorted(buckets)}
    return families


def build_families(tap_names, spec="auto"):
    """Return (families, left_out). spec: "auto" or explicit mapping string."""
    if spec == "auto":
        return auto_layer_families(tap_names), []
    return parse_families_spec(spec, tap_names)


# ---------------------------------------------------------------------------
# Pooling: one family tap per family
# ---------------------------------------------------------------------------

def pool_family(taps, members, method="concat"):
    """Pool a family's member taps into one family tap (T, W).

    concat (default): member activations concatenated along the feature
        axis — preserves the most, no averaging assumed neutral.
    mean: average of member activations — requires equal widths.
    The chosen method is always recorded in metrics.json.
    """
    vecs = [np.asarray(taps[m]) for m in members]
    if method == "concat":
        return np.concatenate(vecs, axis=1)
    if method == "mean":
        widths = {v.shape[1] for v in vecs}
        if len(widths) != 1:
            raise ValueError("mean pooling needs equal member widths, got "
                             f"{sorted(widths)}")
        return np.mean(vecs, axis=0)
    raise ValueError(f"unknown pooling method: {method!r}")


# ---------------------------------------------------------------------------
# L0 — the always-on family clock
# ---------------------------------------------------------------------------

def family_field(taps, families, rng, n_pair=64, pool="concat"):
    """Run the family-to-family field. Returns (result, meta).

    result["pairs"][fi]["fj"] = {"z": pair z, "D": D_total, "eps_mean": ...}
    for fi before fj in family order (directed early->late by that order).
    equations.symploke / equations.diastema are called unchanged.
    """
    order = list(families)
    fam_taps = {f: pool_family(taps, families[f], method=pool) for f in order}
    widths = [fam_taps[f].shape[1] for f in order]
    n_pair_eff = max(1, min(n_pair, min(widths)))
    pairs = {}
    t0 = time.time()
    for i, fi in enumerate(order):
        pairs[fi] = {}
        for fj in order[i + 1:]:
            early, late = fam_taps[fi], fam_taps[fj]
            P, Q = equations.layer_distributions(early, late)
            D_total, gap = equations.diastema(P, Q)
            eps, nullm, nulls, z = equations.symploke(
                early, late, rng, n_pair=n_pair_eff)
            pairs[fi][fj] = {
                "z": float(z.mean()),
                "z_max": float(z.max()),
                "D": float(D_total),
                "gap_mean": float(gap.mean()),
                "eps_mean": float(eps.mean()),
            }
    meta = {
        "families": {f: list(m) for f, m in families.items()},
        "pool": pool,
        "n_pair_requested": int(n_pair),
        "n_pair_effective": int(n_pair_eff),
        "family_widths": {f: int(w) for f, w in zip(order, widths)},
        "runtime_s": time.time() - t0,
        "n_pairs": sum(len(v) for v in pairs.values()),
    }
    return {"pairs": pairs, "order": order}, meta


# ---------------------------------------------------------------------------
# History + L1 trigger
# ---------------------------------------------------------------------------

def read_history(hist_path):
    rows = []
    p = Path(hist_path)
    if p.exists():
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def append_history(hist_path, row):
    p = Path(hist_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(row) + "\n")


def detect_lit(field, history, z_abs=DEFAULT_Z_ABS, sigma=DEFAULT_SIGMA):
    """L1: which family pairs light up.

    A pair lights up when its family-scale signal crosses a bar:
    - |z| above the absolute threshold, or
    - |z| more than `sigma` std from that pair's own running history
      (needs MIN_HISTORY polls), or
    - D steps beyond its history band (same rule).
    Returns [(fi, fj, [reasons...]), ...] in field order.
    """
    hist_pairs = {}
    for row in history:
        for fi, fjs in row.get("pairs", {}).items():
            for fj, vals in fjs.items():
                hist_pairs.setdefault((fi, fj), {"z": [], "D": []})
                hist_pairs[(fi, fj)]["z"].append(vals["z"])
                hist_pairs[(fi, fj)]["D"].append(vals["D"])
    lit = []
    for fi, fjs in field["pairs"].items():
        for fj, vals in fjs.items():
            reasons = []
            z, D = vals["z"], vals["D"]
            if abs(z) > z_abs:
                reasons.append(f"|z|={abs(z):.2f} > {z_abs}")
            h = hist_pairs.get((fi, fj), {"z": [], "D": []})
            if len(h["z"]) >= MIN_HISTORY:
                mz, sz = float(np.mean(h["z"])), float(np.std(h["z"])) + 1e-12
                if abs(z - mz) > sigma * sz:
                    reasons.append(f"z {z:.2f} beyond {sigma}σ of history "
                                   f"(mean {mz:.2f}, std {sz:.2f})")
                mD, sD = float(np.mean(h["D"])), float(np.std(h["D"])) + 1e-12
                if abs(D - mD) > sigma * sD:
                    reasons.append(f"D {D:.3f} beyond {sigma}σ of history "
                                   f"(mean {mD:.3f}, std {sD:.3f})")
            if reasons:
                lit.append((fi, fj, reasons))
    return lit


# ---------------------------------------------------------------------------
# L2 — head-level drill-down for lit pairs only
# ---------------------------------------------------------------------------

def drill_down(taps, families, lit_pairs, rng, n_pair=64):
    """For each lit family pair, run the head<->head field on just the
    heads inside those two families. Same equations, restricted tap set."""
    out = {}
    for fi, fj, reasons in lit_pairs:
        members_i = list(families[fi])
        members_j = list(families[fj])
        widths = [np.asarray(taps[m]).shape[1]
                  for m in members_i + members_j]
        n_pair_eff = max(1, min(n_pair, min(widths)))
        head_pairs = {}
        for a in members_i:
            for b in members_j:
                eps, nullm, nulls, z = equations.symploke(
                    np.asarray(taps[a]), np.asarray(taps[b]),
                    rng, n_pair=n_pair_eff)
                head_pairs[f"{a}->{b}"] = {
                    "z": float(z.mean()),
                    "z_max": float(z.max()),
                }
        # headline: strongest head pair, or nothing above the bar
        ranked = sorted(head_pairs.items(),
                        key=lambda kv: abs(kv[1]["z"]), reverse=True)
        out[f"{fi}->{fj}"] = {
            "reasons": reasons,
            "n_pair_effective": int(n_pair_eff),
            "head_pairs": head_pairs,
            "top": [k for k, _ in ranked[:3]],
            "top_z": [v["z"] for _, v in ranked[:3]],
        }
    return out


# ---------------------------------------------------------------------------
# Auto pass: L0 -> history -> L1 -> L2 (capped)
# ---------------------------------------------------------------------------

HONESTY_NOTES = [
    "A family-scale z is the z of the aggregate, not of any head pair inside it.",
    ("Pooling can hide a relation: opposite relations cancel under averaging "
     "(mean pooling) — measured in test_family.py — so the family clock has "
     "false negatives BY CONSTRUCTION. It is a screen that says where to "
     "look, never a proof that a quiet pair is empty."),
    ("'Did not light up' means BELOW THIS SCREEN'S RESOLUTION, never "
     "'there is nothing there' (separability rule)."),
    "Pooling method is a measurement decision, recorded in metrics.json.",
    "Systasis: named, not claimed. No anchor approximation used.",
]


def run_auto(taps, families, rng, out_dir, hist_path=None, n_pair=64,
             pool="concat", z_abs=DEFAULT_Z_ABS, sigma=DEFAULT_SIGMA,
             max_drill=DEFAULT_MAX_DRILL, seed=None):
    """One full auto pass. Returns metrics dict and writes metrics.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = read_history(hist_path) if hist_path else []
    field, meta = family_field(taps, families, rng, n_pair=n_pair, pool=pool)
    lit = detect_lit(field, history, z_abs=z_abs, sigma=sigma)
    drilled = lit[:max_drill]
    l2 = drill_down(taps, families, drilled, rng,
                    n_pair=n_pair) if drilled else {}
    metrics = {
        "scale": "family-auto",
        "pool": pool,
        "families": meta["families"],
        "n_pair_requested": meta["n_pair_requested"],
        "n_pair_effective": meta["n_pair_effective"],
        "family_widths": meta["family_widths"],
        "L0_pairs": field["pairs"],
        "L0_runtime_s": meta["runtime_s"],
        "L1_lit": [{"pair": f"{fi}->{fj}", "reasons": r}
                   for fi, fj, r in lit],
        "L1_capped": len(lit) > max_drill,
        "L2": l2,
        "history_polls_before": len(history),
        "honesty": HONESTY_NOTES,
        "provenance": {
            "seed": str(seed),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }
    if hist_path:
        append_history(hist_path, {
            "ts": metrics["provenance"]["timestamp"],
            "pairs": field["pairs"],
            "pool": pool,
            "n_pair_effective": meta["n_pair_effective"],
        })
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=1)
    return metrics
