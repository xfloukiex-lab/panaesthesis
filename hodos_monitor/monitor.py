"""The monitor: watch a directory for new checkpoints, portrait each, keep history.

watch(incoming_dir, out_dir, every_s=600, once=False, replicates=3,
      master_seed=20260918):
  - polls incoming_dir for *.npz files not yet in out_dir/state.json
  - for each new checkpoint: runs `replicates` portraits (replicate seeds from
    the master seed; replicate 0 == master seed), one row per checkpoint
    appended to history.jsonl (checkpoint means + per-replicate metrics)
  - regenerates TRENDS.md + trends.png from the full history
  - evaluates the flag rules -> FLAGS.md + stdout:
      (a) any verification FLAGGED on the newest checkpoint
      (b) z_mean_strain deviating >2 sigma from history (needs >=2 priors)
      (c) acc drop >0.05 vs the previous checkpoint (clean and degraded, separately)
"""
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from hodos_monitor import compare, portrait, stimuli

DEFAULT_MASTER_SEED = 20260918
MEAN_KEYS = [
    "D_total", "gap_min", "gap_max", "gap_std", "gap_mean",
    "z_mean_strain", "z_mean_coast", "z_max_strain", "z_max_coast",
    "eps_mean", "eps_max", "eps_min",
    "tau_end", "tau_slope_strain", "tau_slope_coast",
    "z_arrow", "Phi", "acc_clean", "acc_degraded",
]


def _read_history(hist_path):
    rows = []
    if hist_path.exists():
        with open(hist_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _aggregate_row(weights_path, rep_metrics, master_seed, replicates):
    means = {k: sum(m[k] for m in rep_metrics) / len(rep_metrics) for k in MEAN_KEYS}
    prov0 = rep_metrics[0]["provenance"]
    return {
        "checkpoint": Path(weights_path).stem,
        "weights_path": str(weights_path),
        "sha256": prov0["weights_sha256"],
        "arch_name": prov0["arch_name"],
        "family": prov0["family"],
        "n_params": prov0["n_params"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "master_seed": str(master_seed),
        "replicates": replicates,
        "metrics_mean": means,
        "metrics_replicates": rep_metrics,
        "verification_any_flagged": any(
            m["verification"]["status"] != "PASSED" for m in rep_metrics),
    }


def evaluate_flags(rows):
    """Flag rules. Returns a list of flag dicts for the newest checkpoint."""
    flags = []
    if not rows:
        return flags
    cur = rows[-1]
    priors = rows[:-1]
    ckpt = cur["checkpoint"]

    # (a) any verification FLAGGED
    for i, m in enumerate(cur["metrics_replicates"]):
        if m["verification"]["status"] != "PASSED":
            flags.append({
                "rule": "verification",
                "checkpoint": ckpt,
                "detail": f"replicate {i}: {m['verification']['status']} — "
                          + "; ".join(m["verification"]["notes"]),
            })

    # (b) z_mean_strain deviating >2 sigma from history (needs >=2 priors)
    if len(priors) >= 2:
        xs = [r["metrics_mean"]["z_mean_strain"] for r in priors]
        mu = statistics.fmean(xs)
        sd = statistics.pstdev(xs)
        x = cur["metrics_mean"]["z_mean_strain"]
        if sd < 1e-12:
            deviant = abs(x - mu) > 1e-9
        else:
            deviant = abs(x - mu) > 2 * sd
        if deviant:
            flags.append({
                "rule": "z_mean_strain>2sigma",
                "checkpoint": ckpt,
                "detail": f"z_mean_strain={x:.3f} vs prior mean={mu:.3f} sd={sd:.4f} "
                          f"over {len(priors)} prior checkpoints",
            })

    # (c) acc drop >0.05 vs previous checkpoint
    if priors:
        prev = priors[-1]
        for key in ("acc_clean", "acc_degraded"):
            drop = prev["metrics_mean"][key] - cur["metrics_mean"][key]
            if drop > 0.05:
                flags.append({
                    "rule": "acc_drop>0.05",
                    "checkpoint": ckpt,
                    "detail": f"{key}: {prev['metrics_mean'][key]:.3f} -> "
                              f"{cur['metrics_mean'][key]:.3f} (drop {drop:.3f}) "
                              f"vs previous checkpoint {prev['checkpoint']}",
                })
    return flags


def _render_flags_md(flags):
    L = ["# Hodos monitor — flags\n"]
    if not flags:
        L.append("No flags.\n")
    else:
        for fl in flags:
            L.append(f"- **{fl['rule']}** @ {fl['checkpoint']}: {fl['detail']}")
        L.append("")
    return "\n".join(L)


def watch(incoming_dir, out_dir, every_s=600, once=False, replicates=3,
          master_seed=DEFAULT_MASTER_SEED):
    incoming_dir = Path(incoming_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "state.json"
    hist_path = out_dir / "history.jsonl"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"processed": {}}

    while True:
        new_files = sorted(p for p in incoming_dir.glob("*.npz")
                           if p.name not in state["processed"])
        for weights_path in new_files:
            ckpt_dir = out_dir / "checkpoints" / weights_path.stem
            try:
                rep_metrics = []
                for r in range(replicates):
                    seed = stimuli.replicate_seed(master_seed, r)
                    m = portrait.run(weights_path, seed, ckpt_dir / f"rep{r}")
                    rep_metrics.append(m)
                    print(f"  portrait done: {weights_path.stem} rep{r} "
                          f"D={m['D_total']:.4f} acc_c={m['acc_clean']:.3f} "
                          f"acc_d={m['acc_degraded']:.3f} "
                          f"z_strain={m['z_mean_strain']:.3f} "
                          f"ver={m['verification']['status']}", flush=True)
                row = _aggregate_row(weights_path, rep_metrics, master_seed, replicates)
                with open(hist_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
                state["processed"][weights_path.name] = {
                    "sha256": row["sha256"],
                    "at": row["timestamp"],
                }
                state_path.write_text(json.dumps(state, indent=1))
                print(f"checkpoint recorded: {weights_path.stem}", flush=True)
            except Exception as e:
                # Honest skip: the monitor can't portrait this checkpoint
                # (e.g. input/task mismatch). Record the reason and keep
                # watching the rest instead of crashing the whole pass.
                err = f"{type(e).__name__}: {e}"
                state.setdefault("skipped", {})[weights_path.name] = {
                    "error": err,
                    "at": datetime.now(timezone.utc).isoformat(),
                }
                state_path.write_text(json.dumps(state, indent=1))
                print(f"checkpoint SKIPPED: {weights_path.stem} — {err}", flush=True)

        rows = _read_history(hist_path)
        compare.regenerate(rows, out_dir)
        flags = evaluate_flags(rows)
        (out_dir / "FLAGS.md").write_text(_render_flags_md(flags))
        if flags:
            for fl in flags:
                print(f"FLAG [{fl['rule']}] {fl['checkpoint']}: {fl['detail']}", flush=True)
        else:
            print("no flags", flush=True)

        if once:
            break
        time.sleep(every_s)
