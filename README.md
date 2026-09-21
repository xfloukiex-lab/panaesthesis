# Panaesthesis

**Panaesthesis** (Greek *pan-*, all + *aisthesis*, perception — "all-perception")
is a monitoring instrument for **any system with internal relations** — not only
neural networks. You put sensors on a system's internal parts, it measures the
relations between them through the four Hodos equations, and — where the system
allows it — you reshape a relation and watch the whole field reorganize.

It is software, not art. Every curve is computed from the published Hodos math,
not tuned.

Two instruments:

- **Hodoscope** — the microscope. Watch and measure a system's internal
  relational life: `portrait`, `live` (and `live --twin`), `field` (every site
  at once), `family`, `watch`.
- **Hodotome** — the scalpel. Reach in, change a named relation, continue the
  run downstream, and measure what happened: `intervene`, `splice` / `cut`,
  the LoRA branch readout, `--gif` views of the run.

The four equations, in one line each:

- **Diastema** — how far apart two parts live, as distributions, over the run.
- **Symploke** — how much a relationship *creates* beyond its parts (surplus
  over the product of marginals, z-scored against a shuffle null).
- **Systasis** — the constitution of the parts (named, reported honestly when it
  does not clear).
- **Chronos** — lived time: how much time a process lived per step.

## Connect your own system

The instrument talks to a small `ModelAdapter` seam, never a system directly, so
a new kind of system is a new adapter of a few dozen lines. Three ship:

- `npz:<path>` — a numpy checkpoint (Conv / Affine / MLP).
- `torch:<file.py>:<attr>` — any `torch.nn.Module` (needs `torch`).
- `data:<path>` — **bring your own data.** A table of observations × parts,
  where each part carries a short profile: an `.npz` with `values` (or `inputs`)
  of shape `(observations, parts, profile)` and optional `parts` names. Each
  part becomes a sensor; the instrument maps the relations among them and lets
  you reshape one. Nothing about any domain is assumed — spectra, channels,
  sensors, flows are all just parts.

## Install

Python 3, with `numpy`, `matplotlib`, `pillow`:

```
pip install numpy matplotlib pillow
```

`torch` is optional — needed only for the `torch:` adapter.

## Quick start

Try the bundled font-free demo:

```
python -m hodos_monitor.app        # the desktop app: Connect -> Try the demo
```

Or from the command line, map every relation in a dataset you bring:

```
python -m hodos_monitor field --model data:your_data.npz --out ./out
```

where `your_data.npz` holds `inputs` of shape `(observations, parts, profile)`.
Coupled parts read closer than unrelated ones; the full matrix is written to
`field_metrics.json` and drawn to `field.png`.

Portrait a model instead:

```
python -m hodos_monitor portrait --model npz:model.npz --out ./portraits/v1
python -m hodos_monitor portrait --model torch:mymodel.py:net --stimulus array:inputs.npz --out ./portraits/mlp
```

`--taps early,late,output` overrides the default tap choice; `--n-pair N` sets
the Symploke pairing count.

## Reshape a relation

The monitor watches; the Hodotome acts. Name specific wires to couple or cut;
everything else runs bit-identical, so cause maps to effect:

```
python -m hodos_monitor splice --model npz:model.npz --splice 5:12 --cut 3:44 --out ./spliced/v1
python -m hodos_monitor intervene --model torch:mymodel.py:net --level head --op cut --out ./out
```

Each reshape is paired with a whole-field before/after map, so you see the
system reorganize, not just one number move.

## Honesty labels (kept on every reading)

- **Compute, don't decorate** — every curve is the published math, not tuned.
- **Nulls as loudly as positives** — a relation that does not clear its null is
  reported as such.
- **Systasis is ghosted** — it did not clear; its box stays ghosted, a reminder,
  not a result.
- **Chronos τ is one sampling's reading** — never presented as universal.
- **Section caps are the input regimes we fed**, not regimes the equations
  derived. When the computed reading does not follow them, that mismatch *is* the
  finding.

## License

Apache-2.0. See `LICENSE`.

Built on the Hodos equations from the Vektorgeist research programme
(Alexander Parnell · Vektorgeist).
