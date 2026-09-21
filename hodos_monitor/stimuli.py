"""Stimulus protocols.

reader120: the EXACT 120-glyph clean(40)->degraded(40)->clean(40) protocol from
hodos-ai-portrait-v1.py. _canvas / render_clean / render_hard are copied
verbatim; the only change is that the rng is passed in (v1 used a module-level
rng seeded 20260918). Replicate 0 with master seed 20260918 therefore reproduces
v1's stimulus bit-for-bit, and the same rng object must then be handed to
equations.symploke so the null-shuffle consumption order matches v1 exactly.

Replicate seeds: replicate 0 uses the master seed directly; replicate r>0 uses
(master_seed, r) as the seed tuple. Deterministic, documented, no collisions
with replicate 0.

Lane 2 (classifier widget stimulus) is an interface stub for now.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hodos_monitor.vendor import factory
from hodos_monitor.vendor.factory import CHARS, CHAR2IDX

T_READER120 = 120
HARD_START = 40
HARD_END = 80


def _canvas(char, r):
    px = int(r.uniform(22, 44))
    font = ImageFont.truetype(r.choice(factory.FONTS), px)
    theme = r.random()
    if theme < 0.45:
        bg, fg = int(r.uniform(235, 255)), int(r.uniform(0, 40))
    elif theme < 0.8:
        bg, fg = int(r.uniform(0, 30)), int(r.uniform(220, 255))
    else:
        bg = int(r.uniform(0, 255))
        fg = int(np.clip(bg + r.choice([-1, 1]) * r.uniform(28, 70), 0, 255))
    cw = max(8, int(font.getlength(char)) + px)
    im = Image.new("L", (cw + px, px * 2), bg)
    d = ImageDraw.Draw(im)
    d.text((px // 2, px // 2), char, fill=fg, font=font)
    return im, bg


def render_clean(char, r):
    """Same canvas as factory.render_glyph, but NO degradation, NO speckle. (v1 verbatim)"""
    im, bg = _canvas(char, r)
    a = np.asarray(im).astype(np.float32)
    ink = (a < 127).astype(np.float32) if bg > 127 else (a > 127).astype(np.float32)
    return factory._square_norm(ink).astype(np.float32), CHAR2IDX[char]


def render_hard(char, r):
    """Factory canvas + strong degradation + heavier speckle (the HARD stretch). (v1 verbatim)"""
    im, bg = _canvas(char, r)
    im = factory.degrade(im, r, strong=True)
    a = np.asarray(im).astype(np.float32)
    ink = (a < 127).astype(np.float32) if bg > 127 else (a > 127).astype(np.float32)
    sq = factory._square_norm(ink)
    flip = r.random(sq.shape) < r.uniform(0.03, 0.08)
    sq = np.where(flip, 1.0 - sq, sq)
    return sq.astype(np.float32), CHAR2IDX[char]


def render_reader120(rng):
    """120 glyphs: 40 clean / 40 strongly degraded / 40 clean. (v1 loop verbatim)

    Returns (Xs, ys): Xs (T,1,24,24) float64, ys (T,) int labels.
    """
    Xs, ys = [], []
    for t in range(T_READER120):
        c = CHARS[rng.integers(len(CHARS))]
        if HARD_START <= t < HARD_END:
            x, y = render_hard(c, rng)
        else:
            x, y = render_clean(c, rng)
        Xs.append(x)
        ys.append(y)
    Xs = np.array(Xs)[:, None, :, :].astype(np.float64)
    ys = np.array(ys)
    return Xs, ys


def replicate_seed(master_seed, replicate):
    """Seed for one replicate. Replicate 0 == master seed exactly (v1 parity)."""
    if replicate == 0:
        return int(master_seed)
    return (int(master_seed), int(replicate))


def classifier120(rng):  # noqa: ARG001
    """Lane 2 stub: widget-classifier stimulus (clean->degraded->clean crops).

    TODO: implement the 120-widget protocol mirroring reader120 — clean widget
    crops for the coast stretches, degrade_widget-densified crops for the strain
    stretch — once the classifier lane is scheduled. Raises until then.
    """
    raise NotImplementedError(
        "TODO lane 2: classifier120 widget stimulus not implemented yet."
    )


# ---------------------------------------------------------------------------
# Stimulus protocol: any input feed, one instrument
# ---------------------------------------------------------------------------

class Stimulus:
    """A feed of inputs for the instrument to read, one step at a time.

    step(t) -> (x, y, regime):
        x      the model input for step t (numpy array, WITHOUT batch dim;
               the instrument batches it)
        y      label or None (None = accuracy is skipped, honestly)
        regime a short string naming the operating regime at t, e.g.
               "strain" / "coast". Used for verification windows and the
               live calibration test card.

    masks() -> {regime: bool[T]}.
    calibration_inputs(rng, n_strain) -> list of raw inputs for the live
        instrument's test card (white balance across the operating range).
    describe() -> provenance fragment.
    """

    def __len__(self):
        raise NotImplementedError

    def step(self, t):
        raise NotImplementedError

    def regime_at(self, t):
        """Regime string at step t. MUST NOT consume any rng — masks() calls
        this before the measurement loop, and stepping would desync the
        stimulus stream."""
        raise NotImplementedError

    def masks(self):
        regimes = [self.regime_at(t) for t in range(len(self))]
        out = {}
        for r in dict.fromkeys(regimes):
            out[r] = np.array([x == r for x in regimes])
        return out

    def calibration_inputs(self, rng, n_strain):  # noqa: ARG002
        """Test-card inputs. Default: first n_strain 'strain'-regime steps."""
        masks = self.masks()
        strain = masks.get("strain")
        if strain is None or not strain.any():
            return []
        idx = np.flatnonzero(strain)[:n_strain]
        return [self.step(int(t))[0] for t in idx]

    def describe(self):
        return {"stimulus": type(self).__name__, "n_steps": len(self)}


class Reader120Stimulus(Stimulus):
    """The v1 120-glyph protocol as a Stimulus.

    Consumes its rng in EXACTLY the old iter_reader120/render_reader120
    order, so runs through this class are bit-identical to the legacy
    functions on the same seed.
    """

    def __init__(self, seed):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return T_READER120

    def step(self, t):
        # Returns x WITHOUT the batch dim: (1,24,24). Channel dims are part
        # of the sample; the instrument adds the batch dim.
        c = CHARS[self.rng.integers(len(CHARS))]
        if HARD_START <= t < HARD_END:
            sq, y = render_hard(c, self.rng)
            regime = "strain"
        else:
            sq, y = render_clean(c, self.rng)
            regime = "coast"
        return sq[None].astype(np.float64), int(y), regime

    def regime_at(self, t):
        # rng-free: masks() must not consume the stimulus stream.
        return "strain" if HARD_START <= t < HARD_END else "coast"

    def calibration_inputs(self, rng, n_strain):
        # The legacy live test card: degraded glyphs from a DEDICATED rng
        # stream (the stimulus rng is never touched by calibration).
        cards = []
        for _ in range(n_strain):
            c = CHARS[rng.integers(len(CHARS))]
            sq, _ = render_hard(c, rng)
            cards.append(sq[None].astype(np.float64))
        return cards

    def describe(self):
        d = super().describe()
        d.update({"stimulus": "reader120", "stimulus_seed": str(self.seed),
                  "strain_range": [HARD_START, HARD_END]})
        return d


class ArrayStimulus(Stimulus):
    """Inputs from a file. .npz keys:

        inputs   (N, ...)   required — raw model inputs, no batch dim
        labels   (N,)       optional — without it, accuracy is skipped
        regime   (N,)       optional — per-step regime strings (any dtype
                             convertible to str); without it every step is
                             regime "all" and verification is skipped
        strain_mask (N,)    optional bool — alternative to regime; True =
                             "strain", False = "coast"

    Also accepts a bare .npy array (inputs only).
    """

    def __init__(self, path):
        path = str(path)
        if path.endswith(".npy"):
            inputs = np.load(path)
            labels, regimes = None, None
        else:
            d = np.load(path, allow_pickle=True)
            if "inputs" not in d.files:
                raise ValueError(
                    f"array stimulus {path} needs an 'inputs' key "
                    f"(found: {d.files})")
            inputs = d["inputs"]
            labels = d["labels"] if "labels" in d.files else None
            regimes = None
            if "regime" in d.files:
                regimes = [str(r) for r in d["regime"]]
            elif "strain_mask" in d.files:
                regimes = ["strain" if m else "coast"
                           for m in d["strain_mask"].astype(bool)]
        self.inputs = np.asarray(inputs)
        self.labels = None if labels is None else np.asarray(labels)
        if self.labels is not None and len(self.labels) != len(self.inputs):
            raise ValueError("labels length != inputs length")
        self._regimes = regimes
        self.path = path

    def __len__(self):
        return len(self.inputs)

    def step(self, t):
        y = None if self.labels is None else self.labels[t].item() \
            if self.labels.ndim else self.labels[t]
        regime = "all" if self._regimes is None else self._regimes[t]
        return np.asarray(self.inputs[t]), y, regime

    def regime_at(self, t):
        # rng-free: masks() must not consume the stimulus stream.
        return "all" if self._regimes is None else self._regimes[t]

    def describe(self):
        d = super().describe()
        d.update({"stimulus": "array", "path": self.path,
                  "has_labels": self.labels is not None,
                  "has_regimes": self._regimes is not None})
        return d


def load_stimulus(spec, **kwargs):
    """'reader120' [:seed] | 'array:<path>' -> Stimulus.

    A Stimulus instance passes straight through. A bare .npz/.npy path is
    treated as an array stimulus.
    """
    if isinstance(spec, Stimulus):
        return spec
    if spec == "reader120" or spec.startswith("reader120:"):
        _, _, seed = spec.partition(":")
        seed = int(seed) if seed else kwargs.get(
            "seed", kwargs.get("master_seed", 20260918))
        return Reader120Stimulus(seed)
    if spec.startswith("array:"):
        return ArrayStimulus(spec[len("array:"):])
    if spec.endswith((".npz", ".npy")):
        return ArrayStimulus(spec)
    raise ValueError(
        f"unknown stimulus spec {spec!r}: use 'reader120[:seed]' or "
        f"'array:<path.npz>'")
