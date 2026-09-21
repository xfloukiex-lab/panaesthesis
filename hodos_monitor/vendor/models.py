#!/usr/bin/env python3
"""Tiny-brain model definitions.

Architectures are declared as plain data (ARCH dicts) so the Kaggle PyTorch
mirror can implement them 1:1. The numpy builder below and the torch script must
produce identical layer order, shapes, and init — verified by the weight-transfer
check (max abs diff < 1e-5 on a probe batch) before any product use.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import Conv2d, ReLU, MaxPool2d, Affine, Dropout

READER_ARCH = {
    "name": "tb_read_v1",
    "input": [1, 24, 24],
    "init": "he_normal",
    "layers": [
        {"type": "conv", "in_ch": 1, "out_ch": 12, "k": 5},
        {"type": "relu"},
        {"type": "maxpool"},
        {"type": "conv", "in_ch": 12, "out_ch": 24, "k": 3},
        {"type": "relu"},
        {"type": "maxpool"},
        {"type": "affine", "in_f": "auto", "out_f": 64},
        {"type": "relu"},
        {"type": "dropout", "p": 0.25},
        {"type": "affine", "in_f": 64, "out_f": 91},
    ],
    "classes": "eye _CHARS (92)",
}

CLASSIFIER_ARCH = {
    "name": "tb_class_v1",
    "input": [1, 48, 48],
    "init": "he_normal",
    "layers": [
        {"type": "conv", "in_ch": 1, "out_ch": 16, "k": 5},
        {"type": "relu"},
        {"type": "maxpool"},
        {"type": "conv", "in_ch": 16, "out_ch": 32, "k": 3},
        {"type": "relu"},
        {"type": "maxpool"},
        {"type": "conv", "in_ch": 32, "out_ch": 48, "k": 3},
        {"type": "relu"},
        {"type": "maxpool"},
        {"type": "affine", "in_f": "auto", "out_f": 96},
        {"type": "relu"},
        {"type": "dropout", "p": 0.30},
        {"type": "affine", "in_f": 96, "out_f": 7},
    ],
    "classes": ["button", "link", "icon_button", "toggle", "tab",
                "static_text", "decorative"],
}


def build(arch, seed=0):
    """Build layer list from an ARCH dict. Probes shapes to resolve 'auto'."""
    rng = np.random.RandomState(seed)
    layers = []
    for spec in arch["layers"]:
        t = spec["type"]
        if t == "conv":
            layers.append(Conv2d(spec["in_ch"], spec["out_ch"], spec["k"], rng))
        elif t == "relu":
            layers.append(ReLU())
        elif t == "maxpool":
            layers.append(MaxPool2d())
        elif t == "affine":
            in_f = spec["in_f"]
            if in_f == "auto":
                # probe: run a dummy through layers so far
                from core import forward as _fw
                in_shape = tuple(arch["input"])
                probe = np.zeros((1,) + in_shape)
                out = _fw(layers, probe, train=False)
                in_f = int(np.prod(out.shape[1:]))
            layers.append(Affine(in_f, spec["out_f"], rng))
        elif t == "dropout":
            layers.append(Dropout(spec["p"], rng))
        else:
            raise ValueError(t)
    return layers


def param_count(layers):
    n = 0
    for l in layers:
        for a in ("W", "b"):
            if hasattr(l, a):
                n += getattr(l, a).size
    return n


def save_weights(layers, path):
    """Save named raw tensors -> .npz (the interchange format)."""
    d = {}
    for i, l in enumerate(layers):
        if hasattr(l, "W"):
            d[f"L{i:02d}_{type(l).__name__}_W"] = l.W.astype(np.float32)
            d[f"L{i:02d}_{type(l).__name__}_b"] = l.b.astype(np.float32)
    np.savez(path, **d)


def load_weights(layers, path):
    """Load .npz into a matching layer list. Raises on shape mismatch."""
    d = np.load(path)
    for i, l in enumerate(layers):
        if hasattr(l, "W"):
            kW = f"L{i:02d}_{type(l).__name__}_W"
            kb = f"L{i:02d}_{type(l).__name__}_b"
            W, b = d[kW], d[kb]
            if W.shape != l.W.shape or b.shape != l.b.shape:
                raise ValueError(f"shape mismatch at {kW}: file {W.shape} vs arch {l.W.shape}")
            l.W[:] = W.astype(np.float64)
            l.b[:] = b.astype(np.float64)


def save_arch(arch, path):
    Path(path).write_text(json.dumps(arch, indent=1))


if __name__ == "__main__":
    for arch in (READER_ARCH, CLASSIFIER_ARCH):
        layers = build(arch, seed=0)
        print(arch["name"], "params:", param_count(layers))
    save_arch(READER_ARCH, Path(__file__).resolve().parent.parent / "arch_read_v1.json")
    save_arch(CLASSIFIER_ARCH, Path(__file__).resolve().parent.parent / "arch_class_v1.json")
    print("arch json written")
