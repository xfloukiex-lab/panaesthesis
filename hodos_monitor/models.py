"""Checkpoint loading.

load_checkpoint(weights_path) -> (layers, early_idx, late_idx, logits_idx, arch_name, arch)

- Reads the .npz, infers the architecture from weight shapes (no hardcoded
  layer indices): conv blocks are conv/relu/maxpool, head is affine/relu/
  (dropout)/affine. Supports the READER_ARCH family at any width (33k v1/v2,
  121k v2.1) and CLASSIFIER_ARCH.
- Builds via the vendored models.build (logic untouched), loads weights via the
  vendored load_weights (raises on shape mismatch — honest failure).
- Taps layers BY KIND: early = output of the first MaxPool2d; late = output of
  the ReLU following the last hidden Affine (before the logits Affine).
- Dropout is forced to eval (train=False).

The layer classes are referenced through the vendored models module's own
namespace so isinstance checks hit the exact classes the vendored builder
instantiates.
"""
import numpy as np

from hodos_monitor.vendor import models as vmodels

_MaxPool2d = vmodels.MaxPool2d
_ReLU = vmodels.ReLU
_Affine = vmodels.Affine
_Dropout = vmodels.Dropout


def infer_arch(weights_path):
    """Infer an arch spec dict from .npz weight shapes.

    Returns {"name", "spec" (vendored-build arch dict), "family",
             "conv_widths", "affine_widths"}.
    """
    d = np.load(weights_path)
    convs = []    # (idx, in_ch, out_ch, k)
    affines = []  # (idx, in_f, out_f)
    for key in d.files:
        if not key.endswith("_W"):
            continue
        parts = key.split("_")
        idx = int(parts[0][1:])
        cls = parts[1]
        shape = d[key].shape
        if cls == "Conv2d":
            out_ch, in_ch, kh, kw = shape
            assert kh == kw, f"non-square kernel in {key}: {shape}"
            convs.append((idx, int(in_ch), int(out_ch), int(kh)))
        elif cls == "Affine":
            in_f, out_f = shape
            affines.append((idx, int(in_f), int(out_f)))
    convs.sort()
    affines.sort()
    n_conv = len(convs)
    if n_conv == 2:
        family, spatial, dropout_p = "reader", 24, 0.25
    elif n_conv == 3:
        family, spatial, dropout_p = "classifier", 48, 0.30
    else:
        raise ValueError(
            f"cannot infer arch from {weights_path}: "
            f"found {n_conv} Conv2d layers (need 2 for reader, 3 for classifier)"
        )
    if not affines:
        raise ValueError(f"no Affine layers in {weights_path}")

    layers = []
    for _, in_ch, out_ch, k in convs:
        layers.append({"type": "conv", "in_ch": in_ch, "out_ch": out_ch, "k": k})
        layers.append({"type": "relu"})
        layers.append({"type": "maxpool"})
    for j, (_, in_f, out_f) in enumerate(affines):
        layers.append({"type": "affine",
                       "in_f": "auto" if j == 0 else in_f,
                       "out_f": out_f})
        if j < len(affines) - 1:
            layers.append({"type": "relu"})
            # Dropout p is unrecoverable from weights; forced to eval below,
            # so the value never affects the forward pass. Family default kept
            # for build() fidelity.
            layers.append({"type": "dropout", "p": dropout_p})

    conv_widths = [c[2] for c in convs]
    affine_widths = [a[2] for a in affines]
    name = family + "-" + "-".join(str(w) for w in conv_widths + affine_widths)
    spec = {
        "name": name,
        "input": [convs[0][1], spatial, spatial],
        "init": "he_normal",
        "layers": layers,
    }
    return {"name": name, "spec": spec, "family": family,
            "conv_widths": conv_widths, "affine_widths": affine_widths}


def load_checkpoint(weights_path):
    """Build layers from inferred arch, load weights, tap early/late by kind."""
    arch = infer_arch(weights_path)
    layers = vmodels.build(arch["spec"], seed=0)
    vmodels.load_weights(layers, str(weights_path))
    for l in layers:
        if isinstance(l, _Dropout):
            l.train = False

    early_idx = next(i for i, l in enumerate(layers)
                     if isinstance(l, _MaxPool2d))
    affine_idx = [i for i, l in enumerate(layers) if isinstance(l, _Affine)]
    logits_idx = affine_idx[-1]
    late_idx = affine_idx[-2] + 1
    if not isinstance(layers[late_idx], _ReLU):
        raise ValueError(
            f"expected ReLU after last hidden affine at layer {late_idx}, "
            f"found {type(layers[late_idx]).__name__}"
        )
    return layers, early_idx, late_idx, logits_idx, arch["name"], arch


def param_count(layers):
    return vmodels.param_count(layers)
