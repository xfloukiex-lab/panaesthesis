"""Model adapters: any model, one instrument.

A ModelAdapter wraps a concrete model behind a tiny uniform interface:

    tap_names()   -> [str]            every tap, in execution order
    default_taps() -> (early, late, output)
    forward(x, overrides=None) -> {tap_name: np.ndarray}
    describe()    -> dict             provenance fragment
    close()       -> None

Conventions every adapter honors:
  - forward takes a BATCHED input (batch dim first), exactly as the old
    portrait loop fed Xs[t:t+1].
  - every returned activation is a numpy array with the batch dim first.
  - "output" is always a tap: the model's final output.
  - overrides maps tap_name -> replacement activation (same shape as the
    tap's natural output, batch dim first). The forward pass continues
    downstream of the overridden tap. This is what the weaver uses to
    reach in and change a relation mid-read.

Two adapters ship:

  NumpyCheckpointAdapter("npz:<path>")
      The original .npz checkpoint format (vendored numpy layers). Arch is
      inferred from weight shapes; the layer pattern follows the family
      convention (conv -> relu -> maxpool per conv block,
      affine -> relu -> dropout per hidden affine). Any depth and width
      works — 2-conv readers, 3-conv classifiers, pure MLPs, deeper CNNs.
      Checkpoints whose index gaps don't match the convention raise an
      honest error instead of being misread.

  TorchModuleAdapter("torch:<pyfile>:<attr>")
      ANY torch.nn.Module, via forward hooks: every named submodule's
      output becomes a tap. Covers MLPs, CNNs, and HuggingFace
      transformers alike. Default taps are a documented heuristic
      (first tapped module, last hidden tapped module, final output) and
      can be overridden with --taps.

Adding a new model type = writing a small adapter (~60 lines). The
equations, the monitor, the weaver, and the live instrument never see
anything but this interface.
"""
import hashlib
import importlib.util
from pathlib import Path

import numpy as np


class ModelAdapter:
    """Uniform interface over any model. Subclass and implement."""

    kind = "base"

    @property
    def name(self):
        raise NotImplementedError

    def tap_names(self):
        raise NotImplementedError

    def default_taps(self):
        """(early, late, output) tap names. Documented heuristic per adapter."""
        raise NotImplementedError

    def forward(self, x, overrides=None):
        """Batched forward. Returns {tap_name: np.ndarray (batch first)}.

        overrides: {tap_name: array} — replace that tap's activation and
        continue the pass downstream (the weaver's entry point).
        """
        raise NotImplementedError

    def describe(self):
        """Provenance fragment for metrics.json."""
        return {"model_kind": self.kind, "model_name": self.name,
                "taps": self.tap_names()}

    def close(self):
        pass


# ---------------------------------------------------------------------------
# .npz checkpoints (the original vendored-numpy format)
# ---------------------------------------------------------------------------

def _npz_weighted_entries(weights_path):
    """(idx, class_name, shape) for every L{ii}_{Class}_W key, sorted by idx."""
    d = np.load(weights_path)
    entries = []
    for key in d.files:
        if not key.endswith("_W"):
            continue
        head, cls, w = key.rsplit("_", 2)
        assert w == "W", key
        idx = int(head[1:])
        entries.append((idx, cls, d[key].shape))
    entries.sort()
    input_shape = None
    if "input_shape" in d.files:
        input_shape = tuple(int(v) for v in d["input_shape"])
    return entries, input_shape


def infer_numpy_arch(weights_path, input_shape=None):
    """Reconstruct a buildable arch spec from a checkpoint's weight shapes.

    The .npz only stores weighted layers; the unweighted followers are
    reconstructed by the family convention (conv -> relu -> maxpool,
    hidden affine -> relu -> dropout). Index gaps that don't match the
    convention raise instead of being misread — that checkpoint needs a
    custom adapter (or the torch route), not a guess.

    Returns {"name", "spec", "family", "input_shape", "weighted_idx"}.
    """
    from hodos_monitor.vendor import models as vmodels  # noqa: F401

    entries, file_input_shape = _npz_weighted_entries(weights_path)
    if not entries:
        raise ValueError(f"no L{{ii}}_{{Class}}_W keys in {weights_path}")
    for _, cls, _ in entries:
        if cls not in ("Conv2d", "Affine"):
            raise ValueError(
                f"unsupported weighted layer class '{cls}' in {weights_path}; "
                f"the numpy adapter handles Conv2d/Affine families — other "
                f"architectures go through the torch adapter or a custom adapter")

    if input_shape is None:
        input_shape = file_input_shape
    if input_shape is None:
        # legacy checkpoints predate the input_shape key; known conventions
        n_conv = sum(1 for _, cls, _ in entries if cls == "Conv2d")
        legacy = {2: (1, 24, 24), 3: (1, 48, 48), 0: None}
        input_shape = legacy.get(n_conv)
    if input_shape is None:
        raise ValueError(
            f"cannot infer input shape for {weights_path}: pass input_shape "
            f"explicitly (or store an 'input_shape' key in the .npz)")

    # Reconstruct the full layer sequence, verifying index gaps.
    layers = []
    pos = 0  # next expected built index
    n = len(entries)
    has_conv = any(e[1] == "Conv2d" for e in entries)
    for j, (idx, cls, shape) in enumerate(entries):
        if idx != pos:
            raise ValueError(
                f"index gap at {weights_path}: expected weighted layer at "
                f"built index {pos}, found {cls} at {idx} — outside the "
                f"family convention; use the torch adapter or a custom adapter")
        last = (j == n - 1)
        if cls == "Conv2d":
            out_ch, in_ch, kh, kw = shape
            if kh != kw:
                raise ValueError(f"non-square kernel in {weights_path}: {shape}")
            layers.append({"type": "conv", "in_ch": int(in_ch),
                           "out_ch": int(out_ch), "k": int(kh)})
            pos += 1
            layers.append({"type": "relu"})
            pos += 1
            layers.append({"type": "maxpool"})
            pos += 1
        else:  # Affine
            in_f, out_f = (int(v) for v in shape)
            if j == 0:
                # First weighted layer: flatten width is unrecoverable from
                # the weight matrix alone. After a conv stack the builder
                # probes it; for a pure MLP it is the flat input size.
                in_f_spec = "auto" if has_conv else int(np.prod(input_shape))
            else:
                in_f_spec = in_f
            layers.append({"type": "affine", "in_f": in_f_spec,
                           "out_f": out_f})
            pos += 1
            if not last:
                layers.append({"type": "relu"})
                pos += 1
                # Dropout p is unrecoverable from weights; forced to eval, so
                # the value never affects the forward pass.
                layers.append({"type": "dropout", "p": 0.25})
                pos += 1
    spec = {"name": Path(weights_path).stem,
            "input": list(input_shape), "init": "he_normal", "layers": layers}
    n_conv = sum(1 for _, cls, _ in entries if cls == "Conv2d")
    family = {0: "mlp"}.get(n_conv, f"conv{n_conv}")
    if n_conv == 2 and tuple(input_shape) == (1, 24, 24):
        family = "reader"
    elif n_conv == 3 and tuple(input_shape) == (1, 48, 48):
        family = "classifier"
    return {"name": spec["name"], "spec": spec, "family": family,
            "input_shape": tuple(input_shape),
            "weighted_idx": [e[0] for e in entries]}


class NumpyCheckpointAdapter(ModelAdapter):
    """Any .npz checkpoint in the vendored-numpy family format."""

    kind = "numpy"

    def __init__(self, weights_path, input_shape=None):
        from hodos_monitor.vendor import models as vmodels
        self.weights_path = Path(weights_path)
        self.arch = infer_numpy_arch(self.weights_path, input_shape)
        layers = vmodels.build(self.arch["spec"], seed=0)
        vmodels.load_weights(layers, str(self.weights_path))
        for l in layers:
            if isinstance(l, vmodels.Dropout):
                l.train = False
        self.layers = layers
        self._vmodels = vmodels
        self._tap_names = [f"L{i:02d}_{type(l).__name__}"
                           for i, l in enumerate(layers)]
        self._early, self._late = self._select_relation_taps()

    @property
    def name(self):
        return self.arch["name"]

    def tap_names(self):
        return list(self._tap_names)

    def _select_relation_taps(self):
        """early/late by layer kind, with documented fallbacks.

        early: first MaxPool2d output -> first ReLU output -> layers[0].
        late:  ReLU after the last hidden Affine -> last hidden Affine
               output -> second-to-last layer.
        """
        vmodels = self._vmodels
        kinds = [type(l).__name__ for l in self.layers]
        try:
            early = kinds.index("MaxPool2d")
        except ValueError:
            try:
                early = kinds.index("ReLU")
            except ValueError:
                early = 0
        affine_idx = [i for i, k in enumerate(kinds) if k == "Affine"]
        if len(affine_idx) >= 2 and kinds[affine_idx[-2] + 1] == "ReLU":
            late = affine_idx[-2] + 1
        elif len(affine_idx) >= 2:
            late = affine_idx[-2]
        else:
            late = max(0, len(self.layers) - 2)
        if late == len(self.layers) - 1:
            late = max(0, len(self.layers) - 2)
        if early >= late:
            early = max(0, late - 1)
        return self._tap_names[early], self._tap_names[late]

    def default_taps(self):
        return self._early, self._late, self._tap_names[-1]

    def forward(self, x, overrides=None):
        x = np.asarray(x)
        if x.ndim == 0:
            raise ValueError("forward needs a batched input (batch dim first)")
        batch = x.shape[0]
        ov = {}
        for k, v in (overrides or {}).items():
            if k not in self._tap_names and k != "output":
                raise KeyError(f"unknown tap {k!r}")
            a = np.asarray(v)
            if a.shape[0] != batch:
                raise ValueError(
                    f"override for tap {k!r} has batch {a.shape[0]}, "
                    f"input batch is {batch}")
            ov[k] = a
        taps = {}
        for name, layer in zip(self._tap_names, self.layers):
            x = ov[name] if name in ov else layer.forward(x)
            taps[name] = np.asarray(x)
        # output tap: the final layer's activation
        taps["output"] = taps[self._tap_names[-1]]
        return taps

    def continue_from(self, tap_name, activation):
        """Run the layers downstream of tap_name on a replacement activation.

        The weaver's entry point for the numpy family: exact replay of the
        remaining layers, no re-entry through the head.
        """
        i = self._tap_names.index(tap_name)
        x = np.asarray(activation)
        for layer in self.layers[i + 1:]:
            x = layer.forward(x)
        return np.asarray(x)

    def describe(self):
        d = super().describe()
        d.update({
            "weights_path": str(self.weights_path),
            "weights_sha256": _sha256(self.weights_path),
            "arch_name": self.arch["name"],
            "family": self.arch["family"],
            "n_params": int(self._vmodels.param_count(self.layers)),
            "input_shape": list(self.arch["input_shape"]),
            "early_tap": self._early,
            "late_tap": self._late,
        })
        return d


# ---------------------------------------------------------------------------
# torch.nn.Module — any PyTorch model, via forward hooks
# ---------------------------------------------------------------------------

def _load_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError(
            "the torch adapter needs PyTorch: pip install torch "
            "(CPU wheel is enough — the instrument only runs inference)")


def _import_spec(spec):
    """'path/to/module.py:attr' -> the attribute (module instance or factory).

    The attribute is split off the RIGHT so a Windows absolute path keeps its
    drive colon: 'C:/models/wrapper.py:net' -> ('C:/models/wrapper.py', 'net').
    """
    path, _, attr = spec.rpartition(":")
    if not path or not attr:
        raise ValueError(
            f"torch model spec must be 'path/to/module.py:attribute', got {spec!r}")
    mod_path = Path(path)
    if not mod_path.exists():
        raise ValueError(f"module file not found: {path}")
    spec_obj = importlib.util.spec_from_file_location(
        f"hodos_torch_user_{mod_path.stem}", str(mod_path))
    module = importlib.util.module_from_spec(spec_obj)
    spec_obj.loader.exec_module(module)
    obj = getattr(module, attr, None)
    if obj is None:
        raise ValueError(f"{path} has no attribute {attr!r}")
    return obj() if isinstance(obj, type) else (obj() if callable(obj) and
            not hasattr(obj, "forward") else obj)


class TorchModuleAdapter(ModelAdapter):
    """Any torch.nn.Module. Every named submodule's output is a tap.

    Hooks capture outputs in forward-execution order. Default taps are a
    documented heuristic — first tapped submodule (early), last tapped
    submodule before the root output (late), root output — override with
    --taps early_tap,late_tap.
    """

    kind = "torch"

    def __init__(self, module, name="torch-model", all_sites=False):
        torch = _load_torch()
        self._torch = torch
        if not isinstance(module, torch.nn.Module):
            raise TypeError(
                f"torch adapter needs a torch.nn.Module, got {type(module).__name__}")
        self.module = module.eval()
        self._name = name
        self._all_sites = all_sites
        self._order = []          # tap names in first-fire (execution) order
        self._seen = set()        # id(module) already hooked
        self._captured = {}
        self._hook_order = []
        self._module_names = []   # named submodule taps (excludes __root__)
        for qual, sub in self.module.named_modules():
            if sub is self.module or id(sub) in self._seen:
                continue
            self._seen.add(id(sub))
            tap = qual if qual else type(sub).__name__
            sub.register_forward_hook(self._make_hook(tap))
            self._order.append(tap)
            self._module_names.append(tap)
        # root output tap
        self._root_hook = self.module.register_forward_hook(
            self._make_hook("__root__"))
        self._n_params = sum(p.numel() for p in self.module.parameters())
        # all-sites enrichment (per-head attention + residual stream); the
        # default path leaves these empty and is byte-identical to before.
        self._block_pre = {}      # block_name -> resid_pre tensor (this forward)
        self._oproj_pre = {}      # oproj_name -> concat-z tensor (this forward)
        self._pre_handles = []
        self._attn_cfg = None
        self._derived_names = []
        if all_sites:
            self._setup_all_sites()

    def _setup_all_sites(self):
        """Pre-hook decoder blocks (resid_pre) + o_proj (concat per-head z), and
        precompute the derived tap-name list from structure + attention config."""
        from hodos_monitor import sites
        self._attn_cfg = sites.find_attn_config(self.module)
        names = self._module_names
        name_set = set(names)
        qual_to_mod = dict(self.module.named_modules())
        for n in names:
            if sites.is_block(n):
                mod = qual_to_mod.get(n)
                if mod is not None:
                    self._pre_handles.append(
                        mod.register_forward_pre_hook(self._make_block_pre(n)))
            if n.endswith(".o_proj"):
                mod = qual_to_mod.get(n)
                if mod is not None:
                    self._pre_handles.append(
                        mod.register_forward_pre_hook(self._make_oproj_pre(n)))
        derived = []
        if self._attn_cfg is not None:
            n_heads, n_kv, _ = self._attn_cfg
            for block in [n for n in names if sites.is_block(n)]:
                attn = sites._attn_prefix(block, names)
                if attn is None:
                    continue
                if f"{attn}.q_proj" in name_set:
                    derived += [f"{attn}.q.h{h}" for h in range(n_heads)]
                if f"{attn}.k_proj" in name_set:
                    derived += [f"{attn}.k.h{h}" for h in range(n_kv)]
                if f"{attn}.v_proj" in name_set:
                    derived += [f"{attn}.v.h{h}" for h in range(n_kv)]
                if f"{attn}.o_proj" in name_set:
                    derived += [f"{attn}.z.h{h}" for h in range(n_heads)]
        for block in [n for n in names if sites.is_block(n)]:
            derived += [f"{block}.resid_pre", f"{block}.resid_post"]
            attn = sites._attn_prefix(block, names)
            if attn and f"{attn}.o_proj" in name_set:
                derived.append(f"{block}.resid_mid")
        self._derived_names = derived

    def _make_block_pre(self, name):
        def pre(mod, args):
            if args and isinstance(args[0], self._torch.Tensor):
                self._block_pre[name] = args[0].detach()
        return pre

    def _make_oproj_pre(self, name):
        def pre(mod, args):
            if args and isinstance(args[0], self._torch.Tensor):
                self._oproj_pre[name] = args[0].detach()
        return pre

    def _make_hook(self, tap):
        def hook(mod, inp, out):
            t = out[0] if isinstance(out, (tuple, list)) and out else out
            if isinstance(t, dict):
                t = next((v for v in t.values()
                          if isinstance(v, self._torch.Tensor)), None)
            if isinstance(t, self._torch.Tensor):
                self._captured[tap] = t.detach()
                if tap not in self._hook_order:
                    self._hook_order.append(tap)
        return hook

    @property
    def name(self):
        return self._name

    def tap_names(self):
        base = [t for t in self._order if t != "__root__"]
        if self._all_sites:
            return base + list(self._derived_names) + ["output"]
        return base + ["output"]

    def default_taps(self):
        inner = [t for t in self._order if t != "__root__"]
        if not inner:
            return "output", "output", "output"
        early = inner[0]
        late = inner[-1] if len(inner) > 1 else inner[0]
        return early, late, "output"

    def forward(self, x, overrides=None):
        torch = self._torch
        if isinstance(x, np.ndarray):
            xt = torch.from_numpy(np.ascontiguousarray(x)).to(torch.float32)
        elif isinstance(x, torch.Tensor):
            xt = x
        else:
            raise TypeError(f"torch adapter input must be np.ndarray or Tensor, "
                            f"got {type(x).__name__}")
        self._captured = {}
        self._hook_order = []
        self._block_pre = {}
        self._oproj_pre = {}
        handles = []
        try:
            for tap, val in (overrides or {}).items():
                arr = np.ascontiguousarray(val)
                # "pre:<module>" replaces that module's INPUT via a forward
                # PRE-hook (for residual-stream / z-head injection, whose sites
                # are module inputs, not outputs). prepend=True so it runs
                # BEFORE the all-sites capture pre-hooks -- the derived taps
                # (resid_pre, z.h) then reflect the injected value in the
                # after-field, not the natural one. Any other key replaces the
                # module OUTPUT via a forward hook, as before.
                if isinstance(tap, str) and tap.startswith("pre:"):
                    mod = self._resolve(tap[4:])

                    def _prep(m, args, _arr=arr, _torch=torch):
                        if args and isinstance(args[0], _torch.Tensor):
                            rep = _torch.from_numpy(_arr).to(
                                args[0].device, args[0].dtype)
                            return (rep,) + tuple(args[1:])
                        return None
                    handles.append(mod.register_forward_pre_hook(
                        _prep, prepend=True))
                else:
                    mod = self._resolve(tap)

                    def _rep(m, i, o, _arr=arr, _torch=torch):
                        return _torch.from_numpy(_arr).to(o.device, o.dtype) \
                            if isinstance(o, _torch.Tensor) else o
                    handles.append(mod.register_forward_hook(_rep))
            with torch.no_grad():
                out = self.module(xt)
            taps = {}
            for tap in self._order:
                if tap in self._captured:
                    taps[tap] = self._captured[tap].cpu().numpy()
            t = out[0] if isinstance(out, (tuple, list)) and out else out
            taps["output"] = t.detach().cpu().numpy() \
                if isinstance(t, torch.Tensor) else np.asarray(t)
            if self._all_sites:
                from hodos_monitor import sites
                block_pre = {k: v.cpu().numpy() for k, v in self._block_pre.items()}
                oproj_pre = {k: v.cpu().numpy() for k, v in self._oproj_pre.items()}
                taps = sites.enrich(taps, block_pre, oproj_pre,
                                    self._module_names, self._attn_cfg)
            return taps
        finally:
            for h in handles:
                h.remove()

    def _resolve(self, tap):
        if tap in ("output", "__root__"):
            return self.module
        mod = self.module
        for part in tap.split("."):
            mod = getattr(mod, part)
        return mod

    def describe(self):
        torch = self._torch
        d = super().describe()
        d.update({
            "framework": f"torch {torch.__version__}",
            "module_class": type(self.module).__name__,
            "n_params": int(self._n_params),
            "n_taps": len(self.tap_names()),
            "default_taps": list(self.default_taps()),
            "all_sites": self._all_sites,
            "attn_config": (list(self._attn_cfg) if self._attn_cfg else None),
        })
        return d

    def close(self):
        self._root_hook.remove()
        for h in self._pre_handles:
            h.remove()


# ---------------------------------------------------------------------------
# raw data — bring your own dataset (no model, no domain baked in)
# ---------------------------------------------------------------------------

class DataAdapter(ModelAdapter):
    """A raw dataset as a system of related PARTS — bring-your-own-data.

    There is no model and no forward process here: the "system" is a table of
    observations x parts, and each PART (column) is a tap the instrument can
    sensor. The instrument reads the parts across the observations (the
    stimulus rows) and measures the relations among them with the same four
    equations it uses on a model's activations. `forward` is the identity — it
    exposes the input row's components as taps — so a user brings their own
    data and needs no adapter code of their own.

    Nothing about any domain is assumed: a spectrum's m/z bins, a patient's
    channels, a market's instruments, a network's flows are all just columns.

    spec:  data:<path>   where <path> is an .npz carrying
        values | inputs   (N, P)   the table: N observations, P parts (required)
        parts             (P,)     optional part names (else p000, p001, ...)

    A reshape override replaces a part's value on the pass, so the Hodotome can
    cut or drive a part and the field is re-read with that part changed. Because
    a raw table has no downstream, the reshape is edit-and-re-read (there is no
    process to propagate through) — which is stated honestly, not hidden.
    """

    kind = "data"

    def __init__(self, path):
        self.path = str(path)
        d = np.load(self.path, allow_pickle=True)
        key = ("values" if "values" in d.files
               else "inputs" if "inputs" in d.files else None)
        if key is None:
            raise ValueError(
                f"data file {self.path} needs a 'values' or 'inputs' key "
                f"(found: {list(d.files)})")
        vals = np.asarray(d[key])
        if vals.ndim == 2:
            # (N, P): P scalar parts, one value each — width 1. Readable by a
            # scalar relation reader, but the field's distribution-based
            # equations need a profile (see the >=2-wide check downstream).
            self.n_obs, self.n_parts, self.width = (
                int(vals.shape[0]), int(vals.shape[1]), 1)
        elif vals.ndim == 3:
            # (N, P, W): P parts, each carrying a width-W profile per
            # observation — a part with an internal distribution, which is what
            # a Hodos relation is read from.
            self.n_obs, self.n_parts, self.width = (
                int(vals.shape[0]), int(vals.shape[1]), int(vals.shape[2]))
        else:
            raise ValueError(
                f"the data adapter needs a table shaped (observations, parts) "
                f"or (observations, parts, profile); '{key}' in {self.path} "
                f"has shape {vals.shape}")
        if self.n_parts < 2:
            raise ValueError(
                f"a relation needs at least 2 parts; {self.path} has "
                f"{self.n_parts} part(s)")
        if "parts" in d.files:
            parts = [str(p) for p in d["parts"]]
            if len(parts) != self.n_parts:
                raise ValueError(
                    f"'parts' lists {len(parts)} names for {self.n_parts} "
                    f"columns in {self.path}")
        else:
            width = max(3, len(str(self.n_parts - 1)))
            parts = [f"p{j:0{width}d}" for j in range(self.n_parts)]
        self._parts = parts

    @property
    def name(self):
        return Path(self.path).stem

    def tap_names(self):
        return list(self._parts)

    def default_taps(self):
        # first / middle / last part — a documented, domain-free heuristic.
        n = self.n_parts
        return self._parts[0], self._parts[n // 2], self._parts[-1]

    def forward(self, x, overrides=None):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None]
        if x.ndim not in (2, 3) or x.shape[1] != self.n_parts:
            raise ValueError(
                f"data forward expects (batch, {self.n_parts}) or "
                f"(batch, {self.n_parts}, {self.width}); got shape {x.shape}")
        batch = x.shape[0]
        ov = {}
        for k, v in (overrides or {}).items():
            if k not in self._parts and k != "output":
                raise KeyError(f"unknown part {k!r}")
            a = np.asarray(v)
            if a.shape[0] != batch:
                raise ValueError(
                    f"override for part {k!r} has batch {a.shape[0]}, "
                    f"input batch is {batch}")
            ov[k] = a
        taps, cols = {}, []
        for j, part in enumerate(self._parts):
            col = ov[part] if part in ov else x[:, j]
            col = np.asarray(col, dtype=np.float64).reshape(batch, -1)
            taps[part] = col
            cols.append(col)
        taps["output"] = (np.asarray(ov["output"]) if "output" in ov
                          else np.concatenate(cols, axis=1))
        return taps

    def describe(self):
        d = super().describe()
        d.update({
            "data_path": self.path,
            "data_sha256": _sha256(self.path),
            "n_obs": int(self.n_obs),
            "n_parts": int(self.n_parts),
            "default_taps": list(self.default_taps()),
        })
        return d


# ---------------------------------------------------------------------------
# spec loading
# ---------------------------------------------------------------------------

def load_model(spec, **kwargs):
    """'npz:<path>' | 'torch:<pyfile>:<attr>' | 'torchall:<pyfile>:<attr>' | 'data:<path>' -> ModelAdapter.

    kwargs: input_shape (numpy adapter), name (torch adapter),
    all_sites (torch adapter: also tap per-head attention + residual stream;
    the 'torchall:' prefix is shorthand for all_sites=True).
    A ModelAdapter instance passes straight through.
    """
    if isinstance(spec, ModelAdapter):
        return spec
    kind, _, rest = spec.partition(":")
    # NumpyCheckpointAdapter takes only input_shape; all_sites/name are torch-
    # only. A numpy checkpoint already exposes every layer as a site, so the
    # flag is a no-op here — drop it instead of erroring, so the field/family
    # paths do not crash on an .npz.
    npz_kwargs = {k: v for k, v in kwargs.items() if k == "input_shape"}
    if kind == "npz":
        return NumpyCheckpointAdapter(rest, **npz_kwargs)
    if kind == "data":
        return DataAdapter(rest)
    if kind in ("torch", "torchall"):
        obj = _import_spec(rest)
        torch = _load_torch()
        if not isinstance(obj, torch.nn.Module):
            raise TypeError(
                f"torch spec must resolve to a torch.nn.Module, got "
                f"{type(obj).__name__}")
        all_sites = kwargs.get("all_sites", False) or kind == "torchall"
        return TorchModuleAdapter(obj, name=kwargs.get("name", Path(rest).name),
                                  all_sites=all_sites)
    # bare path = legacy behavior: a numpy checkpoint
    if Path(spec).exists():
        return NumpyCheckpointAdapter(spec, **npz_kwargs)
    raise ValueError(
        f"unknown model spec {spec!r}: use 'npz:<path>' or "
        f"'torch:<pyfile>:<attr>'")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
