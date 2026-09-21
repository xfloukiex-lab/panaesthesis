"""Internal-site decomposition: turn a transformer's raw module taps into the
canonical interpretability sites — per-head attention (q/k/v/z), the residual
stream (pre/mid/post), and MLP neuron blocks.

Why this exists: `TorchModuleAdapter` hooks every named submodule, so it already
captures each attention *projection* (q/k/v/o_proj) as one merged tensor and each
MLP submodule (gate/up/down/act_fn) as its output. But "watch the attention
heads / the residual stream" means those sites in their real form:

  - a projection output of shape (B, S, n_heads*head_dim) is ALL heads concatenated;
    a head is a (B, S, head_dim) slice. This module splits it.
  - the residual stream is not a submodule at all — it is the running sum carried
    between blocks: resid_pre (block input), resid_mid (after the attention add),
    resid_post (block output). This module assembles it from the block input +
    the o_proj output + the block output.
  - MLP neurons ARE captured (the act_fn / gate output IS the neuron activation
    vector); this module just labels them so the field readout can select them.

Everything here is pure numpy on already-captured arrays + a config triple
(n_heads, n_kv_heads, head_dim). No torch import, no model-family hardcoding
beyond the HF decoder naming convention it detects by regex.

Mechanism note (design-the-test-first): a per-head Q/K/V/Z *vector* is meaningful
for any stimulus. An attention *pattern* (softmax QK^T weights) is only meaningful
when the stimulus feeds a real token sequence (S>1); with a single-token-per-step
stimulus S=1 and the pattern is the scalar 1. Pattern capture lives in
`attention_patterns()` and is only wired on where the stimulus has S>1.
"""
import re

import numpy as np

# HF decoder block name: "...layers.<N>" (Llama/Qwen/Mistral) or "...h.<N>" (GPT-2).
_BLOCK_RE = re.compile(r"(?:^|\.)(?:layers|h|blocks)\.\d+$")
# attention / mlp submodule suffixes we know how to decompose
_ATTN_SUFFIX = "self_attn"          # Llama/Qwen/Mistral; GPT-2 uses "attn"
_ATTN_SUFFIX_ALT = "attn"
_PROJ = ("q_proj", "k_proj", "v_proj", "o_proj")


def find_attn_config(module):
    """Best-effort (n_heads, n_kv_heads, head_dim) from any HF config on the tree.

    Returns None if no config with num_attention_heads is found (a plain
    nn.Module with no attention config — per-head splitting is then skipped and
    only module + residual taps are produced).
    """
    for m in module.modules():
        cfg = getattr(m, "config", None)
        if cfg is None:
            continue
        n_heads = getattr(cfg, "num_attention_heads", None)
        if n_heads is None:
            continue
        hidden = getattr(cfg, "hidden_size", None)
        head_dim = getattr(cfg, "head_dim", None)
        if head_dim is None and hidden is not None:
            head_dim = hidden // n_heads
        n_kv = getattr(cfg, "num_key_value_heads", None) or n_heads
        if head_dim is None:
            continue
        return int(n_heads), int(n_kv), int(head_dim)
    return None


def is_block(tap_name):
    return bool(_BLOCK_RE.search(tap_name))


def _attn_prefix(block_name, module_names):
    """The attention submodule name under a block, e.g. 'model.layers.0.self_attn'."""
    for suf in (_ATTN_SUFFIX, _ATTN_SUFFIX_ALT):
        cand = f"{block_name}.{suf}"
        if any(n == cand or n.startswith(cand + ".") for n in module_names):
            return cand
    return None


def _split_heads(arr, n_heads, head_dim):
    """(B, S, n_heads*head_dim) -> list of n_heads arrays (B, S, head_dim).

    Tolerates (B, n_heads*head_dim) (no seq dim) too. Returns [] if the last
    dim doesn't factor as n_heads*head_dim (e.g. a fused/packed layout we
    won't guess at).
    """
    a = np.asarray(arr)
    last = a.shape[-1]
    if last != n_heads * head_dim:
        return []
    new_shape = a.shape[:-1] + (n_heads, head_dim)
    a = a.reshape(new_shape)
    # move head axis to front of the split: index by head
    return [a[..., h, :] for h in range(n_heads)]


def enrich(module_taps, block_inputs, oproj_inputs, module_names, attn_cfg):
    """Add per-head + residual-stream taps to a captured module-tap dict.

    module_taps  : {tap_name: np.ndarray}   every named submodule's output
                   (as TorchModuleAdapter already captures), batch-first.
    block_inputs : {block_name: np.ndarray}  resid_pre, from a block pre-hook.
    oproj_inputs : {oproj_name: np.ndarray}  the concat per-head z, from an
                   o_proj pre-hook (input to the output projection).
    module_names : iterable[str]  all hooked submodule names (for structure).
    attn_cfg     : (n_heads, n_kv_heads, head_dim) or None.

    Returns a NEW dict = module_taps plus the derived sites. Original module
    taps are preserved untouched (MLP neuron vectors live there as
    '<block>.mlp.act_fn'). Derived tap names:

      <attn>.q.h{H}  <attn>.k.h{H}  <attn>.v.h{H}  <attn>.z.h{H}
      <block>.resid_pre  <block>.resid_mid  <block>.resid_post
    """
    out = dict(module_taps)
    names = list(module_names)

    # ---- per-head attention ------------------------------------------------
    if attn_cfg is not None:
        n_heads, n_kv, head_dim = attn_cfg
        for block in [n for n in names if is_block(n)]:
            attn = _attn_prefix(block, names)
            if attn is None:
                continue
            head_specs = [
                ("q", module_taps.get(f"{attn}.q_proj"), n_heads),
                ("k", module_taps.get(f"{attn}.k_proj"), n_kv),
                ("v", module_taps.get(f"{attn}.v_proj"), n_kv),
                ("z", oproj_inputs.get(f"{attn}.o_proj"), n_heads),
            ]
            for role, tensor, nh in head_specs:
                if tensor is None:
                    continue
                heads = _split_heads(tensor, nh, head_dim)
                for h, hv in enumerate(heads):
                    out[f"{attn}.{role}.h{h}"] = hv

    # ---- residual stream ---------------------------------------------------
    for block in [n for n in names if is_block(n)]:
        pre = block_inputs.get(block)
        post = module_taps.get(block)
        if pre is not None:
            out[f"{block}.resid_pre"] = pre
        if post is not None:
            out[f"{block}.resid_post"] = post
        attn = _attn_prefix(block, names)
        attn_out = module_taps.get(f"{attn}.o_proj") if attn else None
        if pre is not None and attn_out is not None \
                and np.asarray(pre).shape == np.asarray(attn_out).shape:
            out[f"{block}.resid_mid"] = np.asarray(pre) + np.asarray(attn_out)
    return out


def classify(tap_name):
    """Coarse site class for a tap, for --site-class selection in the field readout."""
    if tap_name.endswith(".resid_pre") or tap_name.endswith(".resid_mid") \
            or tap_name.endswith(".resid_post"):
        return "resid"
    if re.search(r"\.(q|k|v|z)\.h\d+$", tap_name):
        return "head"
    if ".mlp." in tap_name or tap_name.endswith(".mlp"):
        return "mlp"
    if _ATTN_SUFFIX in tap_name or f".{_ATTN_SUFFIX_ALT}." in tap_name:
        return "attn"
    return "module"


def attention_patterns(hf_model, inputs_embeds):
    """Per-layer per-head softmax attention weights for a REAL sequence.

    Only meaningful when inputs_embeds has S>1. Returns
    {f"layer{L}.head{H}.pattern": np.ndarray (S, S)} or {} if the model can't
    produce attentions. Runs one eager forward with output_attentions=True;
    does not touch the streaming loop.
    """
    try:
        import torch
    except ImportError:
        return {}
    try:
        prev = getattr(hf_model.config, "_attn_implementation", None)
        try:
            hf_model.config._attn_implementation = "eager"
        except Exception:
            pass
        with torch.no_grad():
            out = hf_model(inputs_embeds=inputs_embeds, output_attentions=True)
        atts = getattr(out, "attentions", None)
        if not atts:
            return {}
        res = {}
        for L, a in enumerate(atts):          # a: (B, n_heads, S, S)
            a = a.detach().cpu().numpy()[0]
            for H in range(a.shape[0]):
                res[f"layer{L}.head{H}.pattern"] = a[H]
        return res
    finally:
        try:
            if prev is not None:
                hf_model.config._attn_implementation = prev
        except Exception:
            pass
