"""LoRA readout: the "knowing" that a spliced relation needs.

The splicer installs a relation pattern on demand (no training). A learned
relation matters because downstream weights grew up using it; an installed
one just sits there. The LoRA readout is the minimal bridge: a tiny
low-rank adapter bolted onto the frozen model that learns, from a few
labeled steps, to READ the spliced activations.

Mechanism (model-agnostic, pure numpy — it only needs features, base
logits, and labels, so it works for any adapter):
    logits = Z0 + (F @ A.T) @ B.T
where Z0 are the frozen model's own logits, F the late-layer features,
A (rank x d) and B (C x rank) the only trained parameters. B starts at
zero, so training starts from exactly the base model — the adapter is a
pure residual correction. This is LoRA in spirit: the base weights never
move; a small add-on learns the new readout.

The honest experiment the instrument runs:
    LoRA_nat : trained on NATURAL late features   -> acc_lora_nat
    LoRA_spl : trained on SPLICED late features   -> acc_lora_spl
Both get the same rank, steps, learning rate, and data. If
acc_lora_spl > acc_lora_nat, the installed relation did work a readout
could use. If equal, the relation sits there unused — reported, not
hidden. The control matters: a LoRA alone can denoise even without a
splice, so the splice's contribution is the DELTA, never the raw number.
"""
import numpy as np


def train_residual_lora(F, Z0, y, rank=4, steps=300, lr=0.5, seed=0,
                        l2=1e-4):
    """Train the residual LoRA (A, B) on features F with base logits Z0.

    F  : (T, d) late-layer features
    Z0 : (T, C) the frozen model's own logits on the same steps
    y  : (T,) integer labels
    Returns (A, B, losses) with A (rank, d), B (C, rank).
    """
    F = np.asarray(F, dtype=np.float64)
    Z0 = np.asarray(Z0, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    T, d = F.shape
    C = Z0.shape[1]
    if T == 0:
        raise ValueError("train_residual_lora needs at least one step")
    if rank < 1 or rank > min(d, C):
        raise ValueError(
            f"rank must be in [1, min(d, C)]={min(d, C)}, got {rank}")
    rng = np.random.default_rng(seed)
    A = rng.normal(scale=0.01, size=(rank, d))
    B = np.zeros((C, rank))  # adapter starts as a no-op
    Y = np.zeros((T, C))
    Y[np.arange(T), y] = 1.0
    losses = []
    for _ in range(steps):
        H = F @ A.T                    # (T, r)
        L = Z0 + H @ B.T               # (T, C)
        Lm = L - L.max(axis=1, keepdims=True)
        E = np.exp(Lm)
        P = E / E.sum(axis=1, keepdims=True)
        G = (P - Y) / T                # dL/dlogits, mean over steps
        dB = G.T @ H + l2 * B
        dA = (G @ B).T @ F + l2 * A
        B -= lr * dB
        A -= lr * dA
        nll = float(-np.log(np.clip(P[np.arange(T), y], 1e-12, 1)).mean())
        losses.append(nll)
    return A, B, losses


def apply_lora(F, Z0, A, B):
    """Logits with the trained residual adapter applied."""
    F = np.asarray(F, dtype=np.float64)
    Z0 = np.asarray(Z0, dtype=np.float64)
    return Z0 + (F @ A.T) @ B.T


def accuracy_from_logits(logits, y):
    preds = np.asarray(logits).argmax(axis=1)
    return float((preds == np.asarray(y, dtype=int)).mean())
