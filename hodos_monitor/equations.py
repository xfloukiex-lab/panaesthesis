"""Hodos equations — extracted VERBATIM from hodos-ai-portrait-v1.py.

Do not change the math. Constants, in order of appearance in v1:
  NB=40 quantile bins; per-layer standardization (amendment-13 precedent);
  active-units-only histograms; shared quantile edges across layers;
  gcost = Fisher-Rao geodesic via Bhattacharyya, 2*arccos(BC);
  Diastema = DTW with band=12, path-normalized;
  Symploke = 64 paired obs/step via fixed 64-group split of the early units
            (groups from np.random.default_rng(0).permutation), 12x12 joint,
            M = product of marginals, null = K=8 shuffles of the late index;
            n_pair is a parameter (default 64 = v1); the late side uses
            consecutive groups so n_pair=64 on a 64-wide late layer is v1
            bit-for-bit;
  Chronos = excess = max(eps-null,0), z_arrow = third-moment arrow,
            Phi = Normal CDF of z_arrow, tau = cumsum(excess)*Phi.
"""
from math import erf, sqrt

import numpy as np

NB = 40          # quantile bins (v1)
DTW_BAND = 12    # v1
NJ = 12          # joint histogram bins per side (v1)
K_SHUFFLE = 8    # null shuffles (v1)
N_PAIR = 64      # paired observations per step (v1)


def gcost(p, q):
    """Fisher-Rao geodesic via Bhattacharyya: g(p,q) = 2*arccos(BC). (v1 verbatim)"""
    bc = np.clip(np.sum(np.sqrt(np.maximum(p, 0) * np.maximum(q, 0))), 0, 1)
    return 2 * np.arccos(bc)


def layer_distributions(early, late, nb=NB):
    """Per-step distributions on the shared outcome space. (v1 verbatim)

    Per-layer standardization first (lane amendment-13 precedent: a comparison
    across different scales must be scale-free; what is compared is the SHAPE of
    each layer's activation distribution). One fixed affine map per layer, applied
    identically at every step. Distributions are over ACTIVE units only: the ReLU
    zero-mass is structural (dead units), not signal; keeping it collapses the
    quantile bins into degeneracy. Symploke/Chronos below use quantile edges and
    are invariant under the monotonic rescaling.
    """
    T = early.shape[0]
    early_z = (early - early.mean()) / (early.std() + 1e-12)
    late_z = (late - late.mean()) / (late.std() + 1e-12)
    act_e = early != 0
    act_l = late != 0
    pooled = np.concatenate([early_z[act_e], late_z[act_l]])
    edges = np.quantile(pooled, np.linspace(0, 1, nb + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    P, Q = np.zeros((T, nb)), np.zeros((T, nb))
    for t in range(T):
        he = np.histogram(early_z[t][act_e[t]], bins=edges)[0]
        hl = np.histogram(late_z[t][act_l[t]], bins=edges)[0]
        P[t] = he / he.sum() if he.sum() else np.full(nb, 1 / nb)
        Q[t] = hl / hl.sum() if hl.sum() else np.full(nb, 1 / nb)
    return P, Q


def diastema(P, Q, band=DTW_BAND):
    """DTW between the two layer-processes. (v1 verbatim)

    Returns (D_total, gap): path-normalized minimum accumulated ground distance,
    and the per-frame gap curve (engine-derived visualization, not the paper's
    primary scalar form).
    """
    T = P.shape[0]
    C = np.zeros((T, T))
    sqQ = np.sqrt(Q)
    for i in range(T):
        C[i] = 2 * np.arccos(np.clip((np.sqrt(P[i]) * sqQ).sum(axis=1), 0, 1))

    Dm = np.full((T + 1, T + 1), np.inf)
    Dm[0, 0] = 0.0
    for i in range(1, T + 1):
        j0 = max(1, i - band)
        j1 = min(T, i + band)
        prev = np.minimum(np.minimum(Dm[i - 1, j0 - 1:j1], Dm[i, j0 - 1:j1]),
                          Dm[i - 1, j0:j1 + 1])
        Dm[i, j0:j1 + 1] = C[i - 1, j0 - 1:j1] + prev
    i, j = T, T
    path = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        cands = [(Dm[i - 1, j - 1], i - 1, j - 1), (Dm[i - 1, j], i - 1, j),
                 (Dm[i, j - 1], i, j - 1)]
        _, i, j = min(cands)
    path = path[::-1]
    D_total = Dm[T, T] / len(path)
    gap = np.zeros(T)
    for i in range(T):
        gap[i] = np.mean([C[ii, jj] for (ii, jj) in path if ii == i])
    return D_total, gap


def grouped_means(vec, n_groups, seed=0):
    """Split one step's unit vector into n_groups and average each.

    seed=None -> consecutive groups (no permutation); otherwise a fixed
    permutation from the seed. Deterministic in both cases.
    """
    n = vec.shape[0]
    idx = np.arange(n) if seed is None else \
        np.random.default_rng(seed).permutation(n)
    groups = np.array_split(idx, n_groups)
    return np.array([vec[g].mean() for g in groups])


def symploke(early, late, rng, nj=NJ, k=K_SHUFFLE, n_pair=N_PAIR):
    """epsilon(t) = g(J_t, M_t). (v1 verbatim at n_pair=64)

    J_t: joint of (early, late) activations at t. The early units are split
    into n_pair fixed groups (from np.random.default_rng(0).permutation);
    each group's mean is paired with one late group-mean -> n_pair genuine
    paired observations per step (no tiling/duplication). 12x12 joint
    histogram; M_t = product of marginals; null = k shuffles of the late
    index, drawn from the caller's rng (which must have produced the
    stimulus first, exactly as in v1).
    The late side uses CONSECUTIVE groups (no permutation): when the late
    layer has exactly n_pair units this is the identity, so v1 runs are
    bit-identical. Wider late layers are averaged into n_pair groups.
    n_pair must be <= the late width; otherwise the pairing is undefined
    and this raises instead of guessing.
    Returns (eps, null_mean, null_std, z).
    """
    T = early.shape[0]
    if late.shape[1] < n_pair:
        raise ValueError(
            f"symploke needs at least n_pair={n_pair} late units, got "
            f"{late.shape[1]} — pass a smaller --n-pair")
    A = np.array([grouped_means(early[t], n_pair, seed=0) for t in range(T)])
    B = np.array([grouped_means(late[t], n_pair, seed=None) for t in range(T)])
    eE = np.quantile(A, np.linspace(0, 1, nj + 1))
    eE[0] -= 1e-9
    eE[-1] += 1e-9
    eL = np.quantile(B, np.linspace(0, 1, nj + 1))
    eL[0] -= 1e-9
    eL[-1] += 1e-9
    eps = np.zeros(T)
    nullm = np.zeros(T)
    nulls = np.zeros(T)
    for t in range(T):
        ia = np.clip(np.digitize(A[t], eE) - 1, 0, nj - 1)
        ib = np.clip(np.digitize(B[t], eL) - 1, 0, nj - 1)
        J = np.zeros((nj, nj))
        np.add.at(J, (ia, ib), 1)
        J /= J.sum()
        M = np.outer(J.sum(1), J.sum(0))
        eps[t] = gcost(J.ravel(), M.ravel())
        ns = []
        for _ in range(k):
            ib2 = rng.permutation(ib)
            J2 = np.zeros((nj, nj))
            np.add.at(J2, (ia, ib2), 1)
            J2 /= J2.sum()
            ns.append(gcost(J2.ravel(), M.ravel()))
        nullm[t] = np.mean(ns)
        nulls[t] = np.std(ns) + 1e-12
    z = (eps - nullm) / nulls
    return eps, nullm, nulls, z


def chronos(eps, nullm):
    """Lived time. (v1 verbatim) Returns (excess, z_arrow, Phi, tau)."""
    u = eps - nullm
    excess = np.maximum(u, 0)
    d = np.diff(u)
    z_arrow = (d ** 3).sum() / sqrt((d ** 6).sum())
    Phi = 0.5 * (1 + erf(z_arrow / sqrt(2)))
    tau = np.cumsum(excess) * Phi
    return excess, z_arrow, Phi, tau


def verify(z, gap, tau, hard_idx, easy_idx):
    """The v1 verification block, verbatim logic: report mismatch, never force."""
    notes = []
    ok = True
    if not (z[hard_idx].mean() > z[easy_idx].mean() + 1.0):
        notes.append("NOTE: epsilon not elevated during strain — reporting, not forcing")
        ok = False
    if not (gap.std() > 1e-6 and gap.max() - gap.min() > 1e-6):
        notes.append("NOTE: gap curve degenerate")
        ok = False
    if not (tau[-1] > 0 and np.all(np.diff(tau) >= -1e-12)):
        notes.append("NOTE: tau not accumulating")
        ok = False
    return {"passed": bool(ok),
            "status": "PASSED" if ok else "FLAGGED (see notes)",
            "notes": notes}
