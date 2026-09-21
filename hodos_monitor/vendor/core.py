#!/usr/bin/env python3
"""Tiny-brain neural net core: pure numpy. No frameworks.

Implements exactly what the spec needs and nothing more:
  Conv2d (im2col, stride 1) / ReLU / MaxPool2d(2x2,s2) / Affine / Dropout /
  SoftmaxCrossEntropy / Adam.

All layers: forward(x) -> out, backward(dout) -> dx. Parameters live on the
layer as .W/.b with .dW/.db after backward. Deterministic given the init seed.
"""
import numpy as np


# --- im2col / col2im (stride 1, no padding) ------------------------------------
def im2col(x, kh, kw):
    """x: (N,C,H,W) -> col: (N*OH*OW, C*kh*kw), cache."""
    N, C, H, W = x.shape
    OH, OW = H - kh + 1, W - kw + 1
    sN, sC, sH, sW = x.strides
    shape = (N, OH, OW, C, kh, kw)
    strides = (sN, sH, sW, sC, sH, sW)
    cols = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return cols.reshape(N * OH * OW, C * kh * kw).copy(), (N, C, H, W, OH, OW)


def col2im(dcol, cache, kh, kw):
    """dcol: (N*OH*OW, C*kh*kw) -> dx: (N,C,H,W)."""
    N, C, H, W, OH, OW = cache
    d6 = dcol.reshape(N, OH, OW, C, kh, kw)
    dx = np.zeros((N, C, H, W), dtype=dcol.dtype)
    for i in range(kh):
        for j in range(kw):
            patch = d6[:, :, :, :, i, j].transpose(0, 3, 1, 2)  # (N,C,OH,OW)
            dx[:, :, i:i + OH, j:j + OW] += patch
    return dx


class Conv2d:
    def __init__(self, in_ch, out_ch, k, rng):
        s = np.sqrt(2.0 / (in_ch * k * k))  # He
        self.W = rng.normal(0, s, (out_ch, in_ch, k, k)).astype(np.float64)
        self.b = np.zeros(out_ch, dtype=np.float64)
        self.k = k
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        kh = kw = self.k
        self.col, self.cache = im2col(x, kh, kw)
        N, C, H, W, OH, OW = self.cache
        out_ch = self.W.shape[0]
        Wcol = self.W.reshape(out_ch, -1).T  # (C*kh*kw, out_ch)
        out = self.col @ Wcol + self.b  # (N*OH*OW, out_ch)
        return out.reshape(N, OH, OW, out_ch).transpose(0, 3, 1, 2)

    def backward(self, dout):
        # dout: (N, out_ch, OH, OW)
        N, out_ch, OH, OW = dout.shape
        dout_r = dout.transpose(0, 2, 3, 1).reshape(-1, out_ch)
        Wcol = self.W.reshape(out_ch, -1).T
        self.db = dout_r.sum(axis=0)
        self.dW = (dout_r.T @ self.col).reshape(self.W.shape)
        dcol = dout_r @ Wcol.T
        return col2im(dcol, self.cache, self.k, self.k)


class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return np.where(self.mask, x, 0.0)

    def backward(self, dout):
        return np.where(self.mask, dout, 0.0)


class MaxPool2d:
    """2x2 stride 2. Requires even H, W."""

    def forward(self, x):
        N, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0
        self.xr = x.reshape(N, C, H // 2, 2, W // 2, 2)
        self.out = self.xr.max(axis=(3, 5))
        return self.out

    def backward(self, dout):
        # distribute over ties
        m = (self.xr == self.out[:, :, :, None, :, None])
        denom = m.sum(axis=(3, 5), keepdims=True)
        dxr = m * (dout[:, :, :, None, :, None] / np.maximum(denom, 1))
        return dxr.reshape(self.xr.shape[0], self.xr.shape[1],
                           self.xr.shape[2] * 2, self.xr.shape[4] * 2)


class Affine:
    def __init__(self, in_f, out_f, rng):
        s = np.sqrt(2.0 / in_f)
        self.W = rng.normal(0, s, (in_f, out_f)).astype(np.float64)
        self.b = np.zeros(out_f, dtype=np.float64)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        self.x_shape = x.shape
        self.x = x.reshape(x.shape[0], -1)
        return self.x @ self.W + self.b

    def backward(self, dout):
        self.dW = self.x.T @ dout
        self.db = dout.sum(axis=0)
        dx = dout @ self.W.T
        return dx.reshape(self.x_shape)


class Dropout:
    def __init__(self, p, rng):
        self.p = p
        self.rng = rng
        self.train = True

    def forward(self, x):
        if not self.train or self.p <= 0:
            return x
        self.mask = (self.rng.random(x.shape) > self.p) / (1.0 - self.p)
        return x * self.mask

    def backward(self, dout):
        if not self.train or self.p <= 0:
            return dout
        return dout * self.mask


class SoftmaxCrossEntropy:
    def forward(self, logits, y):
        # logits: (N,K), y: (N,) int
        z = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(z)
        self.probs = e / e.sum(axis=1, keepdims=True)
        self.y = y
        N = logits.shape[0]
        return -np.log(self.probs[np.arange(N), y] + 1e-12).mean()

    def backward(self):
        N = self.probs.shape[0]
        d = self.probs.copy()
        d[np.arange(N), self.y] -= 1
        return d / N

    def predict(self, logits):
        return logits.argmax(axis=1)


class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        # params: list of (param_array, grad_array) tuples, refreshed each step
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.t = 0
        self.state = {}  # id -> (m, v)

    def step(self, params):
        self.t += 1
        for p, g in params:
            key = id(p)
            if key not in self.state:
                self.state[key] = (np.zeros_like(p), np.zeros_like(p))
            m, v = self.state[key]
            m[:] = self.b1 * m + (1 - self.b1) * g
            v[:] = self.b2 * v + (1 - self.b2) * g * g
            mh = m / (1 - self.b1 ** self.t)
            vh = v / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


# --- model helpers --------------------------------------------------------------
def collect_params(layers):
    out = []
    for l in layers:
        if hasattr(l, "W"):
            out.append((l.W, l.dW))
            out.append((l.b, l.db))
    return out


def forward(layers, x, train=True):
    for l in layers:
        if isinstance(l, Dropout):
            l.train = train
        x = l.forward(x)
    return x


def zero_grad(layers):
    for p, g in collect_params(layers):
        g.fill(0.0)
