"""Small numpy neural network (MLP) + softmax-regression baseline + calibration tools.

`SoftmaxNet([784, 128, 9])` is the MLP; `SoftmaxNet([784, 9])` is multinomial logistic
regression (the baseline). Same training code, so the comparison is fair.
Labels are 0..8 (digit - 1).
"""
from __future__ import annotations

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class SoftmaxNet:
    def __init__(self, sizes, seed: int = 0):
        self.sizes = list(sizes)
        rng = np.random.default_rng(seed)
        self.W = [rng.normal(0, np.sqrt(2.0 / a), (a, b)).astype(np.float32)
                  for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b, dtype=np.float32) for b in sizes[1:]]
        self.T = 1.0  # temperature (set by temperature scaling)

    # ---- forward / backward -------------------------------------------------
    def logits(self, x: np.ndarray) -> np.ndarray:
        h = x
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if i < len(self.W) - 1:
                h = np.maximum(h, 0)
        return h

    def probs(self, x: np.ndarray, T: float | None = None) -> np.ndarray:
        return softmax(self.logits(x) / (self.T if T is None else T))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.logits(x).argmax(axis=1)

    def _grads(self, x, y, l2):
        acts = [x]
        h = x
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if i < len(self.W) - 1:
                h = np.maximum(h, 0)
            acts.append(h)
        p = softmax(h)
        n = len(x)
        d = p
        d[np.arange(n), y] -= 1
        d /= n
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            gW[i] = acts[i].T @ d + l2 * self.W[i]
            gb[i] = d.sum(axis=0)
            if i > 0:
                d = (d @ self.W[i].T) * (acts[i] > 0)
        return gW, gb

    # ---- training (Adam) ----------------------------------------------------
    def fit(self, xtr, ytr, xval=None, yval=None, epochs=10, lr=1e-3, batch=128,
            l2=1e-5, seed=0, verbose=True):
        rng = np.random.default_rng(seed)
        mW = [np.zeros_like(w) for w in self.W]; vW = [np.zeros_like(w) for w in self.W]
        mb = [np.zeros_like(b) for b in self.b]; vb = [np.zeros_like(b) for b in self.b]
        t = 0
        for ep in range(epochs):
            idx = rng.permutation(len(xtr))
            for s in range(0, len(idx), batch):
                j = idx[s:s + batch]
                gW, gb = self._grads(xtr[j], ytr[j], l2)
                t += 1
                for i in range(len(self.W)):
                    for (p, g, m, v) in ((self.W, gW, mW, vW), (self.b, gb, mb, vb)):
                        m[i] = 0.9 * m[i] + 0.1 * g[i]
                        v[i] = 0.999 * v[i] + 0.001 * g[i] ** 2
                        mh = m[i] / (1 - 0.9 ** t); vh = v[i] / (1 - 0.999 ** t)
                        p[i] -= (lr * mh / (np.sqrt(vh) + 1e-8)).astype(np.float32)
            if verbose and xval is not None:
                print(f"  epoch {ep + 1:2d}/{epochs}  val acc = {(self.predict(xval) == yval).mean():.4f}")
        return self

    # ---- persistence --------------------------------------------------------
    def save(self, path: str) -> None:
        d = {f"W{i}": w for i, w in enumerate(self.W)}
        d.update({f"b{i}": b for i, b in enumerate(self.b)})
        np.savez(path, sizes=np.array(self.sizes), T=np.array(self.T), **d)

    @classmethod
    def load(cls, path: str) -> "SoftmaxNet":
        z = np.load(path)
        net = cls(list(z["sizes"]))
        net.W = [z[f"W{i}"] for i in range(len(net.W))]
        net.b = [z[f"b{i}"] for i in range(len(net.b))]
        net.T = float(z["T"])
        return net


# ---- metrics ---------------------------------------------------------------
def accuracy(p: np.ndarray, y: np.ndarray) -> float:
    return float((p.argmax(axis=1) == y).mean())


def nll(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())


def ece(p: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error: sum_b (n_b/N) |acc_b - conf_b| over confidence bins."""
    conf = p.max(axis=1)
    hit = (p.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return float(total)


def confusion_matrix(pred: np.ndarray, y: np.ndarray, k: int = 9) -> np.ndarray:
    cm = np.zeros((k, k), dtype=int)
    np.add.at(cm, (y, pred), 1)
    return cm


def fit_temperature(logits: np.ndarray, y: np.ndarray, lo=0.05, hi=10.0, iters=60) -> float:
    """Temperature scaling: T minimising validation NLL (golden-section search)."""
    f = lambda T: nll(softmax(logits / T), y)
    gr = (np.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - gr * (b - a), a + gr * (b - a)
    for _ in range(iters):
        if f(c) < f(d):
            b = d
        else:
            a = c
        c, d = b - gr * (b - a), a + gr * (b - a)
    return float((a + b) / 2)
