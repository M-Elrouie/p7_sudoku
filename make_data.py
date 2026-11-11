"""Download + preprocess MNIST reproducibly (fixed seed) -> data/digits19.npz.

    python make_data.py                  # real MNIST (needs internet), seed 0
    python make_data.py --offline-digits # SMOKE-TEST ONLY: scikit-learn's bundled 8x8 digits

Steps: download 4 gzip files (several mirrors, md5 checked) -> parse IDX -> seeded split of the
60,000 training images into 50,000 train / 10,000 validation -> DROP digit 0 everywhere ->
store uint8 (scaling to 0-1 happens at load time). The 10,000 test images are untouched
(except for dropping 0).
"""
import argparse, gzip, hashlib, os, struct, urllib.request
import numpy as np

MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://yann.lecun.com/exdb/mnist/",
]
FILES = {  # name -> expected md5 of the .gz
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}
HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw")
OUT = os.path.join(HERE, "data", "digits19.npz")


def fetch(name, md5):
    path = os.path.join(RAW, name)
    if not os.path.exists(path):
        os.makedirs(RAW, exist_ok=True)
        for m in MIRRORS:
            try:
                print(f"downloading {m}{name}")
                urllib.request.urlretrieve(m + name, path)
                break
            except Exception as e:  # try next mirror
                print("  failed:", e)
        else:
            raise SystemExit(f"could not download {name}; place it in {RAW} manually")
    got = hashlib.md5(open(path, "rb").read()).hexdigest()
    if got != md5:
        print(f"WARNING: md5 of {name} is {got}, expected {md5}")
    return path


def read_idx(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    magic, n = struct.unpack(">II", data[:8])
    if magic == 2051:
        r, c = struct.unpack(">II", data[8:16])
        return np.frombuffer(data, np.uint8, offset=16).reshape(n, r, c)
    return np.frombuffer(data, np.uint8, offset=8)


def drop_zero(x, y):
    m = y != 0
    return x[m], (y[m] - 1).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--offline-digits", action="store_true",
                    help="smoke tests only: sklearn 8x8 digits upscaled to 28x28 (NOT MNIST)")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    if a.offline_digits:
        from sklearn.datasets import load_digits
        d = load_digits()
        img = (d.images / 16.0)                                   # 8x8 in [0,1]
        img = np.kron(img, np.ones((3, 3)))                       # 24x24
        img = np.pad(img, ((0, 0), (2, 2), (2, 2)))               # 28x28
        x = (img * 255).astype(np.uint8); y = d.target.astype(np.uint8)
        perm = rng.permutation(len(x)); x, y = x[perm], y[perm]
        n = len(x); a1, a2 = int(.7 * n), int(.85 * n)
        parts = {"train": (x[:a1], y[:a1]), "val": (x[a1:a2], y[a1:a2]), "test": (x[a2:], y[a2:])}
        source = "OFFLINE sklearn digits (smoke test only)"
    else:
        xtr = read_idx(fetch("train-images-idx3-ubyte.gz", FILES["train-images-idx3-ubyte.gz"]))
        ytr = read_idx(fetch("train-labels-idx1-ubyte.gz", FILES["train-labels-idx1-ubyte.gz"]))
        xte = read_idx(fetch("t10k-images-idx3-ubyte.gz", FILES["t10k-images-idx3-ubyte.gz"]))
        yte = read_idx(fetch("t10k-labels-idx1-ubyte.gz", FILES["t10k-labels-idx1-ubyte.gz"]))
        perm = rng.permutation(60000)
        tr, va = perm[:50000], perm[50000:]
        parts = {"train": (xtr[tr], ytr[tr]), "val": (xtr[va], ytr[va]), "test": (xte, yte)}
        source = "MNIST"
    out = {}
    for k, (x, y) in parts.items():
        x, y = drop_zero(x, y)
        out[f"x_{k}"], out[f"y_{k}"] = x, y
        print(f"{k:5s}: {len(x)} images (digits 1-9)")
    np.savez_compressed(OUT, source=source, seed=a.seed, **out)
    print("saved", OUT, "| source:", source)


if __name__ == "__main__":
    main()
