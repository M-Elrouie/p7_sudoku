"""Train the digit classifier + logistic-regression baseline, calibrate, save, print metrics."""
import argparse, json, os
import numpy as np
from p7.data import load_digits
from p7.nn import SoftmaxNet, accuracy, ece, nll, confusion_matrix, fit_temperature, softmax

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--hidden", type=int, default=128)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

d = load_digits()
print("data source:", d["source"])
os.makedirs("results", exist_ok=True)
rows, models = {}, {}
for name, sizes in (("mlp", [784, a.hidden, 9]), ("logreg", [784, 9])):
    print(f"training {name} {sizes}")
    net = SoftmaxNet(sizes, seed=a.seed).fit(d["x_train"], d["y_train"], d["x_val"], d["y_val"],
                                              epochs=a.epochs, seed=a.seed)
    models[name] = net
    if name == "mlp":  # temperature scaling on VALIDATION logits only
        net.T = fit_temperature(net.logits(d["x_val"]), d["y_val"])
    for T_label, T in (("raw", 1.0), ("scaled", net.T)):
        if name == "logreg" and T_label == "scaled":
            continue
        p = softmax(net.logits(d["x_test"]) / T)
        rows[f"{name}_{T_label}"] = dict(T=T, test_acc=accuracy(p, d["y_test"]),
                                          test_nll=nll(p, d["y_test"]), test_ece=ece(p, d["y_test"]))
net = models["mlp"]
cm = confusion_matrix(net.predict(d["x_test"]), d["y_test"])
np.savetxt("results/confusion_matrix.csv", cm, fmt="%d", delimiter=",")
json.dump(rows, open("results/nn_metrics.json", "w"), indent=2)
net.save("data/mlp.npz"); models["logreg"].save("data/logreg.npz")
print(f"\n{'model':14s} {'T':>5s} {'acc':>7s} {'NLL':>7s} {'ECE':>7s}")
for k, r in rows.items():
    print(f"{k:14s} {r['T']:5.2f} {r['test_acc']:7.4f} {r['test_nll']:7.4f} {r['test_ece']:7.4f}")
print("confusion matrix saved to results/confusion_matrix.csv")
