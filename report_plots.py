# extra plots for the write up. reads results/summary.txt for the bar charts.
# the val curves are copied by hand from the colab output since train.py
# only prints them (should have saved them to a file, oh well)
#
# usage: python report_plots.py

import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

lines = open("results/summary.txt").read().strip().split("\n")

# ---- RQ1/RQ2 bar chart ----
acc = {}
for ln in lines:
    m = re.match(r"(cnn|lstm) \| (\w+) \| val_acc [\d.]+ \| test_acc ([\d.]+)", ln)
    if m:
        acc[(m.group(1), m.group(2))] = float(m.group(3))

fields = ["input", "output", "both"]
fig, ax = plt.subplots(figsize=(6, 3.8))
w = 0.35
for i, f in enumerate(fields):
    ax.bar(i - w / 2, acc[("cnn", f)], w, color="steelblue", label="CNN" if i == 0 else None)
    ax.bar(i + w / 2, acc[("lstm", f)], w, color="darkorange", label="LSTM" if i == 0 else None)
    ax.text(i - w / 2, acc[("cnn", f)] + 0.01, "%.3f" % acc[("cnn", f)], ha="center", fontsize=8)
    ax.text(i + w / 2, acc[("lstm", f)] + 0.01, "%.3f" % acc[("lstm", f)], ha="center", fontsize=8)
ax.axhline(0.2, color="gray", linestyle="--", linewidth=1, label="chance (0.2)")
ax.set_xticks(range(3))
ax.set_xticklabels(["input only", "output only", "input + output"])
ax.set_ylabel("test accuracy")
ax.set_ylim(0, 0.75)
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig("results/rq2_bars.png", dpi=150)
plt.close(fig)
print("saved results/rq2_bars.png")

# ---- RQ4 ablation chart ----
names = []
accs = []
for ln in lines:
    m = re.match(r"RQ4 ablation \[(.+)\] \| cnn test_acc ([\d.]+)", ln)
    if m:
        names.append(m.group(1))
        accs.append(float(m.group(2)))
m = re.search(r"logistic regression.*test_acc ([\d.]+)", "\n".join(lines))
names.append("log. reg. on 11 features")
accs.append(float(m.group(1)))

short_names = ["full text\n(baseline)", "no punctuation\nor markdown", "symbols and\nlayout only",
               "first 50\nwords only", "log. reg. on\n11 features"]
fig, ax = plt.subplots(figsize=(7, 3.6))
cols = ["steelblue", "steelblue", "steelblue", "steelblue", "gray"]
ax.bar(range(len(accs)), accs, color=cols)
for i in range(len(accs)):
    ax.text(i, accs[i] + 0.01, "%.3f" % accs[i], ha="center", fontsize=8)
ax.axhline(0.2, color="gray", linestyle="--", linewidth=1)
ax.set_xticks(range(len(accs)))
ax.set_xticklabels(short_names, fontsize=8)
ax.set_ylabel("test accuracy")
ax.set_ylim(0, 0.7)
fig.tight_layout()
fig.savefig("results/rq4_ablation_bars.png", dpi=150)
plt.close(fig)
print("saved results/rq4_ablation_bars.png")

# ---- validation accuracy per epoch (copied from the training log) ----
curves = {
    "CNN output": [0.5011, 0.5841, 0.5673, 0.6087, 0.6188, 0.6379, 0.6362, 0.6424],
    "LSTM output": [0.4456, 0.5370, 0.5729, 0.5891, 0.6143, 0.6295, 0.6407, 0.6087],
    "CNN both": [0.4619, 0.5297, 0.5420, 0.5443, 0.5779, 0.5959, 0.5925, 0.5981],
    "LSTM both": [0.4019, 0.4535, 0.5437, 0.5561, 0.5577, 0.5790, 0.5863, 0.5790],
    "CNN input": [0.2152, 0.2220, 0.2180, 0.2248, 0.2180, 0.2321, 0.2220, 0.2209],
    "LSTM input": [0.2214, 0.2085, 0.2220, 0.2180, 0.2209, 0.2231, 0.2164, 0.2203],
}
loss_in = {
    "CNN input": [1.7371, 1.6115, 1.5456, 1.4948, 1.4525, 1.4090, 1.3378, 1.2857],
    "LSTM input": [1.6238, 1.5975, 1.5765, 1.5535, 1.5049, 1.4524, 1.3819, 1.3005],
}
ep = list(range(1, 9))
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for k in curves:
    style = "-"
    if "LSTM" in k:
        style = "--"
    axes[0].plot(ep, curves[k], style, marker="o", markersize=3, label=k)
axes[0].set_xlabel("epoch")
axes[0].set_ylabel("validation accuracy")
axes[0].set_title("val accuracy per epoch", fontsize=10)
axes[0].legend(fontsize=7)
for k in loss_in:
    axes[1].plot(ep, loss_in[k], marker="o", markersize=3, label=k + " (train loss)")
axes[1].set_xlabel("epoch")
axes[1].set_ylabel("training loss")
axes[1].set_title("input only: train loss keeps falling", fontsize=10)
axes[1].legend(fontsize=7)
fig.tight_layout()
fig.savefig("results/curves.png", dpi=150)
plt.close(fig)
print("saved results/curves.png")
