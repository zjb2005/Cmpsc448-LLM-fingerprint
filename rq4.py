# RQ4 (extra credit): what is actually different between these models?
# part 1 - measure surface features per family (length, vocabulary, punctuation, markdown)
# part 2 - logistic regression on just those hand made features
# part 3 - ablations: take a signal away, retrain the CNN, see how much accuracy drops
#
# usage: python rq4.py

import os
import re
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from helpers import (set_seed, build_vocab, encode_all, make_labels, load_split,
                     train_model, predict, score, log_result, tokenize, LABELS)
from models import TextCNN

MAXLEN = 300
VOCAB_SIZE = 20000
EPOCHS = 8
SEED = 42

if not os.path.exists("results"):
    os.mkdir("results")

device = "cuda" if torch.cuda.is_available() else "cpu"
train = load_split("train")
val = load_split("val")
test = load_split("test")


# ---------------- part 1: descriptive stats ----------------

def feats(text):
    text = str(text)
    words = text.split()
    nw = len(words)
    if nw == 0:
        nw = 1
    sents = re.split(r"[.!?]+", text)
    sents = [s for s in sents if len(s.strip()) > 0]
    if len(sents) == 0:
        sents = [text]
    uniq = len(set(w.lower() for w in words))
    punct = 0
    for ch in text:
        if ch in ".,;:!?-()\"'":
            punct = punct + 1
    bullets = len(re.findall(r"(?m)^\s*[-*•]\s", text))
    numbered = len(re.findall(r"(?m)^\s*\d+[.)]\s", text))
    bold = text.count("**")
    heads = len(re.findall(r"(?m)^#+\s", text))
    code = text.count("```")
    return {
        "n_words": len(words),
        "avg_word_len": np.mean([len(w) for w in words]) if len(words) > 0 else 0,
        "n_sents": len(sents),
        "avg_sent_len": len(words) / len(sents),
        "ttr": uniq / nw,
        "punct_per_100w": 100.0 * punct / nw,
        "bullets": bullets,
        "numbered": numbered,
        "bold": bold,
        "headings": heads,
        "code_blocks": code,
    }


print("computing features on the training set...")
rows = []
for r, fam in zip(train["response"], train["family"]):
    d = feats(r)
    d["family"] = fam
    rows.append(d)
fdf = pd.DataFrame(rows)

table = fdf.groupby("family").mean(numeric_only=True).round(3)
print(table)
table.to_csv("results/rq4_feature_table.csv")
print("saved results/rq4_feature_table.csv")

# a couple of plots for the report
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
for ax, col, name in zip(axes,
                         ["n_words", "ttr", "bullets"],
                         ["mean response length (words)",
                          "type-token ratio",
                          "mean bullet lines per response"]):
    vals = table[col]
    ax.bar(range(len(vals)), vals.values, color="steelblue")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(vals.index, rotation=30, ha="right")
    ax.set_title(name, fontsize=10)
fig.tight_layout()
fig.savefig("results/rq4_features.png", dpi=150)
plt.close(fig)
print("saved results/rq4_features.png")

# length distribution, this one usually looks pretty different per model
fig, ax = plt.subplots(figsize=(6, 4))
for fam in LABELS:
    v = fdf[fdf["family"] == fam]["n_words"]
    ax.hist(v, bins=40, range=(0, 600), histtype="step", label=fam)
ax.set_xlabel("response length (words)")
ax.set_ylabel("count")
ax.legend()
fig.tight_layout()
fig.savefig("results/rq4_length_hist.png", dpi=150)
plt.close(fig)
print("saved results/rq4_length_hist.png")


# ---------------- part 2: logistic regression on the hand features ----------------

cols = ["n_words", "avg_word_len", "n_sents", "avg_sent_len", "ttr",
        "punct_per_100w", "bullets", "numbered", "bold", "headings", "code_blocks"]


def feat_matrix(df):
    rows = []
    for r in df["response"]:
        d = feats(r)
        rows.append([d[c] for c in cols])
    return np.array(rows)


Xtr = feat_matrix(train)
Xte = feat_matrix(test)
ytr = [LABELS.index(f) for f in train["family"]]
yte = [LABELS.index(f) for f in test["family"]]

# scale it by hand, the length column is way bigger than the rest
mu = Xtr.mean(axis=0)
sd = Xtr.std(axis=0)
sd[sd == 0] = 1.0
Xtr = (Xtr - mu) / sd
Xte = (Xte - mu) / sd

lr = LogisticRegression(max_iter=2000)
lr.fit(Xtr, ytr)
pred = lr.predict(Xte)
a, f1 = score(yte, pred)
log_result("RQ4 logistic regression on %d hand features | test_acc %.4f macro_f1 %.4f"
           % (len(cols), a, f1))


# ---------------- part 3: ablations ----------------

def no_symbols(t):
    # kill punctuation, markdown, newlines. only words are left
    return re.sub(r"[^A-Za-z0-9\s]", " ", str(t))


def only_symbols(t):
    # the opposite, throw the words away and keep the layout
    return re.sub(r"[A-Za-z0-9]+", " ", str(t))


def short(t):
    return " ".join(str(t).split()[:50])


def run_ablation(name, fn):
    print("=" * 50)
    print("ablation:", name)
    set_seed(SEED)
    tr_text = [fn(t) for t in train["response"]]
    va_text = [fn(t) for t in val["response"]]
    te_text = [fn(t) for t in test["response"]]

    vocab = build_vocab(tr_text, VOCAB_SIZE)
    xtr = encode_all(tr_text, vocab, MAXLEN)
    xva = encode_all(va_text, vocab, MAXLEN)
    xte = encode_all(te_text, vocab, MAXLEN)
    ytr2 = make_labels(train)
    yva2 = make_labels(val)
    yte2 = make_labels(test)

    model = TextCNN(len(vocab), len(LABELS))
    model, _ = train_model(model, xtr, ytr2, xva, yva2, device, epochs=EPOCHS)
    p = predict(model, xte, device)
    a, f1 = score(yte2.tolist(), p)
    log_result("RQ4 ablation [%s] | cnn test_acc %.4f macro_f1 %.4f" % (name, a, f1))


run_ablation("full text (baseline)", lambda t: str(t))
run_ablation("no punctuation or markdown", no_symbols)
run_ablation("symbols and layout only", only_symbols)
run_ablation("first 50 words only", short)

print()
print("done, everything is in results/")
