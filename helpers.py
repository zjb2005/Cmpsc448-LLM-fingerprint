# stuff that all the scripts need: tokenizing, vocab, the training loop, plots

import re
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

PAD = 0
UNK = 1

LABELS = ["Claude", "GPT", "Koala", "PaLM", "Vicuna"]


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def tokenize(text):
    if text is None:
        return []
    text = str(text).lower()
    # newlines matter a lot here (some models love bullet lists) so i keep them
    # as their own token instead of just deleting them
    text = text.replace("\n", " nwln ")
    toks = re.findall(r"[a-z0-9']+|[^\sa-z0-9']", text)
    return toks


def build_vocab(texts, max_words=20000):
    counts = {}
    for t in texts:
        for w in tokenize(t):
            if w in counts:
                counts[w] = counts[w] + 1
            else:
                counts[w] = 1
    words = sorted(counts.keys(), key=lambda w: counts[w], reverse=True)
    words = words[:max_words]
    vocab = {"<pad>": PAD, "<unk>": UNK}
    for w in words:
        vocab[w] = len(vocab)
    return vocab


def encode(text, vocab, maxlen):
    toks = tokenize(text)[:maxlen]
    ids = []
    for w in toks:
        if w in vocab:
            ids.append(vocab[w])
        else:
            ids.append(UNK)
    while len(ids) < maxlen:
        ids.append(PAD)
    return ids


def encode_all(texts, vocab, maxlen):
    out = []
    for t in texts:
        out.append(encode(t, vocab, maxlen))
    return torch.tensor(out, dtype=torch.long)


def make_text(df, field):
    # which part of the data we feed the model
    if field == "input":
        return list(df["prompt"])
    elif field == "output":
        return list(df["response"])
    elif field == "both":
        out = []
        for p, r in zip(df["prompt"], df["response"]):
            out.append(str(p) + " sepsep " + str(r))
        return out
    else:
        print("bad field:", field)
        exit()


def make_labels(df):
    y = []
    for f in df["family"]:
        y.append(LABELS.index(f))
    return torch.tensor(y, dtype=torch.long)


def load_split(name):
    df = pd.read_csv("data/" + name + ".csv", keep_default_na=False)
    return df


def batches(x, y, bs, shuffle=True):
    idx = list(range(len(y)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        yield x[j], y[j]


def predict(model, x, device, bs=128):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            xb = x[i:i + bs].to(device)
            out = model(xb)
            p = out.argmax(dim=1)
            preds = preds + p.cpu().tolist()
    return preds


def train_model(model, xtr, ytr, xva, yva, device, epochs=8, bs=64, lr=0.001):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()

    best_acc = -1
    best_state = None
    for ep in range(epochs):
        model.train()
        total = 0.0
        nb = 0
        for xb, yb in batches(xtr, ytr, bs):
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = lossf(out, yb)
            loss.backward()
            opt.step()
            total = total + loss.item()
            nb = nb + 1

        p = predict(model, xva, device)
        acc = accuracy_score(yva.tolist(), p)
        print("  epoch", ep + 1, "train loss %.4f" % (total / nb), "val acc %.4f" % acc)
        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(model.state_dict())

    # go back to whichever epoch did best on val
    model.load_state_dict(best_state)
    return model, best_acc


def score(ytrue, ypred):
    acc = accuracy_score(ytrue, ypred)
    f1 = f1_score(ytrue, ypred, average="macro")
    return acc, f1


def plot_cm(ytrue, ypred, title, path):
    cm = confusion_matrix(ytrue, ypred, labels=list(range(len(LABELS))))
    # normalize by row so the numbers are percentages of the true class
    cmn = cm.astype(float)
    for i in range(len(cmn)):
        s = cmn[i].sum()
        if s > 0:
            cmn[i] = cmn[i] / s

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=45, ha="right")
    ax.set_yticklabels(LABELS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            c = "white"
            if cmn[i][j] < 0.5:
                c = "black"
            ax.text(j, i, "%.2f" % cmn[i][j], ha="center", va="center", color=c, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)


def log_result(line):
    f = open("results/summary.txt", "a")
    f.write(line + "\n")
    f.close()
    print(line)
