import os
import random
import pandas as pd
import torch
from sklearn.metrics import classification_report

from helpers import (set_seed, build_vocab, encode_all, make_text, make_labels,
                     load_split, train_model, predict, score, plot_cm, log_result, LABELS)
from models import TextCNN, TextLSTM

MAXLEN = 300
VOCAB_SIZE = 20000
EPOCHS = 8
SEED = 42

if not os.path.exists("results"):
    os.mkdir("results")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
allpieces = [load_split("train"), load_split("val"), load_split("test")]
df = pd.concat(allpieces).reset_index(drop=True)
print("total rows:", len(df))
print(df["task"].value_counts())
seen = df[df["task"] != "writing"]
unseen = df[df["task"] == "writing"]
print("seen tasks:", len(seen), " unseen (writing):", len(unseen))

random.seed(SEED)
qids = sorted(seen["qid"].unique())
random.shuffle(qids)
n = len(qids)
tr_q = set(qids[:int(0.75 * n)])
va_q = set(qids[int(0.75 * n):int(0.85 * n)])
te_q = set(qids[int(0.85 * n):])

train = seen[seen["qid"].isin(tr_q)]
val = seen[seen["qid"].isin(va_q)]
test_in = seen[seen["qid"].isin(te_q)]
test_out = unseen
print("train", len(train), "val", len(val), "in-domain test", len(test_in),
      "out-of-domain test", len(test_out))

def run(which):
    print("=" * 50)
    print("RQ3 with", which)
    set_seed(SEED)

    tr_text = make_text(train, "output")
    vocab = build_vocab(tr_text, VOCAB_SIZE)

    xtr = encode_all(tr_text, vocab, MAXLEN)
    xva = encode_all(make_text(val, "output"), vocab, MAXLEN)
    xin = encode_all(make_text(test_in, "output"), vocab, MAXLEN)
    xout = encode_all(make_text(test_out, "output"), vocab, MAXLEN)
    ytr = make_labels(train)
    yva = make_labels(val)
    yin = make_labels(test_in)
    yout = make_labels(test_out)

    if which == "cnn":
        model = TextCNN(len(vocab), len(LABELS))
    else:
        model = TextLSTM(len(vocab), len(LABELS))

    model, _ = train_model(model, xtr, ytr, xva, yva, device, epochs=EPOCHS)

    p_in = predict(model, xin, device)
    a1, f1a = score(yin.tolist(), p_in)
    p_out = predict(model, xout, device)
    a2, f1b = score(yout.tolist(), p_out)

    log_result("RQ3 %s | in-domain acc %.4f f1 %.4f | writing (unseen) acc %.4f f1 %.4f"
               % (which, a1, f1a, a2, f1b))
    rep = classification_report(yout.tolist(), p_out, labels=list(range(len(LABELS))),
                                target_names=LABELS, digits=3)
    print(rep)
    f = open("results/rq3_%s_unseen_report.txt" % which, "w")
    f.write(rep)
    f.close()
    plot_cm(yout.tolist(), p_out, "%s tested on unseen task (writing)" % which.upper(),
            "results/rq3_cm_%s.png" % which)

run("cnn")
run("lstm")
print()
print("done, numbers are in results/summary.txt")
