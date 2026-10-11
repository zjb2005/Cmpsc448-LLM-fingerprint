import os
import sys
import torch
from sklearn.metrics import classification_report

from helpers import (set_seed, build_vocab, encode_all, make_text, make_labels,
                     load_split, train_model, predict, score, plot_cm, log_result, LABELS)
from models import TextCNN, TextLSTM

MAXLEN = 300
VOCAB_SIZE = 20000
EPOCHS = 8
BATCH = 64
SEED = 42

def run_one(which, field, train, val, test, device):
    print("=" * 50)
    print("running", which, "on", field)
    set_seed(SEED)

    tr_text = make_text(train, field)
    va_text = make_text(val, field)
    te_text = make_text(test, field)

    vocab = build_vocab(tr_text, VOCAB_SIZE)
    print("vocab size:", len(vocab))

    xtr = encode_all(tr_text, vocab, MAXLEN)
    xva = encode_all(va_text, vocab, MAXLEN)
    xte = encode_all(te_text, vocab, MAXLEN)
    ytr = make_labels(train)
    yva = make_labels(val)
    yte = make_labels(test)

    if which == "cnn":
        model = TextCNN(len(vocab), len(LABELS))
    else:
        model = TextLSTM(len(vocab), len(LABELS))

    model, val_acc = train_model(model, xtr, ytr, xva, yva, device,
                                 epochs=EPOCHS, bs=BATCH)

    pred = predict(model, xte, device)
    acc, f1 = score(yte.tolist(), pred)
    log_result("%s | %s | val_acc %.4f | test_acc %.4f | macro_f1 %.4f"
               % (which, field, val_acc, acc, f1))

    rep = classification_report(yte.tolist(), pred, labels=list(range(len(LABELS))),
                                target_names=LABELS, digits=3)
    print(rep)
    f = open("results/report_%s_%s.txt" % (which, field), "w")
    f.write(rep)
    f.close()

    plot_cm(yte.tolist(), pred, "%s, %s only" % (which.upper(), field),
            "results/cm_%s_%s.png" % (which, field))
    return acc, f1

if not os.path.exists("results"):
    os.mkdir("results")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

train = load_split("train")
val = load_split("val")
test = load_split("test")
print("loaded", len(train), len(val), len(test))

if len(sys.argv) < 2 or sys.argv[1] == "all":
    jobs = []
    for m in ["cnn", "lstm"]:
        for fl in ["output", "input", "both"]:
            jobs.append((m, fl))
else:
    jobs = [(sys.argv[1], sys.argv[2])]

for m, fl in jobs:
    run_one(m, fl, train, val, test, device)

print()
print("all done, check results/summary.txt")
