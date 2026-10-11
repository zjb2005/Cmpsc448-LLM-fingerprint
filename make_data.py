#Require a huggingface account for this one, the dataset is gated, go to https://huggingface.co/datasets/lmsys/chatbot_arena_conversations, click agree, then either run `huggingface-cli login` or set HF_TOKEN.

import os
import re
import random
import pandas as pd
from datasets import load_dataset

random.seed(42)
FAMILY = {
    "gpt-4": "GPT",
    "gpt-3.5-turbo": "GPT",
    "claude-v1": "Claude",
    "claude-instant-v1": "Claude",
    "palm-2": "PaLM",
    "vicuna-13b": "Vicuna",
    "vicuna-7b": "Vicuna",
    "koala-13b": "Koala",
}
MIN_WORDS = 10
MAX_WORDS = 800

def first_turn(conv):
    prompt = None
    answer = None
    for m in conv:
        if m["role"] == "user" and prompt is None:
            prompt = m["content"]
        elif m["role"] == "assistant" and answer is None:
            answer = m["content"]
    return prompt, answer

def guess_task(prompt):
    p = prompt.lower()
    code_words = ["code", "python", "java", "c++", "javascript", "function",
                  "program", "sql", "html", "bug", "compile", "script", "```"]
    for w in code_words:
        if w in p:
            return "coding"
    write_words = ["write a poem", "write a story", "write an essay", "write a song",
                   "poem", "story about", "essay", "email to", "blog post", "lyrics",
                   "write a letter"]
    for w in write_words:
        if w in p:
            return "writing"
    math_words = ["calculate", "solve", "equation", "math", "probability",
                  "derivative", "integral", "how many", "sum of", "percent"]
    for w in math_words:
        if w in p:
            return "math"
    return "general"

print("downloading dataset (this takes a minute the first time)...")
ds = load_dataset("lmsys/chatbot_arena_conversations", split="train")
print("got", len(ds), "battles")

rows = []
for r in ds:
    if r["language"] != "English":
        continue
    for side in ["a", "b"]:
        model = r["model_" + side]
        if model not in FAMILY:
            continue
        prompt, answer = first_turn(r["conversation_" + side])
        if prompt is None or answer is None:
            continue
        prompt = prompt.strip()
        answer = answer.strip()
        if len(prompt) == 0 or len(answer) == 0:
            continue
        n = len(answer.split())
        if n < MIN_WORDS or n > MAX_WORDS:
            continue
        rows.append({
            "qid": r["question_id"],
            "model": model,
            "family": FAMILY[model],
            "prompt": prompt,
            "response": answer,
            "task": guess_task(prompt),
        })

df = pd.DataFrame(rows)
print("after filtering:", len(df), "rows")
print(df["model"].value_counts())
print(df["family"].value_counts())

df = df.drop_duplicates(subset=["family", "response"])
print("after dropping dup responses:", len(df))

counts = df["family"].value_counts()
n_each = int(counts.min())
print("balancing every class down to", n_each)

parts = []
for fam in sorted(df["family"].unique()):
    sub = df[df["family"] == fam]
    sub = sub.sample(n=n_each, random_state=42)
    parts.append(sub)
df = pd.concat(parts)
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
print("final size:", len(df))
print(df["task"].value_counts())

qids = sorted(df["qid"].unique())
random.shuffle(qids)
n = len(qids)
train_q = set(qids[:int(0.7 * n)])
val_q = set(qids[int(0.7 * n):int(0.85 * n)])
test_q = set(qids[int(0.85 * n):])

train = df[df["qid"].isin(train_q)]
val = df[df["qid"].isin(val_q)]
test = df[df["qid"].isin(test_q)]

print("train/val/test =", len(train), len(val), len(test))
print(test["family"].value_counts())

if not os.path.exists("data"):
    os.mkdir("data")
train.to_csv("data/train.csv", index=False)
val.to_csv("data/val.csv", index=False)
test.to_csv("data/test.csv", index=False)
print("saved to data/")
