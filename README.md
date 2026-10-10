# Who Wrote It? Identifying LLMs from Their Responses

CMPSC 448 midterm project.

Team: Junbo Zhao (team leader), Canyi Cheng

## What this is

We take responses from five different LLM families and train a CNN and an LSTM to
guess which family produced a given piece of text.

Data comes from `lmsys/chatbot_arena_conversations` on HuggingFace. Each battle in
that dataset gives us the model name, the user prompt and the model's answer, which
is exactly the `(LLM_name, LLM_input, LLM_output)` triple the project asks for.
We keep five families: GPT (gpt-4, gpt-3.5-turbo), Claude (claude-v1,
claude-instant-v1), PaLM (palm-2), Vicuna (vicuna-7b/13b) and Koala (koala-13b).

## Files

- `make_data.py` - downloads the dataset, filters it, balances the classes and writes
  `data/train.csv`, `data/val.csv`, `data/test.csv`
- `models.py` - the CNN and the LSTM
- `helpers.py` - tokenizer, vocabulary, training loop, confusion matrix plot
- `train.py` - RQ1 and RQ2 experiments
- `rq3.py` - RQ3, cross task generalization (extra credit)
- `rq4.py` - RQ4, what the models actually differ on (extra credit)
- `report_plots.py` - the extra bar charts and training curves used in the report

## How to run

```
pip install torch datasets pandas scikit-learn matplotlib

# you need to accept the dataset terms on huggingface first, then:
export HF_TOKEN=hf_xxxxxxxxxxxx

python make_data.py
python train.py all
python rq3.py
python rq4.py
python report_plots.py
```

Everything gets written into `results/` - the accuracy numbers go into
`results/summary.txt` and the plots are png files in the same folder.

## Notes

- The train/val/test split is done on `question_id`, not on rows. Every battle shows
  the same prompt to two different models, so a plain random split would put the same
  prompt on both sides and the input-only experiment would be meaningless.
- Classes are balanced by downsampling to the size of the smallest family, so the
  random baseline is 20%.
