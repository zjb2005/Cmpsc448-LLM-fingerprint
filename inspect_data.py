import pandas as pd
from pathlib import Path

files = [
    "data/train.csv",
    "data/val.csv",
    "data/test.csv"
]

label_columns = ["llm_name", "LLM_name", "model", "label"]
output_columns = ["llm_output", "LLM_output", "response", "output"]

for file in files:
    path = Path(file)

    print("\n" + "=" * 50)
    print("File:", file)

    if not path.exists():
        print("File not found.")
        continue

    df = pd.read_csv(path)

    print("Number of samples:", len(df))
    print("Columns:", list(df.columns))

    label_col = None
    for col in label_columns:
        if col in df.columns:
            label_col = col
            break

    if label_col:
        print("\nSamples for each LLM:")
        print(df[label_col].value_counts())
    else:
        print("\nLLM label column was not found.")

    print("\nMissing values:")
    print(df.isnull().sum())

    output_col = None
    for col in output_columns:
        if col in df.columns:
            output_col = col
            break

    if output_col:
        response_lengths = (
            df[output_col]
            .fillna("")
            .astype(str)
            .str.split()
            .str.len()
        )

        print("\nAverage response length:")
        print(round(response_lengths.mean(), 2))

print("\nDataset inspection completed.")
