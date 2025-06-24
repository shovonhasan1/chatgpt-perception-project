

import pandas as pd
from pathlib import Path

# 1. Load cleaned dataset
DATA_PATH = Path(
    r"C:\Users\user\PycharmProjects\ChatGpt Percetion\clean_chatgpt_survey.parquet"
)
df = pd.read_parquet(DATA_PATH)
print("✅ Loaded:", DATA_PATH.name, df.shape)

# 2. Helper to ensure a numeric version of a Likert column
LIKERT_5 = {
    "strongly disagree": 1,
    "disagree": 2,
    "neutral": 3,
    "agree": 4,
    "strongly agree": 5,
}


def ensure_numeric(col_raw: str) -> str:

    col_num = f"{col_raw}_num"

    # Case 1: already have a *_num column
    if col_num in df.columns:
        return col_num

    # Case 2: raw column exists
    if col_raw in df.columns:
        if pd.api.types.is_numeric_dtype(df[col_raw]):
            return col_raw  # already numeric
        # Raw is text → create numeric mapping
        df[col_num] = (
            df[col_raw]
            .astype(str)
            .str.lower()
            .str.strip()
            .replace(LIKERT_5)
            .astype("Int64")
        )
        print(f"🛠  Created numeric column: {col_num}")
        return col_num

    # Column truly missing
    raise KeyError(f"Neither {col_raw} nor {col_num} found in DataFrame")


# 3. Mimetic Source Indicator (from numeric Q17 codes)
# Q17 codes: 1 = mainstream media, 2 = social media,
#            3 = class/work,        4 = friends/family
peer_codes = [2, 4]  # mimetic sources
df["mimetic_source_bin"] = df["q17"].isin(peer_codes).astype("int8")

# 4. Employability Belief Index
belief_items_raw = ["q27j", "q30e"]  # Likert items
belief_items_num = [ensure_numeric(c) for c in belief_items_raw]

# Job-confidence variable already numeric from earlier cleaning
belief_items_num.append("job_confidence_num")

belief_cols = [c for c in belief_items_num if c in df.columns]
df["employability_belief"] = df[belief_cols].mean(axis=1, skipna=True)

# 5. AI-Skill Index
skill_items_raw = ["q29i", "q29h", "q29c"]  # AI literacy, programming, problem-solving
skill_cols = [ensure_numeric(c) for c in skill_items_raw]

df["ai_skill_index"] = df[skill_cols].mean(axis=1, skipna=True)

#  6. Inspect new columns
print(
    "\n🔢 Composite indices preview:\n",
    df[["mimetic_source_bin", "employability_belief", "ai_skill_index"]].head(),
)

# 7. Save augmented dataset
OUT_PATH = DATA_PATH.with_name("survey_with_indices.parquet")
df.to_parquet(OUT_PATH, index=False)
print(f"\n✔  Saved augmented dataset → {OUT_PATH}")
