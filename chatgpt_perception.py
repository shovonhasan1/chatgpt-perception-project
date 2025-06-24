

# 1. Imports
import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.impute import SimpleImputer

# 2. Configuration

DATA_PATH   = Path(r"C:\Users\user\PycharmProjects\ChatGpt Percetion\final dataset.xlsx")
SHEET_NAME  = 0                # 0 = first sheet

EXPORT_FMT  = "parquet"        # parquet | csv | excel
EXPORT_PATH = DATA_PATH.with_name(f"clean_chatgpt_survey.{EXPORT_FMT}")

CRITICAL    = ["age", "gender", "q12"]   # adjust if needed



# 3. Utility functions
def snake_case(col: str) -> str:
    col = re.sub(r'[^A-Za-z0-9]+', '_', col).strip('_').lower()
    return re.sub(r'__+', '_', col)

def map_likert(series: pd.Series, mapping: dict, suffix="_num") -> pd.Series:
    return (series.str.lower().str.strip()
                  .replace(mapping)
                  .astype("Int64")
                  .rename(series.name + suffix))

def detect_binary(series: pd.Series) -> bool:
    uniques = series.dropna().str.lower().str.strip().unique()
    return set(uniques) <= {"yes", "no", "y", "n"}

# 4. Load data
df_raw = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
df     = df_raw.copy()
df.columns = [snake_case(c) for c in df.columns]

print(f"Loaded shape: {df.shape}")

# 5. Pre-define common mappings
LIKERT_5   = {"strongly disagree":1, "disagree":2, "neutral":3,
              "agree":4, "strongly agree":5}
FREQ_5     = {"never":1, "rarely":2, "sometimes":3, "often":4, "always":5}
CONF_5     = {"not at all confident":1, "slightly confident":2,
              "moderately confident":3, "very confident":4, "extremely confident":5}

ECON_5     = {"significantly below-average":1, "below-average":2, "average":3,
              "above-average":4, "significantly above-average":5}

YES_NO_BIN = {"yes":1, "y":1, "no":0, "n":0}

# 6. Auto-detect and encode text → numeric
for col in df.columns:
    if df[col].dtype != "object":
        continue
    sample = df[col].dropna().astype(str).str.lower().str.strip()
    if sample.empty:
        continue

    # Frequency (never…always) variables
    if sample.isin(FREQ_5.keys()).all():
        df[col + "_num"] = map_likert(df[col], FREQ_5)
        continue

    # Likert agreement variables
    if sample.isin(LIKERT_5.keys()).all():
        df[col + "_num"] = map_likert(df[col], LIKERT_5)
        continue

    # Confidence scale
    if sample.isin(CONF_5.keys()).all():
        df[col + "_num"] = map_likert(df[col], CONF_5)
        continue

    # Economic status
    if sample.isin(ECON_5.keys()).all():
        df[col + "_num"] = map_likert(df[col], ECON_5)
        continue

    # Yes/No binary
    if detect_binary(df[col]):
        df[col + "_bin"] = (df[col].str.lower().str.strip()
                                      .replace(YES_NO_BIN)
                                      .astype("Int64"))
        continue

# 7. Numeric coercion for obvious numbers (age, etc.)
for col in df.columns:
    if df[col].dtype == "object":

        if df[col].str.fullmatch(r"\d+").sum() > 0:
            df[col] = pd.to_numeric(df[col], errors="ignore")

# 8. Drop rows missing critical info
df = df.dropna(subset=[c for c in CRITICAL if c in df.columns])
print(f"After dropping critical-missing rows: {df.shape}")

# 9. Impute missing values (numeric & categorical separately)
numeric_cols = df.select_dtypes(include="number").columns
cat_cols     = df.select_dtypes(exclude="number").columns

imp_num = SimpleImputer(strategy="median")
imp_cat = SimpleImputer(strategy="most_frequent")

df[numeric_cols] = imp_num.fit_transform(df[numeric_cols])
df[cat_cols]     = imp_cat.fit_transform(df[cat_cols])

# 10. Optional: clip impossible ages / outlier guard
if "age" in df.columns:
    df.loc[~df["age"].between(18, 100), "age"] = np.nan
    df["age"] = imp_num.fit_transform(df[["age"]])  # re-impute if needed

# 11. Export cleaned data
if EXPORT_FMT == "parquet":
    df.to_parquet(EXPORT_PATH, index=False)
elif EXPORT_FMT == "csv":
    df.to_csv(EXPORT_PATH, index=False)
elif EXPORT_FMT == "excel":
    df.to_excel(EXPORT_PATH, index=False)
else:
    raise ValueError("EXPORT_FMT must be parquet | csv | excel")

print(f"✔ Clean dataset saved → {EXPORT_PATH.resolve()}")
print("\nQuick preview:")
print(df.head(3).T)
