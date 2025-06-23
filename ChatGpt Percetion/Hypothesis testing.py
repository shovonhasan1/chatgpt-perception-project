

# 1. Imports
import pandas as pd
from pathlib import Path
from scipy import stats
import statsmodels.formula.api as smf

#  2. Load dataset with composite indices
DATA_PATH = Path(
    r"C:\Users\user\PycharmProjects\ChatGpt Percetion\survey_with_indices.parquet"
)
df = pd.read_parquet(DATA_PATH)
print("✅ Data loaded:", df.shape)

# 3. Rebuild mimetic_source_bin from Q17 codes
# Q17: 1 = mainstream media, 2 = social media,
#      3 = class/work,      4 = friends/family
peer_codes = [2, 4]                       # mimetic sources
df["mimetic_source_bin"] = df["q17"].isin(peer_codes).astype("int8")

if df["mimetic_source_bin"].sum() == 0:
    raise ValueError("No mimetic rows detected – check Q17 codes!")
print(f"🛠  Mimetic rows detected: {df['mimetic_source_bin'].sum():,}")

# 4. economic_status_num from q39 (if missing)
if "economic_status_num" not in df.columns:
    ECON_MAP = {
        "significantly below-average": 1,
        "below-average": 2,
        "average": 3,
        "above-average": 4,
        "significantly above-average": 5,
    }

    def econ_num(v):
        try:
            return int(float(v))
        except (ValueError, TypeError):
            return ECON_MAP.get(str(v).lower().strip(), pd.NA)

    df["economic_status_num"] = df["q39"].apply(econ_num).astype("Int64")
    print("🛠  Created economic_status_num from q39")

# 5. age & gender (simple 15–60 filter for age)
if "age" not in df.columns:
    df["age"] = (
        pd.to_numeric(df["q3"], errors="coerce")
        .where(lambda x: x.between(15, 60))
        .astype("Int64")
    )
    print("🛠  Created age from q3 (15–60)")

if "gender" not in df.columns:
    df["gender"] = df["q2"].astype(str).str.lower().str.strip()
df["gender"] = df["gender"].fillna("unknown")

#  6. Verify required columns
required = [
    "mimetic_source_bin",
    "employability_belief",
    "ai_skill_index",
    "economic_status_num",
    "age",
    "gender",
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise KeyError(f"Missing columns: {missing}")

# 7. H1 – Mimetic vs Non-mimetic t-test
mimetic = df.loc[df.mimetic_source_bin == 1, "employability_belief"].dropna()
non_mimetic = df.loc[df.mimetic_source_bin == 0, "employability_belief"].dropna()

print("\n📌 H1 – Employability Belief by Mimetic Source")
print(f"  Mimetic N = {len(mimetic):,}  mean = {mimetic.mean():.2f}")
print(f"  Non-mimetic N = {len(non_mimetic):,}  mean = {non_mimetic.mean():.2f}")

lev_p = stats.levene(mimetic, non_mimetic).pvalue
t_stat, p_val = stats.ttest_ind(
    mimetic, non_mimetic, equal_var=lev_p > 0.05
)
u_stat, u_p = stats.mannwhitneyu(mimetic, non_mimetic)

print(f"  Levene p = {lev_p:.4f}")
print(f"  t-stat = {t_stat:.3f}  p = {p_val:.4f}")
print(f"  Mann-Whitney U p = {u_p:.4f}")

# 8. H2 – Mimetic Gap > 0 (one-sample t-test)
df["mimetic_gap"] = df["employability_belief"] - df["ai_skill_index"]
gap = df["mimetic_gap"].dropna()
t_gap, p_gap = stats.ttest_1samp(gap, 0, alternative="greater")

print("\n📌 H2 – Mimetic Gap (Belief – Skill) > 0")
print(f"  N = {len(gap):,}  mean gap = {gap.mean():.3f}")
print(f"  t = {t_gap:.3f}  one-tailed p = {p_gap:.4f}")

# 9. H3 – Economic status ↔ Belief (Spearman)
rho, p_rho = stats.spearmanr(
    df["economic_status_num"].dropna(),
    df["employability_belief"].dropna(),
)

print("\n📌 H3 – Spearman: Economic Status vs Employability Belief")
print(f"  ρ = {rho:.3f}  p = {p_rho:.4f}")

# 10. OLS Robustness Regression
print("\n📌 Robustness OLS Regression (coefficients)")
formula = (
    "employability_belief ~ mimetic_source_bin + ai_skill_index + "
    "economic_status_num + age + C(gender)"
)
model_df = df[required].dropna()
model = smf.ols(formula, data=model_df).fit()
print(model.summary().tables[1])

df = pd.read_parquet("survey_with_indices.parquet")
df.to_csv("survey_for_powerbi.csv", index=False)