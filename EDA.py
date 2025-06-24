

# 1. Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", palette="pastel", font_scale=1.1)

# 2. Load Cleaned Dataset
df = pd.read_parquet(
    r"C:\Users\user\PycharmProjects\ChatGpt Percetion\clean_chatgpt_survey.parquet"
)
print("✅ Data loaded. Shape:", df.shape)

# 3. Robust AGE Cleaning
CURRENT_YEAR = 2025

def fix_age(val):

    if pd.isna(val):
        return np.nan

    if 1900 <= val <= CURRENT_YEAR:
        age = CURRENT_YEAR - val
    else:
        age = val
    if 15 <= age <= 60:
        return age
    return np.nan

df["age"] = df["q3"].apply(fix_age)
print("\n🧐 Cleaned age stats:")
print(df["age"].describe())

#  4. Summary Statistics
print("\n📊 Summary Statistics:")
print(df.describe(include='all').T)

#  5. Missing Values Overview
missing = df.isnull().mean().sort_values(ascending=False)
print("\n🔍 Missing Value Proportions (top 20):")
print(missing[missing > 0].head(20))

# 6. Distribution of Ages
plt.figure(figsize=(8, 5))
sns.histplot(df["age"].dropna(), bins=30, kde=True, color="skyblue")
plt.title("Distribution of Student Ages (15–60)")
plt.xlabel("Age")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

#  7. Confidence in Employment
if "job_confidence_num" in df.columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(x="job_confidence_num", data=df, palette="viridis")
    plt.title("Confidence in Getting a Job After Studies")
    plt.xlabel("Confidence Level (1 = Low, 5 = High)")
    plt.ylabel("Number of Students")
    plt.tight_layout()
    plt.show()

# 8. Source of Awareness (Mimetic Influence)
if "source" in df.columns:
    plt.figure(figsize=(8, 5))
    df["source"].value_counts().plot(kind="barh", color="mediumslateblue")
    plt.title("Where Students First Heard of ChatGPT")
    plt.xlabel("Number of Students")
    plt.ylabel("Source")
    plt.tight_layout()
    plt.show()

# 9. Correlation Heatmap
num_df = df.select_dtypes(include="number")

plt.figure(figsize=(12, 10))
sns.heatmap(num_df.corr(), cmap="coolwarm", annot=False, fmt=".2f")
plt.title("Correlation Matrix (Numeric Features)")
plt.tight_layout()
plt.show()

# 10. Pairplot for Mimetic Variables
mimetic_vars = [
    "job_confidence_num",
    "employability_num",
    "ai_literacy_num",
    "economic_status_num",
]
mimetic_vars = [v for v in mimetic_vars if v in df.columns]

if len(mimetic_vars) >= 3:
    sns.pairplot(df[mimetic_vars], corner=True)
    plt.suptitle("Mimetic Cluster Variable Relationships", y=1.02)
    plt.tight_layout()
    plt.show()

# 11. Country-wise Usage (Optional)
if "q1" in df.columns and "usage_frequency_num" in df.columns:
    usage_by_country = (
        df.groupby("q1")["usage_frequency_num"]
        .mean()
        .sort_values(ascending=False)
        .head(15)
    )

    plt.figure(figsize=(10, 6))
    usage_by_country.plot(kind="barh", color="coral")
    plt.title("Top 15 Countries by Avg ChatGPT Usage Frequency")
    plt.xlabel("Average Usage Level")
    plt.ylabel("Country")
    plt.tight_layout()
    plt.gca().invert_yaxis()
    plt.show()

# 12. Save Summary CSV
summary_path = "summary_stats.csv"
df.describe(include='all').T.to_csv(summary_path)
print(f"📁 Summary saved to {summary_path}")
