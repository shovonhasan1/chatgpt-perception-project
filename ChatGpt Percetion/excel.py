import pandas as pd

df = pd.read_parquet("survey_with_indices.parquet")
df.to_csv("survey_for_powerbi.csv", index=False)