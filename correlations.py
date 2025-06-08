# Import required packages
import pandas as pd
from scipy.stats import chi2_contingency

df = pd.read_csv("merged_output.csv")

# Define variable pairs to test
pairs = [
    ("story", "agency"),
    ("story", "event_sequencing"),
    ("story", "world_making"),
    ("deltas", "agency"),
    ("deltas", "event_sequencing"),
    ("deltas", "world_making"),
    ("deltas", "story"),
]

# Convert all values > 1 in deltas to 1 ->
# since a comment can receive multiple deltas
df["deltas"] = df["deltas"].apply(lambda x: 1 if pd.notna(x) and x > 0 else 0)

binary_columns = [
    "deltas", "story", "agency", "event_sequencing", "world_making"
]
df[binary_columns] = df[binary_columns].astype("Int64")


# Function to run chi-square test
def run_chi_square(df_subset, var1, var2):
    contingency = pd.crosstab(df_subset[var1], df_subset[var2])
    if contingency.shape == (2, 2):  # Ensure binary variables
        chi2, p, dof, expected = chi2_contingency(contingency)
        return {"chi2": chi2, "p_value": p, "dof": dof}
    return {"chi2": None, "p_value": None, "dof": None}


# Run tests on whole dataset
print("Chi-Square Tests on Entire Dataset:")
for var1, var2 in pairs:
    result = run_chi_square(df, var1, var2)
    print(f"{var1} vs {var2}: χ² = {result['chi2']:.3f}, \
          p = {result['p_value']:.3g}, dof = {result['dof']}")
