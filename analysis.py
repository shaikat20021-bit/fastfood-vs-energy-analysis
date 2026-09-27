"""
Fast Food Consumption vs. Daily Energy Levels
Mini Data Science Project — CSE-328: Introduction to Data Science

Reproduces the data cleaning pipeline, exploratory data analysis, and
logistic regression model described in the accompanying report
(report/mini_paper.pdf).

Usage:
    pip install -r requirements.txt
    python analysis.py

Outputs:
    - data/fast_food_survey_cleaned.csv   (cleaned dataset)
    - figures/*.png                       (all 5 EDA figures)
    - Console output of the logistic regression summary
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

RAW_PATH = "data/fast_food_survey_raw.csv"
CLEAN_PATH = "data/fast_food_survey_cleaned.csv"
FIGURES_DIR = "figures"

os.makedirs(FIGURES_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------

def load_raw(path=RAW_PATH):
    df = pd.read_csv(path)
    # The raw Google Forms export has inconsistent leading/trailing
    # whitespace in its column headers -- normalize those first.
    df.columns = [c.strip() for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# 2. Clean
# ---------------------------------------------------------------------------

RENAME_MAP = {
    "1. Age Group": "age_group",
    "2. Gender": "gender",
    "3. Occupation": "occupation",
    "4. How often do you eat fast food?": "fastfood_frequency",
    "5. What type of fast food do you consume most frequently?": "fastfood_type",
    "6. At what time do you usually consume fast food?": "consumption_time",
    "7. Why do you prefer fast food?": "reason",
    "8. How would you rate your daily energy level?": "energy_level",
    "9. Do you feel tired after eating fast food?": "post_meal_tired",
    "10. How many hours of sleep do you get daily?": "sleep_hours",
    "11. Do you engage in regular physical activity?": "physical_activity",
    "12. Do you think fast food affects your energy level?": "perceived_effect",
    "13. After eating fast food, how do you usually feel?": "post_meal_feeling",
}

# Ordinal encodings used for correlation analysis and the logistic model.
FREQ_MAP = {
    "Rarely/Never": 0,
    "Occasionally": 1,
    "1–2 times per week": 2,
    "3–5 times per week": 3,
    "Daily": 4,
}
TIRED_MAP = {
    "Never": 0,
    "Rarely": 1,
    "Sometimes": 2,
    "Often": 3,
    "Always": 4,
}
SLEEP_MAP = {
    "Less than 5 hours": 1,
    "5–6 hours": 2,
    "7–8 hours": 3,
    "More than 8 hours": 4,
}
ACTIVITY_MAP = {
    "Never": 0,
    "Rarely": 1,
    "Yes (few times a week)": 2,
    "Yes (daily)": 3,
}


def clean(df):
    # Duplicate removal happens on the raw export (including Timestamp) so
    # that two respondents who happened to answer identically -- but at
    # different times -- aren't mistaken for the same resubmission. This
    # matches the report's stated outcome of 72 unique, valid rows.
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed:
        print(f"Removed {removed} duplicate row(s).")

    df = df.drop(columns=["Timestamp"])
    df = df.rename(columns=RENAME_MAP)

    # Mode imputation for the optional free-text / optional fields that
    # contain missing values.
    optional_cols = [
        "fastfood_type", "consumption_time", "reason",
        "sleep_hours", "physical_activity",
    ]
    for col in optional_cols:
        if df[col].isna().any():
            mode_val = df[col].mode(dropna=True)[0]
            df[col] = df[col].fillna(mode_val)

    # Encode ordinal features into numeric scales.
    df["Freq_Num"] = df["fastfood_frequency"].map(FREQ_MAP)
    df["Tired_Num"] = df["post_meal_tired"].map(TIRED_MAP)
    df["Sleep_Num"] = df["sleep_hours"].map(SLEEP_MAP)
    df["Activity_Num"] = df["physical_activity"].map(ACTIVITY_MAP)

    # Binary target for the logistic regression bonus model: energy
    # ratings of 1-2 are "Low Energy" (True), 3-5 are not (False).
    df["Low_Energy"] = df["energy_level"].apply(lambda x: 1 if x <= 2 else 0)

    return df


# ---------------------------------------------------------------------------
# 3. EDA figures (mirrors Figures 1-5 in the report)
# ---------------------------------------------------------------------------

def make_figures(df):
    # Figure 1: Occupation distribution (pie chart)
    plt.figure(figsize=(6, 6))
    df["occupation"].value_counts().plot.pie(autopct="%1.1f%%", ylabel="")
    plt.title("Occupation Distribution")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig1_occupation_distribution.png", dpi=150)
    plt.close()

    # Figure 2: Target variable distribution (histogram + KDE)
    plt.figure(figsize=(6, 4))
    sns.histplot(df["energy_level"], bins=5, kde=True, color="skyblue")
    plt.title("Distribution of Daily Energy Level")
    plt.xlabel("Energy Level (1-5)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig2_energy_distribution.png", dpi=150)
    plt.close()

    # Figure 3: Average energy level by fast food frequency (bar chart)
    order = ["Rarely/Never", "Occasionally", "1–2 times per week",
             "3–5 times per week", "Daily"]
    avg_by_freq = (
        df.groupby("fastfood_frequency")["energy_level"]
        .mean()
        .reindex(order)
    )
    plt.figure(figsize=(6, 4))
    sns.barplot(x=avg_by_freq.index, y=avg_by_freq.values, hue=avg_by_freq.index,
                palette="viridis", legend=False)
    plt.title("Average Energy Level by Fast Food Frequency")
    plt.xlabel("Fast Food Frequency")
    plt.ylabel("Average Energy Level (1-5)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig3_avg_energy_by_frequency.png", dpi=150)
    plt.close()

    # Figure 4: Variance analysis (boxplot)
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="fastfood_frequency", y="energy_level", data=df,
                order=order, hue="fastfood_frequency", palette="Set2", legend=False)
    plt.title("Energy Level vs Fast Food Frequency")
    plt.xlabel("Fast Food Frequency")
    plt.ylabel("Energy Level (1-5)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig4_variance_boxplot.png", dpi=150)
    plt.close()

    # Figure 5: Correlation heatmap
    corr_cols = ["energy_level", "Freq_Num", "Tired_Num", "Sleep_Num", "Activity_Num"]
    corr = df[corr_cols].corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-0.4, vmax=1.0, fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig5_correlation_heatmap.png", dpi=150)
    plt.close()

    print(f"Saved 5 figures to {FIGURES_DIR}/")
    print(f"\nAverage daily energy level across all respondents: {df['energy_level'].mean():.2f} / 5")
    print(f"Respondents reporting 'Sometimes' or 'Often' post-meal fatigue: "
          f"{(df['post_meal_tired'].isin(['Sometimes', 'Often']).mean() * 100):.1f}%")


# ---------------------------------------------------------------------------
# 4. Logistic regression (bonus model)
# ---------------------------------------------------------------------------

def run_logistic_regression(df):
    X = df[["Freq_Num", "Sleep_Num", "Activity_Num"]]
    X = sm.add_constant(X)
    y = df["Low_Energy"]

    model = sm.Logit(y, X)
    result = model.fit(disp=0)

    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION: predicting Low_Energy")
    print("(Freq_Num + Sleep_Num + Activity_Num)")
    print("=" * 60)
    print(result.summary())

    return result


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    raw = load_raw()
    df = clean(raw)
    df.to_csv(CLEAN_PATH, index=False)
    print(f"Cleaned dataset saved to {CLEAN_PATH} ({len(df)} rows)")

    make_figures(df)
    run_logistic_regression(df)
