# Fast Food Consumption vs. Daily Energy Levels

A mini data science project (CSE-328: Introduction to Data Science) investigating whether fast-food consumption frequency correlates with self-reported daily energy levels and short-term post-meal fatigue, using primary survey data.

**Authors:** MD AL Sayeed Shaikat, Suma Akter

## Overview

- **Data collection:** A Google Forms survey distributed via convenience sampling, collecting 72 valid responses across 14 demographic, behavioral, and physiological features.
- **Research questions:**
  1. Does higher fast-food consumption frequency negatively correlate with baseline daily energy level?
  2. Does frequent consumption increase the likelihood of short-term post-meal fatigue ("food coma")?
  3. How do sleep duration and physical activity buffer or worsen these effects?

## Repository Structure

```
.
├── analysis.py                          # full cleaning + EDA + logistic regression pipeline
├── requirements.txt
├── data/
│   └── fast_food_survey_raw.csv         # raw Google Forms export (72 responses)
├── figures/                              # generated on running analysis.py
└── report/
    └── mini_paper.pdf                    # full written report (ACM format)
```

## How to Run

```bash
pip install -r requirements.txt
python analysis.py
```

This will:
1. Clean the raw survey export (mode-imputation for missing optional fields, duplicate removal, ordinal encoding)
2. Save the cleaned dataset to `data/fast_food_survey_cleaned.csv`
3. Generate all 5 EDA figures into `figures/`
4. Fit and print a logistic regression model predicting "Low Energy" (self-rated 1-2) from fast-food frequency, sleep, and physical activity

## Key Findings

- **Average daily energy level:** 3.24 / 5 across all 72 respondents
- **The "3-5 times a week" dip:** this frequency band reported the lowest average energy (2.94), while "Occasionally" consumers reported the healthiest baseline (3.27)
- **Post-meal fatigue:** over 60% of respondents reported feeling tired, sleepy, or lazy "Sometimes" or "Often" after eating fast food
- **Weak linear correlation, real fatigue effect:** the direct linear correlation between fast-food frequency and baseline energy was weak (heatmap), but post-meal tiredness showed a moderate negative correlation (-0.32) with energy — supporting a short-term crash effect rather than a lasting baseline shift
- **Buffering factors:** sleep and physical activity both showed small positive correlations with energy level, consistent with acting as physiological buffers against poor diet

See `report/mini_paper.pdf` for the full write-up, methodology, and discussion of limitations.

## Limitations

- Self-reported survey data is subject to subjective bias
- Sample size (72) is sufficient for exploratory analysis but too small for a generalizable predictive model
- The survey did not track caloric intake, hydration, or caffeine consumption — all plausible confounders

## License

This project is shared for academic/portfolio purposes. Feel free to reference the methodology, but please cite appropriately if reusing the dataset or analysis.
