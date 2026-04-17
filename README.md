# Algorithmic Redlining: Auditing and Mitigating Geographic Bias in Ride-Share Pricing

**Course:** Ethics in Artificial Intelligence — Module 3 (Computational Perspective)  
**Academic Year:** 2024/2025  
**Professor:** Roberta Calegari — Università di Bologna

---

## Abstract

Dynamic pricing algorithms in ride-sharing platforms rely on spatial and temporal features to predict fares. Because geography is strongly correlated with demographics (race, income), these models risk producing systematically different pricing errors across neighborhoods — a phenomenon sometimes called *algorithmic redlining*. This project audits a fare-prediction model for geographic bias using publicly available ride-share trip data and US Census demographics, then applies fairness-aware regression techniques to measure the trade-off between model accuracy and equitable pricing.

## Goal

1. Train a fare-prediction model on real ride-share trip records.
2. Link trip origins to demographic profiles via census tract identifiers.
3. Measure whether prediction errors distribute unevenly across demographic groups.
4. Apply a fairness constraint and quantify what accuracy the model gives up for equity.

## Datasets

| Dataset | Source | What it provides |
|---|---|---|
| **Chicago TNP Trips** | [Chicago Data Portal](https://data.cityofchicago.org/Transportation/Transportation-Network-Providers-Trips-2018-2022-/m6dm-c72p) | Trip distance, duration, fare, timestamp, pickup/dropoff census tract |
| **US Census ACS 5-Year** | [Census API](https://data.census.gov/) or [tidycensus](https://walker-data.com/tidycensus/) | Race, median household income, poverty rate by census tract |

> **Note on data size:** The full Chicago TNP dataset is very large (100M+ rows). In practice, sample a manageable subset — e.g. one quarter of one year, or trips from a specific set of community areas. 500K–1M rows is more than enough for meaningful results.

## Pipeline

### Step 0 — Environment Setup

```
Python 3.10+
pandas, numpy, scikit-learn, xgboost
fairlearn
matplotlib, seaborn
```

### Step 1 — Data Acquisition and Preparation

- Download a filtered slice of the Chicago TNP trips dataset (CSV export from the portal, filtered by date range).
- Download ACS 5-Year estimates for Cook County census tracts (total population by race, median household income).
- **Merge:** Join trips to census demographics on the `Pickup Census Tract` field.
- **Label each tract** with a majority-demographic category (e.g., majority-white vs. majority-minority, or income quartile). This is the sensitive attribute — it is *not* a model input, only used for auditing.
- **Feature set for the model:** trip distance (miles), trip duration (seconds), hour of day, day of week, month. Keep it simple — the goal is not to build a great pricing model, but to audit one.
- **Target:** trip fare (dollars).
- Drop rows with missing fare, zero distance, or unmapped tracts.

### Step 2 — Baseline Model

- Train an **XGBoost regressor** (or Gradient Boosting, or even a simpler Ridge regression) on the feature set above.
- Evaluate with standard regression metrics: RMSE, MAE, R².
- This is the "business-as-usual" model — no fairness constraints.

### Step 3 — Bias Audit

- Split predictions by the sensitive group label (e.g., majority-white tracts vs. majority-minority tracts).
- Compare:
  - **Mean prediction error per group** (are fares systematically over- or under-predicted for one group?)
  - **RMSE per group** (is the model less accurate for one group?)
  - **Mean predicted price-per-mile per group** vs. actual price-per-mile (is one group being quoted higher effective rates?)
- Visualize these with grouped bar charts or box plots. This is the core finding of the project.

### Step 4 — Fairness Mitigation

- Use **Fairlearn's `ExponentiatedGradient`** with the **`BoundedGroupLoss`** constraint, wrapping the same XGBoost regressor.
- This retrains the model under a mathematical constraint that forces prediction loss to be roughly equal across groups.
- Re-evaluate: compute the same metrics from Step 3 on the mitigated model.
- **Report the trade-off:** how much RMSE did the overall model lose, and how much did the group gap shrink?

### Step 5 — Report and Analysis

Write the final report (PDF, LaTeX or similar) covering:
- Problem framing and ethical motivation (algorithmic redlining, proxy discrimination).
- Dataset description and preprocessing choices.
- Baseline results + bias audit findings (with figures).
- Mitigated model results + trade-off analysis (with figures).
- Discussion: limitations, what census-tract-level grouping can and cannot tell us, broader implications.

## Key Libraries and References

| Resource | Role |
|---|---|
| [`fairlearn/fairlearn`](https://github.com/fairlearn/fairlearn) | Fairness constraints for regression (`BoundedGroupLoss`, `ExponentiatedGradient`) |
| [`dssg/aequitas`](https://github.com/dssg/aequitas) | Optional — additional bias audit metrics and visualizations |
| [Chicago Data Portal — TNP Trips](https://data.cityofchicago.org/Transportation/Transportation-Network-Providers-Trips-2018-2022-/m6dm-c72p) | Primary dataset |
| Chen et al., "Peeking Beneath the Hood of Uber" (IMC) | Foundational audit methodology for ride-share pricing |
| Lindholm et al., "Discrimination-free Insurance Pricing" | Theoretical framing of fair pricing in regression settings |
| Kasy, "Algorithmic bias and racial inequality" (Oxford) | Why profit-maximizing algorithms can produce inequality by design |

## Repository Structure

```
.
├── README.md
├── data/                  # Raw and processed data (not committed if large — document download steps)
├── notebooks/
│   ├── 01_data_prep.ipynb
│   ├── 02_baseline_model.ipynb
│   ├── 03_bias_audit.ipynb
│   └── 04_mitigation.ipynb
├── report/
│   └── report.pdf
└── requirements.txt
```

## Deliverables

- **Report:** PDF covering background, methodology, results, and ethical discussion.
- **Code repository:** Jupyter notebooks reproducing the full pipeline from data loading to mitigation results.

## Limitations and Honest Notes

- Census tract demographics are **aggregate proxies**, not individual-level attributes. This means we are measuring *geographic* disparity, not individual discrimination. This is a known limitation and should be discussed in the report.
- The fare prediction model is intentionally simple. We are not trying to reverse-engineer Uber's actual algorithm — we are demonstrating the *audit and mitigation methodology* on a realistic task.
- Fairlearn's regression fairness tools are less mature than their classification tools. If `BoundedGroupLoss` behaves unexpectedly, falling back to a simpler approach (e.g., re-weighting training samples by group) is a valid alternative.
- The Chicago dataset reports fare *ranges* in some years rather than exact amounts. Check the data dictionary for your chosen date range and adjust accordingly.
