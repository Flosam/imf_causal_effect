# Causal Impact of IMF Programs on Democratic Stability

> Research code sample emphasizing clarity, reproducibility, and modular design.

## Overview

This project constructs a cross-country panel dataset by combining multiple political and macroeconomic data sources (e.g. World Bank, IMF, V-Dem, PWT), performs source-specific cleaning and feature engineering, and estimates heterogeneous treatment effects using a double machine learning framework.

The repository is structured as a reproducible research pipeline:
- modular data loading and cleaning
- transparent feature engineering
- explicit dataset merging logic
- final estimation and visualization

---

## Project Structure
- src/
  - config.py : Set important paremeters for running the analysis
  - cleaning.py : Clean individual and merged datasets
  - data_loader.py : Load each individual dataset
  - features.py : Create additional data features such as lags, ratios and dummies
  - merge.py : Merge all datasets individually into one main dataset
  - models.py : Causal estimation models
  - plots.py : Plotting functions
  - utils.py : Miscellaneous functions
  
- run_analysis.py : Main file. Run the full analysis

---

## Data Sources

The project combines the following datasets:

- **World Bank (WDI)**: macroeconomic indicators (API access)
- **IMF Arrangements Data**: program participation
- **IMF**: macroeconomic variables
- **V-Dem**: political institutions and democracy indices
- **Penn World Table (PWT)**: national accounts data
- **Major Episodes of Political Violence (MEPV)**: political violence
- **Geddes Wright and Frantz Autocratic Regimes (GWF)**: classification of autocratic regimes

Raw data files are expected in `data/raw/`.  
Paths and indicator codes are defined in `src/config.py`.

---

## Methodology

### Empirical Objective

The goal is to estimate heterogeneous treatment effects of IMF program participation on macroeconomic outcomes in a country-year panel.

Let:
- \( $Y_{it}$ \) denote an outcome variable. Here it is a measure of democratic quality
- \( $D_{it}$ \) denote IMF program participation
- \( $X_{it}$ \) denote a vector of economic and political covariates

The estimand of interest is the conditional average treatment effect:

$
\tau(x) = \mathbb{E}[Y_{it}(1) - Y_{it}(0) \mid X_{it} = x]
$

---

### Estimation Strategy

Treatment effects are estimated using a **Doubly Robust (DR) Learner** within a Double Machine Learning (DML) framework.

The estimator combines:
- a treatment model \( $\hat{e}(X) = \mathbb{P}(D=1 \mid X)$ \)
- an outcome model \( $\hat{m}(D, X) = \mathbb{E}[Y \mid D, X]$ \)

Cross-fitting is used to mitigate overfitting bias.  
Consistency is achieved if either nuisance model is correctly specified.

---

### Feature Engineering

Feature construction follows two principles:

1. **Dataset-specific features**  
   Constructed prior to merging (e.g. lagged political indices, cumulative exposure measures)

2. **Cross-dataset features**  
   Constructed after merging (e.g. interaction terms, normalization)

This separation avoids leakage and improves interpretability.

---

### Identification Caveats

- Estimates rely on selection-on-observables
- Results should be interpreted under standard conditional independence and overlap assumptions
- The modular structure facilitates robustness checks and alternative specifications

---

## Data Documentation Template

Each dataset included in the project follows the template below.

---

### Dataset: *[Name]*

**Source**  
- Provider:  
- URL:  
- Access method: API / manual download

**Raw Format**  
- File type: CSV / Excel  
- Unit of observation:  
- Time coverage:  
- Identifiers:

**Cleaning Steps**
- Standardize country identifiers
- Drop non-sovereign entities
- Restrict sample period
- Keep relevant variables only

All cleaning logic is implemented in `src/cleaning.py`.

**Key Variables**

| Variable | Description |
|--------|-------------|
| country_code | ISO-3 country code |
| year | Calendar year |
| gdp_pc | GDP per capita |
| imf_program | IMF program indicator |

**Derived Features**
- Lagged variables
- Growth rates
- Cumulative exposure measures

Feature construction is implemented in `src/features.py`.

**Merge Logic**
- Join keys:  
- Join type:  
- Validation:

Merge logic is implemented in `src/merge.py`.

---

## Running the Analysis

### Environment Setup

Dependencies are managed using `uv`.

```bash
uv sync
