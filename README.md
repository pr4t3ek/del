# DataCo Supply Chain Delivery Analytics

An interactive delivery-risk and logistics decision-support application built with Python and
Flask, running locally at `http://127.0.0.1:5000`.

---

## Project overview

This is a decision-support system, not a standalone classifier. It predicts late-delivery
risk, evaluates shipping-mode alternatives against expected total supply-chain cost, and
recommends a logistics service policy under uncertainty — connecting
**prediction → decision → economic outcome**.

| Role | Field |
|---|---|
| Prediction target | `Late_delivery_risk` |
| Decision variable | `Shipping Mode` |
| Economic outcome | `Order Profit Per Order` and related financial measures |

## Business problem

> Which operational and external factors significantly predict delivery delays, and what is the
> optimal logistics network design decision under uncertainty to minimize total supply chain cost?

The application answers the practical form of that question: which orders are at risk, which
lever moves them, and whether pulling that lever pays for itself.

## Dataset

`DataCoSupplyChainDataset.csv` — 180,519 order line items, 53 columns, covering 2015-01-01 to
2018-01-31, with `DescriptionDataCoSupplyChain.csv` as the data dictionary.

Three properties of the file shape the whole pipeline:

1. **Encoding is ISO-8859-1, not UTF-8.** Reading it as UTF-8 raises `UnicodeDecodeError` on
   the Spanish place names (`Japón`, `Turquía`, `México`).
2. **The rows are line items; the outcome is per order.** All 65,752 orders carry exactly one
   value of `Late_delivery_risk`, `Shipping Mode` and shipping date across their line items —
   the 180,519 rows are repeated measures averaging 2.75 lines per order.
3. **Both CSVs are tracked with Git LFS.** Without `git lfs pull` they are ~130-byte pointer
   files and pandas fails with a confusing parser error.

### Data dictionary reconciliation

The application compares the CSV against the supplied dictionary and reports two genuine
discrepancies:

- `Order Zipcode` is present in the CSV but undocumented in the dictionary (and 86.2% missing).
- The CSV column is `shipping date (DateOrders)` with a lowercase "s"; the dictionary uses
  `Shipping date (DateOrders)`. A literal lookup raises `KeyError`, so all column access goes
  through a case- and whitespace-insensitive resolver.

## Data leakage methodology

Three columns reconstruct the target almost perfectly and are excluded from every model:

| Column | Why excluded |
|---|---|
| `Delivery Status` | Equals the target exactly on 100% of rows |
| `Days for shipping (real)` | Compared against the promise, reproduces the target on 97.6% |
| `shipping date (DateOrders)` | Differenced against the order date, recovers realised transit |

Including any of them produces the ~99% accuracy commonly reported for this dataset — a figure
that scores the outcome against itself rather than forecasting anything.

A fourth column needs different handling. `Days for shipment (scheduled)` **is** known at
decision time, but in this dataset it is a deterministic lookup on `Shipping Mode`
(Same Day→0, First Class→1, Second Class→2, Standard Class→4), so the two are perfectly
collinear. It is retained in the statistical model to make the multicollinearity diagnostic
concrete, and dropped from the ML feature set.

`Order Status` is also excluded: terminal values (`COMPLETE`, `CLOSED`) are post-fulfilment,
and `CANCELED` / `SUSPECTED_FRAUD` reconstruct `Shipping canceled` exactly.

The **Classification** page shows the full predictor-availability table — every column, whether
it was used, and why.

## Statistical methodology

Tests run at the **order grain** (65,752 orders), not the line-item grain. Testing on line items
would treat a three-line order as three independent observations, inflating every statistic by
roughly the average basket size.

- Categorical factors: chi-square test of independence with **Cramér's V**
- Numeric factors: **Mann-Whitney U** with rank-biserial *r* — non-parametric, because order
  value, profit and discount are heavily skewed
- More than two groups: **Kruskal-Wallis H** with epsilon-squared
- **Benjamini-Hochberg** correction across the test family, α = 0.05

At this sample size almost any non-zero association reaches p < 0.001, so every test reports an
effect size and the summary table is ordered by effect size rather than p-value.

## Machine-learning methodology

Four classifiers, each inside one `sklearn` `Pipeline` with imputation, scaling and one-hot
encoding fitted on training data only: Logistic Regression, Decision Tree, Random Forest and
Gradient Boosting (`HistGradientBoostingClassifier`).

**Split.** 80/20 using `GroupShuffleSplit` grouped on `Order Id`, so no order appears on both
sides. A plain stratified split would scatter line items of the same order across train and
test, and because the outcome is identical across an order's lines, the model would be scored
partly on outcomes it had already seen. `StratifiedGroupKFold` is used for cross-validation. A
chronological split is available via `config.SPLIT_STRATEGY = "time"`.

**Robustness.** `handle_unknown='ignore'` on the encoder so unseen categories never break
prediction; high-cardinality categoricals bucketed to their top-N levels plus `Other`
(`Order City` alone has 3,597 levels); median and most-frequent imputation; `random_state = 42`
throughout.

**Class balance.** The target is near-balanced at 57.3% late in the modelling population, so no
resampling or class weighting is applied.

## Model selection

Preferred model chosen on **held-out ROC-AUC**. AUC rather than accuracy because the decision
layer consumes ranked probabilities, and at this base rate accuracy rewards whichever model
matches the base rate rather than the one that best separates late from on-time orders.

## Decision optimization methodology

For each order and each candidate mode:

```
Expected total cost = freight[mode]                                    (assumption)
                    + P(late | order, mode) × penalty(order)           (model × assumption)
                    + holding_rate × order_value × transit[mode]       (assumption)
                    + speed_value × transit[mode]                      (assumption)

where penalty(order) = fixed_cost + rate × order_value
```

`P(late)` comes from the trained model, scored counterfactually with each mode substituted in.
Mean transit per mode is measured from the data. Every monetary coefficient is an assumption
set in the interface.

**Why there is a time term.** Expected cost written as `freight + P(late) × penalty` alone is
degenerate on this dataset: Standard Class has both the lowest assumed freight cost *and* the
lowest observed late rate, so it absorbs 99.8% of orders under any assumptions and the
optimisation answers itself before it starts. The holding and value-of-speed terms price the
fact that faster modes deliver sooner even when they miss their promise, which restores a
genuine order-dependent trade-off. Setting both to zero in the UI reproduces the degenerate
case, which the decision page shows explicitly.

On top of the per-order engine: a break-even ladder (Standard → Second → First → Same Day),
five service-network policies compared on cost, reliability, complexity and robustness, and
sensitivity sweeps that report each recommendation as a stability region rather than a point
estimate.

## Scenario assumptions

**The DataCo dataset contains no freight-cost, carrier or logistics-cost field**, so total
logistics cost cannot be measured from it. Every monetary parameter below is an analyst
assumption, adjustable on the Decision Optimization page, and labelled as such wherever it
appears.

| Parameter | Default | Meaning |
|---|---|---|
| Freight cost — Standard Class | $6 | Assumed cost per order |
| Freight cost — Second Class | $10 | Assumed cost per order |
| Freight cost — First Class | $18 | Assumed cost per order |
| Freight cost — Same Day | $32 | Assumed cost per order |
| Late penalty (fixed) | $15 | Assumed service-recovery cost |
| Late penalty (rate) | 5% | Assumed share of order value at risk |
| Holding rate per day | 0.05% | Assumed pipeline capital cost |
| Value of one day faster | $8 | Assumed business value of speed |

Three presets — low, medium and high delay cost — are provided for scenario comparison.

## Evaluation metrics

Accuracy, precision, recall, F1, ROC-AUC, PR-AUC, specificity, sensitivity, Brier score,
Youden's J, confusion matrix, ROC and precision-recall curves, calibration curve, and an
interactive threshold simulator. For the logistic model: log-likelihood, AIC, BIC, deviance,
McFadden pseudo-R², coefficients with odds ratios, p-values and 95% confidence intervals, and
VIF.

## Limitations

- **Correlation is not causation.** Every association reported is descriptive.
- **Shipping-mode assignment is observational.** Counterfactual scores are model-implied
  scenarios, not predicted outcomes of an intervention.
- **No freight-cost data exists**, so all cost figures rest on stated assumptions.
- **Late delivery has no measurable profit impact in this data** (correlation −0.0056), so the
  penalty for lateness cannot be calibrated from the data and must be assumed.
- **The delay process here is close to mode-determined.** With `Shipping Mode` removed, all 36
  remaining operational and external factors together reach 0.535 AUC against a chance level of
  0.500. Conclusions about which factors drive delay describe *this dataset's structure* and
  should not be read as a general finding about real-world logistics. The methodology — leakage
  screening, order-grain testing, effect sizes alongside p-values, and pricing the
  reliability/speed trade-off explicitly — does transfer.
- **The final year is partial**: 2018 contains January only.
- **Privacy.** Although email and password are masked placeholders, the source file carries
  customer names, street addresses and per-row geocodes. All such fields are excluded from every
  model here.
- **Physical network design is out of scope.** The dataset has no warehouse, distribution-centre,
  facility or carrier fields, and origin geography has only two values, so the supported decision
  is logistics *service*-network policy: which shipping mode is used for which orders.

---

## How to install

Requires Python 3.10 or later.

```bash
python -m venv venv
```

macOS / Linux:

```bash
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Fetch the data (required)

Both CSVs are stored with Git LFS. Without this step they are ~130-byte pointer files and the
application will tell you so rather than failing obscurely:

```bash
# macOS: brew install git-lfs   |   Debian/Ubuntu: apt-get install -y git-lfs
git lfs install
git lfs pull
```

The loader searches `uploads/`, then `data/`, then the repository root, so the LFS-tracked files
work where they are — no second 96 MB copy is needed.

## Train

```bash
python train_models.py
```

Runs the full pipeline — load, validate against the dictionary, clean, engineer features, screen
for leakage, split, train, evaluate, and precompute every statistic, chart payload and
decision-analysis result — then writes models to `models/` and analytics to `outputs/`. Takes
roughly 3–4 minutes.

## Run

```bash
python app.py
```

Open:

```
http://127.0.0.1:5000
```

The application loads the saved artifacts at startup and never retrains, so pages are instant.

## Tests

```bash
python -m pytest tests/ -q
```

Covers the leakage guard, the cost model and its degeneracy property, order aggregation
invariants, and the group split.

---

## Project structure

```
app.py                  Flask routes, JSON endpoints, error handling
train_models.py         One-time training and precomputation pipeline
config.py               Paths, seed, scenario cost assumptions
requirements.txt
README.md

src/
  data_dictionary.py    Loading, LFS check, column resolver, dictionary reconciliation
  data_preprocessing.py Cleaning, order aggregation, feature engineering, health profiling
  leakage.py            Predictor availability screen and guard
  eda.py                Precomputed EDA and correlation payloads
  statistics_tests.py   Chi-square, Cramér's V, Mann-Whitney, Kruskal-Wallis, BH correction
  classification.py     Model zoo, splits, metrics, curves, threshold sweep, calibration
  regression.py         Profit model and the profit-vs-lateness null result
  diagnostics.py        statsmodels logistic, odds ratios, VIF
  feature_importance.py Permutation, tree and coefficient importance; business interpretation
  decision_analysis.py  Cost model, decision engine, break-even, policies, sensitivity
  reporting.py          CSV exports and executive summary

templates/              16 pages (base, index, data, eda, statistics, classification,
                        diagnostics, risk_drivers, shipping_mode, decision, simulator,
                        insights, presentation_mode, pipeline, governance, upload, downloads,
                        error)
static/css/style.css    Dashboard styling
static/js/              dashboard.js, decision.js, simulator.js, plotly.min.js (vendored)
tests/                  pytest suite
models/  outputs/       Generated artifacts
```

`plotly.min.js` is vendored locally rather than loaded from a CDN, so the application runs with
no internet connection.

## Navigation

**Home** · **Data** · **EDA** · **Statistical Tests** · **Classification** · **Model
Diagnostics** · **Risk Drivers** · **Shipping Mode** · **Decision Optimization** · **Simulator**
· **Business Insights** · **Presentation Mode**, plus Pipeline, Governance, Downloads and Upload.

## Classroom demonstration (5–10 minutes)

1. **Home** — the business problem and the prediction → decision → outcome chain.
2. **Data** — 180,519 line items against 65,752 orders, and why the grain matters.
3. **EDA** — late rate by shipping mode, then by market and region. The flat bars are the finding.
4. **Statistical Tests** — 1 of 18 factors is material; effect size, not p-value, is the test.
5. **Classification** — the leakage table, then the model comparison.
6. **Model Diagnostics** — confusion matrix, ROC, PR, calibration; move the threshold slider.
7. **Risk Drivers** — one driver, then nothing.
8. **Shipping Mode** — Second Class and Standard Class are the same operation with different
   promises.
9. **Simulator** — set an order, read the risk meter and the recommended mode.
10. **Simulator** — change the mode and one characteristic; watch the model-implied change.
11. **Simulator** — tick "override the promise" to model the promise-redesign lever.
12. **Decision Optimization** — the expected-cost table and the policy comparison.
13. **Decision Optimization** — set both time sliders to zero and watch the recommendation
    collapse to one mode; that is why the cost model prices time.
14. **Decision Optimization** — raise the value of delivery speed and watch the optimal policy
    flip. This is the uncertainty in the business question made visible.
15. **Business Insights** — the executive summary and the final recommendations.
