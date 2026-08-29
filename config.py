"""
Central configuration for the DataCo Supply Chain Delivery Analytics application.

Everything that a reviewer might want to change - file locations, the random seed,
and above all the *scenario cost assumptions* - lives here rather than being scattered
through the code.

IMPORTANT (spec section 24): the DataCo dataset contains NO freight-cost field. Every
monetary parameter in SCENARIO_DEFAULTS below is an ASSUMPTION supplied by the analyst,
not an observed value from the data. The application labels them as such wherever they
are displayed.
"""

from pathlib import Path

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = BASE_DIR / "uploads"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
RESULT_DIR = OUTPUT_DIR / "model_results"
EXPORT_DIR = OUTPUT_DIR / "exports"

DATASET_FILENAME = "DataCoSupplyChainDataset.csv"
DICTIONARY_FILENAME = "DescriptionDataCoSupplyChain.csv"

# The CSVs are Git-LFS tracked at the repository root. Searching the repo root first
# avoids keeping a second 95 MB copy inside data/. Uploaded files take precedence so the
# section-50 upload flow can override the bundled dataset.
DATA_SEARCH_PATH = [UPLOAD_DIR, DATA_DIR, BASE_DIR]

# Artifacts written by train_models.py and read by app.py
CLASSIFIER_PATH = MODEL_DIR / "classification_model.pkl"
CLASSIFIER_PIPELINE_PATH = MODEL_DIR / "classification_preprocessing_pipeline.pkl"
REGRESSOR_PATH = MODEL_DIR / "regression_model.pkl"
REGRESSOR_PIPELINE_PATH = MODEL_DIR / "regression_preprocessing_pipeline.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
ANALYTICS_PATH = RESULT_DIR / "analytics.json"
SCORED_SAMPLE_PATH = RESULT_DIR / "scored_sample.csv"

# --------------------------------------------------------------------------------------
# Reproducibility (spec section 43)
# --------------------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20

# "group" -> GroupShuffleSplit on Order Id. This is the default because the delivery
# outcome is recorded per ORDER while rows are per LINE ITEM (2.75 lines per order on
# average). A plain stratified split would place line items of the same order on both
# sides of the split and inflate test scores.
# "time"  -> chronological split, train on the earlier period, test on the later one.
SPLIT_STRATEGY = "group"
TIME_SPLIT_CUTOFF = "2017-01-01"

CV_FOLDS = 4

# --------------------------------------------------------------------------------------
# Modelling
# --------------------------------------------------------------------------------------
TARGET = "Late_delivery_risk"
DECISION_VARIABLE = "Shipping Mode"
ECONOMIC_TARGET = "Order Profit Per Order"

# High-cardinality categoricals are bucketed to the top-N levels plus "Other" so the
# one-hot matrix stays tractable at 180k rows.
HIGH_CARDINALITY_TOP_N = {
    "Order City": 30,
    "Order State": 25,
    "Order Country": 40,
    "Customer City": 25,
    "Customer State": 25,
    "Category Name": 25,
    "Product Name": 25,
}

# Orders whose shipment was cancelled never actually shipped, so they carry no delivery
# outcome to learn from. They are excluded from the delay model but kept for EDA.
EXCLUDE_CANCELLED_FROM_MODEL = True
CANCELLED_DELIVERY_STATUS = "Shipping canceled"

# Rows sampled into the browser-facing scored table (never ship 180k rows to a page).
SCORED_SAMPLE_SIZE = 5000

# --------------------------------------------------------------------------------------
# Scenario cost assumptions (spec sections 24-28)
#
#   *** THESE ARE ASSUMPTIONS, NOT OBSERVED DATA. ***
#
# The DataCo dictionary has no freight-cost field, so total logistics cost cannot be
# measured. The application therefore exposes a transparent expected-cost model whose
# parameters the user can change from the UI.
#
# Note on the "value of delivery speed" term: without a time-related cost, Standard Class
# has BOTH the lowest assumed freight cost AND the lowest observed late rate in this
# dataset, so it dominates every alternative and the decision engine degenerates to a
# single answer regardless of the other assumptions. The speed term represents the
# business value of receiving goods sooner (customer experience, competitive service,
# reduced pipeline inventory) and restores a genuine trade-off.
# --------------------------------------------------------------------------------------
SCENARIO_DEFAULTS = {
    # Assumed freight cost charged per order, by shipping mode ($).
    "freight_cost": {
        "Standard Class": 6.00,
        "Second Class": 10.00,
        "First Class": 18.00,
        "Same Day": 32.00,
    },
    # Assumed economic loss when an order is delivered late: a fixed service-recovery
    # cost plus a share of order value (goodwill / churn exposure).
    "late_penalty_fixed": 15.00,
    "late_penalty_rate": 0.05,
    # Assumed inventory / pipeline holding cost per day, as a fraction of order value.
    "holding_rate_per_day": 0.0005,
    # Assumed business value of each day of faster delivery ($ per order per day).
    "speed_value_per_day": 8.00,
}

SCENARIO_PRESETS = {
    "low": {
        "label": "Low delay cost",
        "description": "Late delivery carries little economic consequence.",
        "late_penalty_fixed": 5.00,
        "late_penalty_rate": 0.01,
        "speed_value_per_day": 3.00,
    },
    "medium": {
        "label": "Medium delay cost",
        "description": "Moderate financial consequence from late delivery.",
        "late_penalty_fixed": 15.00,
        "late_penalty_rate": 0.05,
        "speed_value_per_day": 8.00,
    },
    "high": {
        "label": "High delay cost",
        "description": "Late delivery is economically expensive.",
        "late_penalty_fixed": 40.00,
        "late_penalty_rate": 0.15,
        "speed_value_per_day": 15.00,
    },
}

# Optional decision constraints (spec section 26). None = unconstrained.
CONSTRAINT_DEFAULTS = {
    "max_late_probability": None,   # e.g. 0.50 -> never choose a mode above 50% late risk
    "max_premium_share": None,      # e.g. 0.20 -> at most 20% of orders in premium modes
}
PREMIUM_MODES = ["First Class", "Same Day"]

# Risk banding for the simulator. Boundaries are configurable (spec section 32) and
# default to the tertiles of predicted risk rather than arbitrary round numbers.
RISK_BANDS = {"low": 0.33, "high": 0.66}

# --------------------------------------------------------------------------------------
# Statistical testing (spec section 11)
# --------------------------------------------------------------------------------------
ALPHA = 0.05
MULTIPLE_TESTING_METHOD = "fdr_bh"   # Benjamini-Hochberg

# --------------------------------------------------------------------------------------
# Flask
# --------------------------------------------------------------------------------------
SECRET_KEY = "dataco-supply-chain-analytics"
MAX_CONTENT_LENGTH = 300 * 1024 * 1024   # 300 MB upload ceiling (dataset is ~96 MB)
# 127.0.0.1 serves this machine only. To share the app across a local network use
# 0.0.0.0 - either here, or without editing this file:
#     python app.py --host 0.0.0.0
# Note that the application has no authentication, so anyone who can reach the machine
# can open every page and download every export.
HOST = "127.0.0.1"
PORT = 5000
DEBUG = False
