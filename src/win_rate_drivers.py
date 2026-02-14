"""
Win Rate Driver Analysis Engine

This module identifies which factors are hurting or improving win rate,
ranked by business impact.

Two-model design:
  Model 1 (Baseline): Deal characteristics only (industry, region, product,
  lead source, deal amount, cycle days). This asks: "Do deal PROFILES predict
  outcomes?"

  Model 2 (With RSFS): Adds the Rep-Segment Fit Score. This asks: "Does deal
  EXECUTION (who works it) predict outcomes, beyond deal profile?"

  By comparing the two, we can measure exactly how much signal comes from
  the execution dimension vs. the deal profile dimension.

Approach:
We use logistic regression because:
1. Coefficients map directly to "this factor changes win probability by X%"
2. The output is interpretable in business terms (e.g., "Outbound leads are
   3% less likely to close") rather than abstract feature importances
3. For driver identification, interpretability matters more than maximizing
   predictive accuracy
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from scipy import stats


def prepare_features(df: pd.DataFrame, include_rsfs: bool = False) -> tuple:
    """
    Prepare features for logistic regression.

    Feature engineering decisions:
    - One-hot encode categoricals (industry, region, product_type, lead_source)
    - Log-transform deal_amount (reduces skew, makes relationship more linear)
    - Standardize sales_cycle_days (puts on same scale as dummies)
    - Drop one category per feature (reference encoding) for interpretability
    - Optionally include RSFS (Rep-Segment Fit Score) to capture execution signal
    """
    feature_cols_cat = ["industry", "region", "product_type", "lead_source"]

    # One-hot encode (drop_first for interpretability -- coefficients are
    # relative to the dropped reference category)
    encoder = OneHotEncoder(drop="first", sparse_output=False)
    cat_encoded = encoder.fit_transform(df[feature_cols_cat])
    cat_names = encoder.get_feature_names_out(feature_cols_cat)

    # Numeric features
    log_amount = np.log1p(df["deal_amount"].values).reshape(-1, 1)
    cycle_days = df["sales_cycle_days"].values.reshape(-1, 1)

    # Combine base features
    X = np.hstack([cat_encoded, log_amount, cycle_days])
    feature_names = list(cat_names) + ["log_deal_amount", "sales_cycle_days"]

    # Optionally add RSFS
    if include_rsfs and "rsfs" in df.columns:
        rsfs_vals = df["rsfs"].values.reshape(-1, 1)
        X = np.hstack([X, rsfs_vals])
        feature_names.append("rsfs")

    y = df["is_won"].values

    return X, y, feature_names, encoder


def fit_driver_model(df: pd.DataFrame, include_rsfs: bool = False) -> dict:
    """
    Fit logistic regression and extract driver rankings.

    Args:
        df: Sales data with is_won target. If include_rsfs=True, must
            have an 'rsfs' column (call compute_rsfs first).
        include_rsfs: If True, include RSFS as an additional feature.

    Returns a dict with:
    - model: fitted LogisticRegression
    - drivers: DataFrame of factors ranked by impact
    - cv_scores: cross-validation accuracy (sanity check, not the point)
    - reference_categories: what each coefficient is relative to
    """
    X, y, feature_names, encoder = prepare_features(df, include_rsfs=include_rsfs)

    # Fit model -- C=1.0 (moderate regularization to avoid overfitting)
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    model.fit(X, y)

    # Cross-validation (sanity check only)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

    # Extract coefficients
    coefficients = model.coef_[0]

    # Convert to marginal effects (approximate: at the mean prediction point)
    # For logistic regression, marginal effect ≈ β * p * (1-p)
    # where p is the mean predicted probability
    mean_pred = model.predict_proba(X)[:, 1].mean()
    marginal_factor = mean_pred * (1 - mean_pred)
    marginal_effects = coefficients * marginal_factor

    drivers = pd.DataFrame({
        "factor": feature_names,
        "coefficient": coefficients,
        "marginal_effect": marginal_effects,
        "abs_impact": np.abs(marginal_effects),
        "direction": ["Helps Win Rate" if c > 0 else "Hurts Win Rate"
                       for c in coefficients],
    }).sort_values("abs_impact", ascending=False)

    # Add statistical significance via z-scores
    # Standard errors via inverse of Hessian approximation
    se = _compute_standard_errors(model, X)
    drivers["z_score"] = coefficients / se
    drivers["p_value"] = 2 * (1 - stats.norm.cdf(np.abs(drivers["z_score"])))
    drivers["significant"] = drivers["p_value"] < 0.05

    # Reference categories for interpretability
    ref_cats = {}
    for col_idx, col in enumerate(["industry", "region", "product_type", "lead_source"]):
        ref_cats[col] = encoder.categories_[col_idx][0]

    return {
        "model": model,
        "drivers": drivers.reset_index(drop=True),
        "cv_accuracy": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "reference_categories": ref_cats,
        "feature_names": feature_names,
        "mean_predicted_prob": mean_pred,
    }


def _compute_standard_errors(model, X):
    """Compute standard errors for logistic regression coefficients."""
    # Predicted probabilities
    probs = model.predict_proba(X)[:, 1]
    # Weight matrix (diagonal of p*(1-p))
    W = probs * (1 - probs)
    # Weighted X'X
    XWX = (X.T * W) @ X
    try:
        cov_matrix = np.linalg.inv(XWX)
        se = np.sqrt(np.diag(cov_matrix))
    except np.linalg.LinAlgError:
        se = np.ones(X.shape[1])  # fallback
    return se


def format_driver_report(result: dict) -> str:
    """
    Generate a plain-English driver report for a CRO.

    This is what would appear in the product's UI or automated email.
    """
    drivers = result["drivers"]
    ref = result["reference_categories"]

    lines = []
    lines.append("=" * 70)
    lines.append("WIN RATE DRIVER ANALYSIS")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Model accuracy (5-fold CV): {result['cv_accuracy']:.1%} "
                 f"(±{result['cv_std']:.1%})")
    lines.append(f"Mean predicted win probability: {result['mean_predicted_prob']:.1%}")
    lines.append("")
    lines.append("Reference categories (all effects are RELATIVE to these):")
    for col, ref_cat in ref.items():
        lines.append(f"  {col}: {ref_cat}")
    lines.append("")
    lines.append("-" * 70)
    lines.append("TOP FACTORS AFFECTING WIN RATE (ranked by business impact):")
    lines.append("-" * 70)
    lines.append("")

    for _, row in drivers.head(10).iterrows():
        effect_pct = row["marginal_effect"] * 100
        sig_marker = "*" if row["significant"] else ""
        direction = "[+]" if row["marginal_effect"] > 0 else "[-]"

        lines.append(
            f"  {direction} {row['factor']:30s}  "
            f"{effect_pct:+.2f}pp  "
            f"(p={row['p_value']:.3f}){sig_marker}"
        )

    lines.append("")
    lines.append("* = statistically significant at 95% confidence")
    lines.append("")

    # Business interpretation
    lines.append("-" * 70)
    lines.append("BUSINESS INTERPRETATION:")
    lines.append("-" * 70)

    # Filter by both statistical AND practical significance (>= 0.5pp effect)
    min_effect = 0.005  # 0.5 percentage points
    hurting = drivers[
        (drivers["marginal_effect"] < -min_effect) & (drivers["significant"])
    ].head(3)
    helping = drivers[
        (drivers["marginal_effect"] > min_effect) & (drivers["significant"])
    ].head(3)

    if len(hurting) > 0:
        lines.append("")
        lines.append("Factors HURTING win rate:")
        for _, row in hurting.iterrows():
            effect_pct = abs(row["marginal_effect"] * 100)
            lines.append(
                f"  - {row['factor']} reduces win probability by ~{effect_pct:.1f}pp"
            )
    else:
        lines.append("")
        lines.append("No statistically significant factors hurting win rate detected.")
        lines.append("(This is expected in uniformly distributed sample data.)")

    if len(helping) > 0:
        lines.append("")
        lines.append("Factors HELPING win rate:")
        for _, row in helping.iterrows():
            effect_pct = row["marginal_effect"] * 100
            lines.append(
                f"  - {row['factor']} increases win probability by ~{effect_pct:.1f}pp"
            )
    else:
        lines.append("")
        lines.append("No statistically significant factors improving win rate detected.")

    return "\n".join(lines)
