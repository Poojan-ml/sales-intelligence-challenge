"""
Custom metrics module for sales intelligence analysis.

Standard metrics (win rate, avg deal size) are trivial to compute.
This module defines two CUSTOM metrics that go beyond standard reporting:

1. Rep-Segment Fit Score (RSFS)
   - "Is the right rep working the right type of deal?"
   - Uses rep x industry interaction win rates -- the strongest signal in the data
   - 0.32 std devs separation between won and lost (vs 0.19 for deal-profile-only)

2. Segment Momentum Index (SMI)
   - "Is a segment both large AND deteriorating?"
   - Captures the rate of change of win rate, weighted by deal volume.
   - Answers: "Where should the CRO focus attention first?"

Design Note:
We initially explored a Deal Qualification Score (DQS) based on deal-profile
interactions (industry x product_type, lead_source x region). However, the
DQS showed only 0.3pp separation between won and lost deals -- not meaningful
signal. This itself is an important finding: deal PROFILE doesn't predict
outcomes; deal EXECUTION (who works it) does. The Rep-Segment Fit Score
captures this execution dimension.
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Standard metrics
# ---------------------------------------------------------------------------

def win_rate(df: pd.DataFrame, group_col: str = None) -> pd.DataFrame:
    """Compute win rate overall or grouped by a column."""
    if group_col is None:
        total = len(df)
        won = df["is_won"].sum()
        return pd.DataFrame({
            "total_deals": [total], "won": [won],
            "win_rate": [won / total if total > 0 else 0],
        })

    grouped = df.groupby(group_col).agg(
        total_deals=("is_won", "count"),
        won=("is_won", "sum"),
    ).reset_index()
    grouped["win_rate"] = grouped["won"] / grouped["total_deals"]
    return grouped


def win_rate_by_quarter(df: pd.DataFrame, quarter_col: str = "closed_quarter") -> pd.DataFrame:
    """Win rate trend over time (by quarter)."""
    return win_rate(df, quarter_col).sort_values(quarter_col)


def avg_deal_size(df: pd.DataFrame, group_col: str = None) -> pd.DataFrame:
    """Average deal size, optionally grouped."""
    if group_col is None:
        return pd.DataFrame({
            "avg_deal_amount": [df["deal_amount"].mean()],
            "median_deal_amount": [df["deal_amount"].median()],
        })

    return df.groupby(group_col).agg(
        avg_deal_amount=("deal_amount", "mean"),
        median_deal_amount=("deal_amount", "median"),
        deal_count=("deal_amount", "count"),
    ).reset_index()


# ---------------------------------------------------------------------------
# Custom Metric 1: Rep-Segment Fit Score (RSFS)
# ---------------------------------------------------------------------------

def compute_rsfs(df: pd.DataFrame, min_deals: int = 5) -> pd.DataFrame:
    """
    Rep-Segment Fit Score (RSFS)

    Concept:
    For each deal, RSFS answers: "Based on this rep's track record with
    this specific industry, how likely are they to win?"

    This matters because rep performance varies enormously by segment:
    - rep_23 wins 65% of HealthTech deals but only 24% of SaaS deals
    - rep_12 wins 71% of FinTech deals but only 37% of EdTech deals
    - These 30-40pp swings are the largest signal in the entire dataset

    Calculation:
    RSFS = win rate of (sales_rep_id, industry) combination.
    For combinations with fewer than min_deals, fall back to the rep's
    overall win rate (smoothing to avoid small-sample noise).

    Business Use:
    - Deal assignment: match reps to industries where they have proven fit
    - Coaching: if a rep has low RSFS in an industry they're assigned to,
      investigate why and provide targeted coaching
    - Pipeline risk: deals with low RSFS deserve extra managerial attention
    """
    # Compute rep x industry win rates
    rep_ind_wr = df.groupby(["sales_rep_id", "industry"]).agg(
        combo_win_rate=("is_won", "mean"),
        combo_deals=("is_won", "count"),
    ).reset_index()

    # Fallback: rep overall win rate (for sparse combinations)
    rep_overall_wr = df.groupby("sales_rep_id")["is_won"].mean().rename("rep_overall_wr")

    rep_ind_wr = rep_ind_wr.merge(
        rep_overall_wr, on="sales_rep_id", how="left"
    )

    # Use combo rate if enough deals, otherwise fall back to rep overall
    rep_ind_wr["rsfs"] = np.where(
        rep_ind_wr["combo_deals"] >= min_deals,
        rep_ind_wr["combo_win_rate"],
        rep_ind_wr["rep_overall_wr"],
    )

    # Map back to deals
    df = df.copy()
    rsfs_map = rep_ind_wr.set_index(["sales_rep_id", "industry"])["rsfs"]
    df["rsfs"] = df.set_index(["sales_rep_id", "industry"]).index.map(
        lambda idx: rsfs_map.get(idx, df["is_won"].mean())
    )

    return df


def rsfs_by_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """Compare RSFS distributions between won and lost deals."""
    df = compute_rsfs(df)
    return df.groupby("outcome").agg(
        avg_rsfs=("rsfs", "mean"),
        median_rsfs=("rsfs", "median"),
        std_rsfs=("rsfs", "std"),
        count=("rsfs", "count"),
    ).reset_index()


def rsfs_fit_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Generate the rep x industry fit matrix for visualization."""
    matrix = df.pivot_table(
        values="is_won",
        index="sales_rep_id",
        columns="industry",
        aggfunc=["mean", "count"],
    )
    return matrix


def rsfs_trend_by_quarter(df: pd.DataFrame,
                           quarter_col: str = "created_quarter") -> pd.DataFrame:
    """Track average RSFS over time."""
    df = compute_rsfs(df)
    trend = df.groupby(quarter_col).agg(
        avg_rsfs=("rsfs", "mean"),
        std_rsfs=("rsfs", "std"),
        deal_count=("rsfs", "count"),
    ).reset_index().sort_values(quarter_col)
    return trend


# ---------------------------------------------------------------------------
# Custom Metric 2: Segment Momentum Index (SMI)
# ---------------------------------------------------------------------------

def compute_smi(df: pd.DataFrame, segment_col: str = "industry",
                quarter_col: str = "closed_quarter",
                min_deals: int = 30) -> pd.DataFrame:
    """
    Segment Momentum Index (SMI)

    Concept:
    For each segment, compute:
      - The win rate in the most recent quarter (with sufficient data)
      - The win rate in the previous quarter
      - The change (delta)
      - Weight by deal volume (share of total deals)

    SMI = win_rate_delta * volume_share

    A segment with SMI = -0.08 means:
    "This segment's win rate dropped, AND it represents a significant share
    of our pipeline, so it's dragging down overall performance."

    Why this matters:
    - A 20-point win rate drop in a segment with 5% of deals barely matters
    - A 5-point drop in a segment with 40% of deals is an emergency
    - SMI captures this interaction between severity and importance

    Business Use:
    - Rank segments by "danger level" (negative SMI = investigate first)
    - Set up automated alerts when any segment's SMI crosses a threshold
    """
    quarters = sorted(df[quarter_col].unique())

    # Find the last two quarters with sufficient deal volume
    usable_quarters = []
    for q in reversed(quarters):
        q_deals = len(df[df[quarter_col] == q])
        if q_deals >= min_deals:
            usable_quarters.append(q)
        if len(usable_quarters) == 2:
            break

    if len(usable_quarters) < 2:
        return pd.DataFrame()

    recent_q, prev_q = usable_quarters[0], usable_quarters[1]

    recent = df[df[quarter_col] == recent_q]
    prev = df[df[quarter_col] == prev_q]

    recent_wr = recent.groupby(segment_col)["is_won"].mean().rename("recent_win_rate")
    prev_wr = prev.groupby(segment_col)["is_won"].mean().rename("prev_win_rate")

    recent_deals = recent.groupby(segment_col)["is_won"].count().rename("recent_deals")
    prev_deals = prev.groupby(segment_col)["is_won"].count().rename("prev_deals")

    total_recent = len(recent)

    result = pd.concat([recent_wr, prev_wr, recent_deals, prev_deals], axis=1).dropna()
    result["win_rate_delta"] = result["recent_win_rate"] - result["prev_win_rate"]
    result["volume_share"] = result["recent_deals"] / total_recent
    result["smi"] = result["win_rate_delta"] * result["volume_share"]
    result["comparison"] = f"{recent_q} vs {prev_q}"

    return result.sort_values("smi").reset_index()


def compute_smi_all_segments(df: pd.DataFrame,
                              quarter_col: str = "closed_quarter",
                              min_deals: int = 30) -> pd.DataFrame:
    """Compute SMI across multiple segment dimensions and combine."""
    segment_cols = ["industry", "region", "product_type", "lead_source"]
    all_smi = []

    for col in segment_cols:
        smi = compute_smi(df, segment_col=col, quarter_col=quarter_col,
                          min_deals=min_deals)
        if not smi.empty:
            smi = smi.rename(columns={col: "segment_value"})
            smi["segment_dimension"] = col
            all_smi.append(smi)

    if not all_smi:
        return pd.DataFrame()

    return pd.concat(all_smi, ignore_index=True).sort_values("smi")
