"""
Data loading and preprocessing module.

In a production system, this would abstract over the data source
(CRM API, database, warehouse). For this analysis, it reads from CSV
and applies standard enrichments needed across all parts.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_sales_data(filepath: str = None) -> pd.DataFrame:
    """Load and enrich the sales dataset with derived time-based features."""
    if filepath is None:
        filepath = Path(__file__).parent.parent / "data" / "Company_sales_data.csv"

    df = pd.read_csv(filepath)

    # Parse dates
    df["created_date"] = pd.to_datetime(df["created_date"])
    df["closed_date"] = pd.to_datetime(df["closed_date"])

    # Time-based enrichments
    df["created_quarter"] = df["created_date"].dt.to_period("Q").astype(str)
    df["closed_quarter"] = df["closed_date"].dt.to_period("Q").astype(str)
    df["created_month"] = df["created_date"].dt.to_period("M").astype(str)
    df["closed_month"] = df["closed_date"].dt.to_period("M").astype(str)

    # Binary outcome for modeling
    df["is_won"] = (df["outcome"] == "Won").astype(int)

    # Deal size buckets (business-meaningful thresholds)
    df["deal_size_bucket"] = pd.cut(
        df["deal_amount"],
        bins=[0, 5000, 15000, 30000, 50000, np.inf],
        labels=["<5K", "5K-15K", "15K-30K", "30K-50K", "50K+"],
    )

    # Sales cycle buckets
    df["cycle_bucket"] = pd.cut(
        df["sales_cycle_days"],
        bins=[0, 14, 30, 60, 90, np.inf],
        labels=["<2wk", "2-4wk", "1-2mo", "2-3mo", "3mo+"],
    )

    return df


def get_quarter_order(df: pd.DataFrame) -> list:
    """Return chronologically sorted list of quarters present in the data."""
    quarters = sorted(df["created_quarter"].unique())
    return quarters


def get_closed_quarter_order(df: pd.DataFrame) -> list:
    """Return chronologically sorted list of closed quarters."""
    quarters = sorted(df["closed_quarter"].unique())
    return quarters
