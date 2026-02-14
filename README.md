# Sales Intelligence Challenge

## Approach

This solution follows a **top-down diagnostic framework**, starting from the business problem and working down into data and systems:

1. **Part 1 (Problem Framing):** Decoded the CRO's complaint. The real problem isn't "win rate is low" -- it's "the CRO has no diagnostic visibility into WHY it's low or WHERE to intervene."

2. **Part 2 (EDA & Insights):** Validated hypotheses with data. Found three key insights:
   - **Rep-industry fit** is the strongest signal in the data (10.9pp spread, 50pp in interactions)
   - The Q1 2024 decline is **industry-systemic but rep-divergent** (all industries dropped, but reps diverged)
   - **Inbound lead quality collapse** is the biggest momentum problem (8pp drop, 26% of volume)
   - Introduced two custom metrics: **RSFS** (Rep-Segment Fit Score) and **SMI** (Segment Momentum Index)

3. **Part 3 (Decision Engine):** Built a two-model Win Rate Driver Analysis proving **execution > profile** (RSFS alone AUC 0.59 vs. deal characteristics AUC 0.48). Translated this into a rule-based system: RED/YELLOW/GREEN deal flags, **co-lead partnerships** (top-3 round-robin pairing low-fit reps with high-RSFS co-leads), three-tier rep audit, and a ~$4M addressable revenue gap estimate.

4. **Part 4 (System Design):** Designed a lightweight Sales Insight & Alert System built around the execution-over-profile thesis — including co-lead matching engine, monthly rep audits, example alerts, scheduling cadence, and failure mode analysis.

5. **Part 5 (Reflection):** Honest assessment of weakest assumptions (co-lead lift is correlational, not causal), production risks (adoption, attribution, alert fatigue), and next steps.

## Project Structure

```
sales-intelligence-challenge/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── data/
│   └── Company_sales_data.csv          # Input dataset (5000 deals)
├── notebooks/
│   ├── 01_problem_framing.ipynb        # Part 1 - Business problem analysis (no code)
│   ├── 02_eda_and_insights.ipynb       # Part 2 - EDA, insights, custom metrics
│   ├── 03_decision_engine.ipynb        # Part 3 - Win Rate Driver Analysis
│   ├── 04_system_design.ipynb          # Part 4 - System architecture
│   └── 05_reflection.ipynb            # Part 5 - Reflection
├── src/
│   ├── __init__.py
│   ├── data_loader.py                  # Data loading & enrichment
│   ├── metrics.py                      # Standard + custom metrics (RSFS, SMI)
│   └── win_rate_drivers.py             # Decision engine (logistic regression)
└── outputs/
    └── *.png                           # Generated charts
```

## How to Run

### Prerequisites
- Python 3.9+
- pip

### Setup
```bash
cd sales-intelligence-challenge
pip install -r requirements.txt
```

### Run the Notebooks
```bash
cd notebooks
jupyter notebook
```

Open notebooks in order: `01_problem_framing.ipynb` through `05_reflection.ipynb`.

### Run the Analysis as Scripts (alternative)
```bash
cd sales-intelligence-challenge
python -c "
import sys; sys.path.insert(0, '.')
from src.data_loader import load_sales_data
from src.metrics import compute_rsfs, compute_smi_all_segments
from src.win_rate_drivers import fit_driver_model, format_driver_report

df = load_sales_data()
result = fit_driver_model(df)
print(format_driver_report(result))
"
```

## Key Decisions

| Decision | Choice | Reasoning |
|----------|--------|-----------|
| Decision engine option | **Option B: Win Rate Drivers** | Directly answers the CRO's question ("what is going wrong"). Other options are downstream of this analysis. |
| Model type | **Logistic Regression** | Coefficients are interpretable as "this factor changes win probability by X%". CROs need explanations, not black-box predictions. |
| Custom metric 1 | **RSFS (Rep-Segment Fit Score)** | Uses rep x industry win rates to capture execution quality. The strongest signal in the data (50pp interaction spread). |
| Custom metric 2 | **SMI (Segment Momentum Index)** | win_rate_delta x volume_share. Identifies segments that are both large AND deteriorating -- the most dangerous combination. |
| Code structure | **Notebooks + src/ modules** | Notebooks for narrative/storytelling, src/ for reusable production-grade code. Shows both analytical and engineering thinking. |
| Two-model comparison | **Deal chars only vs. Deal chars + RSFS** | Proves execution (who works the deal) matters more than profile (what the deal looks like). |
| Actionable output | **Co-lead partnerships (not reassignment)** | Full reassignment doubles workload; co-lead partnerships keep advisory burden manageable while creating built-in knowledge transfer. |
| Honest model assessment | **Reported ~55-57% accuracy transparently** | Didn't over-tune or cherry-pick. The weak model is itself an insight about CRM data limitations. |

## Custom Metrics Explained

### Rep-Segment Fit Score (RSFS)
- **What:** For each deal, the historical win rate of the assigned rep in the deal's specific industry
- **Why it works:** Rep x industry is the strongest interaction in the data (50pp spread). rep_23 wins 65% of HealthTech but only 24% of SaaS deals.
- **Use:** Deal assignment (route deals to high-fit reps), pipeline risk assessment, coaching prioritization
- **Design note:** We initially explored a deal-profile-only metric (DQS) based on industry x product and source x region interactions. It showed only 0.3pp separation between won and lost -- not meaningful. This itself is an important finding: deal profile doesn't predict outcomes; deal execution does.

### Segment Momentum Index (SMI)
- **What:** `win_rate_change x volume_share` per segment
- **Use:** Ranks segments by "danger level" -- a large segment with declining win rate is more urgent than a small one with a bigger drop
- **Interpretation:** Negative SMI = investigate first; Positive SMI = learn from and replicate
