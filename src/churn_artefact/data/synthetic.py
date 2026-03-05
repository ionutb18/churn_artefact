from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def generate_saas_dataset(
    n_rows: int = 5000,
    churn_rate_target: float = 0.22,
    label_noise: float = 0.05,
    seed: int = 42,
    time_index: bool = True
) -> pd.DataFrame:
    """Controlled SaaS-like simulation (associative signals; NOT real data)."""
    rng = np.random.default_rng(seed)

    plan_tier = rng.choice(["basic", "pro", "enterprise"], size=n_rows, p=[0.55, 0.35, 0.10])
    contract_type = rng.choice(["monthly", "annual"], size=n_rows, p=[0.7, 0.3])
    auto_renew = rng.choice([0, 1], size=n_rows, p=[0.25, 0.75])

    tenure_months = rng.integers(0, 48, size=n_rows)
    seat_count = rng.integers(1, 200, size=n_rows)

    base_fee = np.select(
        [plan_tier == "basic", plan_tier == "pro", plan_tier == "enterprise"],
        [rng.normal(25, 5, size=n_rows), rng.normal(60, 10, size=n_rows), rng.normal(180, 30, size=n_rows)]
    )
    base_fee = np.clip(base_fee, 5, None)
    discount_flag = rng.choice([0, 1], size=n_rows, p=[0.8, 0.2])
    monthly_fee = base_fee * (1.0 - 0.15 * discount_flag)

    logins_30d = rng.poisson(lam=np.clip(18 - 0.25 * tenure_months, 3, 25), size=n_rows)
    active_days_30d = np.clip(rng.normal(loc=np.minimum(20, 10 + 0.2 * logins_30d), scale=4, size=n_rows), 0, 30).round().astype(int)
    feature_usage_score = np.clip(rng.normal(loc=0.55 + 0.12 * (plan_tier == "pro") + 0.2 * (plan_tier == "enterprise"), scale=0.18, size=n_rows), 0, 1)
    avg_session_minutes = np.clip(rng.normal(loc=8 + 6 * feature_usage_score, scale=3, size=n_rows), 1, 60)

    onboarding_completed = rng.choice([0, 1], size=n_rows, p=[0.25, 0.75])
    time_to_first_value_days = np.clip(rng.normal(loc=14 - 6*onboarding_completed + 4*(plan_tier=="basic"), scale=6, size=n_rows), 0, 60)

    tickets_30d = rng.poisson(lam=np.clip(1.5 + 3.0*(1-feature_usage_score) + 0.6*(plan_tier=="enterprise"), 0.2, 7), size=n_rows)
    payment_failures_30d = rng.poisson(lam=np.clip(0.25 + 0.8*(contract_type=="monthly") + 0.6*(discount_flag==1), 0, 3), size=n_rows)
    nps_proxy = np.clip(rng.normal(loc=30 + 40*feature_usage_score - 8*tickets_30d, scale=15, size=n_rows), -100, 100)

    cohort_month = rng.integers(0, 24, size=n_rows) if time_index else np.zeros(n_rows, dtype=int)

    z = (
        -1.2 * feature_usage_score
        -0.05 * active_days_30d
        -0.03 * logins_30d
        +0.8 * (contract_type == "monthly").astype(int)
        +0.35 * (plan_tier == "basic").astype(int)
        +0.55 * payment_failures_30d
        +0.22 * tickets_30d
        +0.04 * time_to_first_value_days
        -0.02 * tenure_months
        -0.015 * (nps_proxy)
        -0.25 * auto_renew
    )

    lo, hi = -10.0, 10.0
    for _ in range(40):
        mid = (lo + hi) / 2
        pm = _sigmoid(z + mid)
        if pm.mean() > churn_rate_target:
            hi = mid
        else:
            lo = mid
    p = _sigmoid(z + (lo + hi) / 2)

    y = rng.binomial(1, p, size=n_rows).astype(int)

    if label_noise > 0:
        flip = rng.binomial(1, label_noise, size=n_rows).astype(bool)
        y[flip] = 1 - y[flip]

    df = pd.DataFrame({
        "plan_tier": plan_tier,
        "contract_type": contract_type,
        "auto_renew": auto_renew,
        "tenure_months": tenure_months,
        "seat_count": seat_count,
        "monthly_fee": monthly_fee.round(2),
        "discount_flag": discount_flag,
        "logins_30d": logins_30d,
        "active_days_30d": active_days_30d,
        "feature_usage_score": feature_usage_score.round(3),
        "avg_session_minutes": avg_session_minutes.round(2),
        "onboarding_completed": onboarding_completed,
        "time_to_first_value_days": time_to_first_value_days.round(1),
        "tickets_30d": tickets_30d,
        "payment_failures_30d": payment_failures_30d,
        "nps_proxy": nps_proxy.round(1),
        "cohort_month": cohort_month,
        "churn_next_30d": y
    })

    for col, rate in [("nps_proxy", 0.03), ("time_to_first_value_days", 0.02)]:
        m = rng.binomial(1, rate, size=n_rows).astype(bool)
        df.loc[m, col] = np.nan

    return df

def maybe_generate_and_save(path: str | Path, generator_cfg: Dict[str, Any], seed: int = 42) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    df = generate_saas_dataset(seed=seed, **generator_cfg)
    df.to_csv(path, index=False)
    return path
