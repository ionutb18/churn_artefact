from __future__ import annotations
from pathlib import Path
from typing import Dict
import pandas as pd

DEFAULT_ACTION_MAP: Dict[str, str] = {
    # Telco-like / subscription signals
    "Contract": "Offer longer-term contract incentives or value communication to reduce month-to-month churn.",
    "PaymentMethod": "Investigate billing friction; simplify payment options; proactively contact at-risk customers with payment issues.",
    "PaperlessBilling": "Check communication clarity; ensure invoices/notifications are understandable and timely.",
    "TechSupport": "Provide proactive support; improve onboarding to reduce support-related dissatisfaction.",
    "OnlineSecurity": "Bundle perceived-value features; communicate security benefits.",
    "tenure": "Early-life retention: improve onboarding, time-to-value, and first 30 days success plan.",
    "MonthlyCharges": "Price sensitivity: test discounts/plan fit; highlight ROI; reduce surprise charges.",
    "TotalCharges": "Lifecycle/value: segment customers and tailor offers to engagement stage.",
    # SaaS synthetic signals
    "feature_usage_score": "Drive product adoption: in-app guidance, feature discovery, training webinars.",
    "payment_failures_30d": "Billing friction: dunning optimisation, alternative payment methods, proactive outreach.",
    "tickets_30d": "Support escalation: prioritise CSM intervention, resolve recurring issues, improve help content.",
    "time_to_first_value_days": "Improve onboarding: reduce setup time, guided activation, templates.",
    "active_days_30d": "Engagement: nudges, lifecycle emails, value reminders tied to goals.",
    "logins_30d": "Engagement: re-activation campaigns and personalised prompts."
}

def build_actionability_report(feature_importance_csv: str | Path, out_dir: str | Path, action_map: Dict[str, str] | None = None, top_n: int = 12) -> Path:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    action_map = action_map or DEFAULT_ACTION_MAP
    imp = pd.read_csv(feature_importance_csv)
    top = imp.head(top_n).copy()

    def map_action(feature: str) -> str:
        if feature in action_map:
            return action_map[feature]
        for k, v in action_map.items():
            if feature.startswith(k + "_") or k in feature:
                return v
        return "Investigate this feature: validate data quality and design an A/B test for any proposed intervention."

    top["suggested_intervention"] = top["feature"].astype(str).apply(map_action)
    out_path = out_dir / "actionability_report.csv"
    top.to_csv(out_path, index=False)
    return out_path
