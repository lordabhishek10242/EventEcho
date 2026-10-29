import numpy as np
import pandas as pd
import pytest

from src.pooled_analysis import caar_tables, pattern_generalization, window_choice_tables


def _study(ars: dict[tuple[str, str], float], missing: tuple[str, str, int] | None = None) -> pd.DataFrame:
    rows = []
    for (event_id, stock), ar in ars.items():
        for day in range(-5, 6):
            if missing == (event_id, stock, day):
                continue
            rows.append({"event_id": event_id, "stock": stock, "event_day": day, "abnormal_return": ar})
    return pd.DataFrame(rows)


def _cars(study: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (event_id, stock), group in study.groupby(["event_id", "stock"]):
        for window, (lo, hi) in {"CAR_-1_+1": (-1, 1), "CAR_-3_+3": (-3, 3), "CAR_-5_+5": (-5, 5)}.items():
            part = group[group["event_day"].between(lo, hi)]
            rows.append({"event_id": event_id, "stock": stock, "window": window,
                         "CAR": part["abnormal_return"].sum() if len(part) == hi - lo + 1 else np.nan})
    return pd.DataFrame(rows)


EVENTS = pd.DataFrame({
    "event_id": ["E1", "E2", "E3", "E4"],
    "category": ["Monetary Policy", "Monetary Policy", "Corporate Earnings", "Economic Release"],
})
AUDIT = pd.DataFrame({"event_id": ["E1", "E2", "E3", "E4"], "event_window_overlap_count": [0, 0, 0, 0]})
CLUSTERS = {"E1": "C1", "E2": "C2", "E3": "C3", "E4": "C4"}


def test_caar_path_is_cumulative_average_and_matches_pooled_car():
    ars = {(e, s): v for e, v in zip(["E1", "E2", "E3", "E4"], [0.01, 0.03, -0.02, 0.0]) for s in ("A", "B")}
    study = _study(ars)
    path, windows = caar_tables(study, _cars(study), EVENTS, ["A", "B"], AUDIT, CLUSTERS, ["E1", "E2"], ["E3", "E4"])
    everything = path[(path["group"] == "All events")].set_index("event_day")
    mean_ar = np.mean([0.01, 0.03, -0.02, 0.0])
    assert everything.loc[-5, "AAR_pct"] == pytest.approx(mean_ar * 100)
    assert everything.loc[5, "CAAR_pct"] == pytest.approx(mean_ar * 11 * 100)
    pooled = windows[(windows["group"] == "All events") & (windows["window"] == "CAR_-5_+5")].iloc[0]
    assert pooled["mean_CAR_pct"] == pytest.approx(everything.loc[5, "CAAR_pct"])
    assert pooled["events"] == 4 and pooled["dependency_clusters"] == 4


def test_incomplete_event_path_is_dropped_not_treated_as_zero():
    ars = {(e, s): 0.01 for e in ["E1", "E2", "E3", "E4"] for s in ("A",)}
    study = _study(ars, missing=("E4", "A", 0))
    path, _ = caar_tables(study, _cars(study), EVENTS, ["A"], AUDIT, CLUSTERS, ["E1", "E2"], ["E3", "E4"])
    everything = path[path["group"] == "All events"]
    assert set(everything["events"]) == {3}
    assert everything.set_index("event_day").loc[5, "CAAR_pct"] == pytest.approx(11.0)


def test_groups_with_fewer_than_three_clusters_get_no_interval():
    ars = {(e, s): v for e, v in zip(["E1", "E2", "E3", "E4"], [0.01, 0.02, 0.03, 0.04]) for s in ("A",)}
    study = _study(ars)
    _, windows = caar_tables(study, _cars(study), EVENTS, ["A"], AUDIT, CLUSTERS, ["E1", "E2"], ["E3", "E4"])
    monetary = windows[(windows["group"] == "monetary_policy") & (windows["window"] == "CAR_-1_+1")].iloc[0]
    assert monetary["status"] == "fewer_than_3_clusters_no_interval"
    assert np.isnan(monetary["mean_CAR_ci95_lower_pct"])
    single = windows[(windows["group"] == "corporate_earnings")].iloc[0]
    assert single["status"] == "single_event_descriptive_only"
    everything = windows[(windows["group"] == "All events") & (windows["window"] == "CAR_-1_+1")].iloc[0]
    assert everything["mean_CAR_ci95_lower_pct"] <= everything["mean_CAR_pct"] <= everything["mean_CAR_ci95_upper_pct"]


def test_window_choice_counts_verdict_changes():
    sig = pd.DataFrame([
        {"event_id": "E1", "stock": "A", "window": w, "CAR_pct": c, "standard_error_pct": 1.0, "hac_standard_error_pct": 1.0,
         "significant_after_holm_5pct": s, "hac_significant_after_holm_5pct": s}
        for w, c, s in (("CAR_-1_+1", 2.0, True), ("CAR_-3_+3", 1.0, False), ("CAR_-5_+5", -1.0, False))
    ])
    summary, agreement = window_choice_tables(sig)
    assert list(summary["hac_holm_significant"]) == [1, 0, 0]
    narrow_wide = agreement[(agreement["narrow_window"] == "CAR_-1_+1") & (agreement["wide_window"] == "CAR_-5_+5")].iloc[0]
    assert narrow_wide["verdict_changes"] == 1 and narrow_wide["hac_significant_only_narrow"] == 1
    assert narrow_wide["same_sign_share_pct"] == 0


def test_pattern_generalization_scores_sign_predictions():
    ars = {("E1", "A"): 0.01, ("E2", "A"): 0.02, ("E3", "A"): 0.01, ("E4", "A"): -0.01}
    events = EVENTS.assign(category=["Monetary Policy"] * 4)
    study = _study(ars)
    result = pattern_generalization(_cars(study), events, ["A"], ["E1", "E2"], ["E3", "E4"])
    rule = result[(result["test"] == "predict_holdout_sign_from_development_class_mean") & (result["window"] == "CAR_-1_+1")].iloc[0]
    assert rule["statistic"] == 1 and rule["holdout_n"] == 2
    assert bool(rule["replicated"]) is False
