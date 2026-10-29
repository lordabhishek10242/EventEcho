"""Cluster-bootstrap uncertainty for event-level strategy performance.

Events are joined into dependence clusters when their CAR windows overlap or
when the event registry assigns them the same episode. The bootstrap resamples
whole clusters, not individual events. With few clusters, intervals remain
exploratory and should not be interpreted as proof of a trading edge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _dependency_clusters(
    events: pd.DataFrame,
    event_window_audit: pd.DataFrame,
) -> dict[str, str]:
    """Return connected-component labels using shared episode or window overlap."""
    if "event_id" not in events:
        raise ValueError("events must include event_id")
    if events["event_id"].isna().any() or events["event_id"].astype(str).duplicated().any():
        raise ValueError("event IDs must be present and unique")

    ids = events["event_id"].astype(str).tolist()
    parent = {event_id: event_id for event_id in ids}

    def find(event_id: str) -> str:
        while parent[event_id] != event_id:
            parent[event_id] = parent[parent[event_id]]
            event_id = parent[event_id]
        return event_id

    def union(left: str, right: str) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            # Stable root makes cluster labels deterministic across runs.
            first, second = sorted((root_left, root_right))
            parent[second] = first

    metadata = events.copy()
    metadata["event_id"] = metadata["event_id"].astype(str)
    if "episode_id" not in metadata:
        metadata["episode_id"] = metadata["event_id"]
    metadata["episode_id"] = metadata["episode_id"].where(
        metadata["episode_id"].notna(), metadata["event_id"]
    ).astype(str)
    for _, group in metadata.groupby("episode_id", sort=False):
        members = group["event_id"].tolist()
        for other in members[1:]:
            union(members[0], other)

    required_audit = {"event_id", "event_window_overlap_event_ids"}
    if not required_audit.issubset(event_window_audit.columns):
        raise ValueError(f"event_window_audit must include {sorted(required_audit)}")
    known = set(ids)
    for _, row in event_window_audit.iterrows():
        event_id = str(row["event_id"])
        if event_id not in known:
            continue
        partners = row["event_window_overlap_event_ids"]
        if pd.isna(partners) or not str(partners).strip():
            continue
        for partner in str(partners).split(";"):
            partner = partner.strip()
            if partner in known and partner != event_id:
                union(event_id, partner)

    roots = {event_id: find(event_id) for event_id in ids}
    # Canonical labels are independent of input row order.
    canonical = {root: f"CLUSTER_{index:03d}" for index, root in enumerate(sorted(set(roots.values())), 1)}
    return {event_id: canonical[root] for event_id, root in roots.items()}


def clustered_strategy_inference(
    event_returns: pd.DataFrame,
    events: pd.DataFrame,
    event_window_audit: pd.DataFrame,
    development_ids: tuple[str, ...],
    holdout_ids: tuple[str, ...],
    n_bootstrap: int = 5000,
    seed: int = 2026,
) -> pd.DataFrame:
    """Estimate cluster-bootstrap CIs for development and holdout strategy metrics.

    The point estimates use the event-weighted estimand, matching the existing
    sample summary. Bootstrap replicates sample dependency clusters with
    replacement and retain every available event inside each sampled cluster.
    """
    if n_bootstrap < 100:
        raise ValueError("n_bootstrap must be at least 100")
    required = {"event_id", "portfolio_net_return", "benchmark_return"}
    if not required.issubset(event_returns.columns):
        raise ValueError(f"event_returns must include {sorted(required)}")
    if event_returns["event_id"].isna().any() or event_returns["event_id"].astype(str).duplicated().any():
        raise ValueError("event_returns must contain one row per non-null event_id")

    clusters = _dependency_clusters(events, event_window_audit)
    frame = event_returns.copy()
    frame["event_id"] = frame["event_id"].astype(str)
    frame["dependency_cluster"] = frame["event_id"].map(clusters)
    frame = frame[frame["dependency_cluster"].notna()].copy()
    samples = {
        "Development": set(map(str, development_ids)),
        "Holdout": set(map(str, holdout_ids)),
    }
    rng = np.random.default_rng(seed)
    output: list[dict] = []

    for sample_name, sample_ids in samples.items():
        sample = frame[frame["event_id"].isin(sample_ids)].copy()
        sample["excess_return"] = sample["portfolio_net_return"] - sample["benchmark_return"]
        cluster_ids = sorted(sample["dependency_cluster"].dropna().unique())
        cluster_groups = {
            cluster_id: sample[sample["dependency_cluster"] == cluster_id]
            for cluster_id in cluster_ids
        }
        cluster_count = len(cluster_ids)
        valid_excess = sample["excess_return"].dropna()
        point = {
            "mean_net_return_pct": float(sample["portfolio_net_return"].mean() * 100) if not sample.empty else np.nan,
            "mean_excess_return_pct": float(valid_excess.mean() * 100) if not valid_excess.empty else np.nan,
            "benchmark_outperformance_rate_pct": float((valid_excess > 0).mean() * 100) if not valid_excess.empty else np.nan,
        }
        draws = {key: [] for key in point}
        if cluster_count >= 2:
            for _ in range(n_bootstrap):
                picked = rng.choice(cluster_ids, size=cluster_count, replace=True)
                boot = pd.concat([cluster_groups[cluster_id] for cluster_id in picked], ignore_index=True)
                boot_excess = boot["portfolio_net_return"] - boot["benchmark_return"]
                draws["mean_net_return_pct"].append(float(boot["portfolio_net_return"].mean() * 100))
                valid = boot_excess.dropna()
                draws["mean_excess_return_pct"].append(float(valid.mean() * 100) if not valid.empty else np.nan)
                draws["benchmark_outperformance_rate_pct"].append(
                    float((valid > 0).mean() * 100) if not valid.empty else np.nan
                )

        row: dict[str, object] = {
            "sample": sample_name,
            "events_available": int(sample["event_id"].nunique()),
            "independent_dependency_clusters": cluster_count,
            "largest_cluster_events": max((len(group) for group in cluster_groups.values()), default=0),
            "bootstrap_replicates": n_bootstrap,
            "bootstrap_seed": seed,
            "bootstrap_unit": "connected component of overlapping CAR windows and shared episode IDs",
            "inference_status": "cluster_bootstrap_exploratory" if cluster_count >= 2 else "insufficient_independent_clusters",
        }
        for metric, value in point.items():
            row[f"{metric}_point"] = value
            values = np.asarray(draws[metric], dtype=float)
            values = values[np.isfinite(values)]
            row[f"{metric}_ci95_lower"] = float(np.quantile(values, 0.025)) if len(values) else np.nan
            row[f"{metric}_ci95_upper"] = float(np.quantile(values, 0.975)) if len(values) else np.nan
        output.append(row)
    return pd.DataFrame(output)
