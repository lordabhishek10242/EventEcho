"""Stable result schemas, artifact writing, and provenance manifest."""

from __future__ import annotations

import html as html_lib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ARTIFACT_DESCRIPTIONS = {
    "event_master.csv": "Event identifiers, source metadata, original and effective dates.",
    "multi_stock_event_study.csv": "Per-event, per-stock daily returns, expected returns, and abnormal returns.",
    "multi_stock_car.csv": "Complete cumulative abnormal returns for each event, stock, and window.",
    "cross_stock_consistency.csv": "Cross-stock CAR summaries by event and window.",
    "event_window_audit.csv": "Trading-session intersections across event-study and volatility windows.",
    "event_level_audit.csv": "Event-level overlap, shared-episode/category, and verification-caution diagnostics.",
    "event_pair_overlap_audit.csv": "Unique event-pair dependence audit; shared metadata is not causal evidence.",
    "event_relationship_highlights.csv": "Ranked event-level audit highlights for reviewer attention.",
    "trade_candidates.csv": "Per-event and per-stock simulated entries, exits, costs, and benchmark returns.",
    "event_trading_results.csv": "Equal-weighted event portfolio returns and benchmark excess returns.",
    "sample_performance.csv": "Event-weighted development and holdout performance summaries.",
    "clustered_strategy_inference.csv": "Development and holdout strategy metrics with cluster-bootstrap intervals resampling overlapping-window/shared-episode components.",
    "transaction_cost_sensitivity.csv": "Strategy summary metrics at configured per-side cost levels.",
    "final_strategy_assessment.csv": "Selected headline trading metrics at the configured base cost.",
    "multi_stock_significance.csv": "Conventional and Newey-West/HAC-sensitivity CAR inference, each with Holm-adjusted p-values.",
    "estimation_window_sensitivity.csv": "Per-event, stock, and CAR-window inference at 60-, 100-, and 150-session market-model estimation lengths.",
    "estimation_window_sensitivity_summary.csv": "Summary of CAR stability and Holm-significance counts across pre-specified estimation-window lengths.",
    "robustness_evidence_ledger.csv": "Per-event/stock/window ledger of cross-specification significance stability, CAR sensitivity, overlap and timing caveats.",
    "single_stock_event_study.csv": "Reliance subset of the generated multi-stock daily event-study output.",
    "covid_extreme_day_sensitivity.csv": "EVT002 extreme-day diagnostics; partial-window sums are not reported as comparable CARs.",
    "event_timing_sensitivity.csv": "Alternative first-session versus following-session CARs for events with unresolved timing metadata.",
    "event_volatility.csv": "Pre/post realized return volatility and event-window residual volatility by event and stock.",
    "category_car_summary.csv": "Episode-level category CAR summaries with overlapping event windows excluded.",
    "category_volatility_summary.csv": "Episode-level category volatility summaries with overlap and sample-size diagnostics.",
    "caar_event_time.csv": "Average abnormal return and CAAR by event day for all events, event classes, sectors, stocks and samples, with cluster-bootstrap intervals.",
    "caar_window_summary.csv": "Pooled mean CAR by group and window with cluster-bootstrap intervals and sample-size status.",
    "window_choice_summary.csv": "How CAR size, standard errors and significance counts change across the three event windows.",
    "window_choice_agreement.csv": "Pairwise sign agreement and significance-verdict changes between event windows.",
    "pattern_generalization.csv": "Whether development-sample CAR patterns repeat in the later holdout events.",
    "headline_classifier_metrics.csv": "Headline classifier accuracy and macro-F1 under grouped cross-validation and on the registry headlines, with baselines.",
    "headline_classifier_per_class.csv": "Per-class precision, recall and F1 from grouped cross-validation.",
    "headline_classifier_confusion.csv": "Confusion counts from grouped cross-validation.",
    "event_headline_classification.csv": "Predicted class for each registry headline next to its hand-assigned class.",
}


def _truthy(value: object) -> bool | None:
    """Parse CSV booleans; return None when the value is unset."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(number) else number


def price_convention_summary(manifest: dict) -> dict:
    """Describe the stock/benchmark price fields recorded for this exact run.

    Reads the run manifest's ``price_fields`` first and the snapshot
    provenance ``adjustment_policy`` second. Nothing is assumed: a missing
    record is reported as not recorded instead of a default label.
    """
    fields = manifest.get("price_fields") or {}
    policy = (manifest.get("input_data_provenance") or {}).get("adjustment_policy") or {}
    stock = fields.get("stock_field") or policy.get("stock_field")
    benchmark = fields.get("benchmark_field") or policy.get("benchmark_field")
    if not stock or not benchmark:
        return {
            "stock_field": stock, "benchmark_field": benchmark, "status": "not_recorded",
            "note": "The run manifest does not record both price fields, so the return convention cannot be confirmed for this dashboard.",
        }
    if str(stock).strip().lower() == str(benchmark).strip().lower():
        adjusted = str(stock).strip().lower() in {"adj close", "adjusted close"}
        note = f"Both stocks and the benchmark use the {stock} field, so both sides are measured on the same price basis."
        if not adjusted:
            note += " These are price returns: dividends are excluded on both sides, so this is not a total shareholder return comparison."
        return {"stock_field": stock, "benchmark_field": benchmark, "status": "consistent", "note": note}
    return {
        "stock_field": stock, "benchmark_field": benchmark, "status": "mismatch",
        "note": f"Return-convention mismatch: stocks use {stock} while the benchmark uses {benchmark}. Stock-versus-benchmark comparisons mix different dividend treatments.",
    }


def dashboard_evidence(manifest: dict, datasets: dict[str, list[dict]]) -> dict:
    """Pre-compute dashboard evidence from the run's own tables.

    Everything here is derived from the manifest and CSVs embedded in the same
    page, so the summary cannot drift from the downloadable outputs. Three
    different overlap notions are kept apart: computed CAR-window overlap,
    computed volatility-window overlap, and the broad manual registry flag.
    """
    audit = datasets.get("event_window_audit.csv", [])
    events = datasets.get("event_master.csv", [])
    pairs = datasets.get("event_pair_overlap_audit.csv", [])
    car_overlap = sorted(str(r["event_id"]) for r in audit if (_number(r.get("event_window_overlap_count")) or 0) > 0)
    vol_overlap = sorted(str(r["event_id"]) for r in audit if (_number(r.get("volatility_window_overlap_count")) or 0) > 0)
    master = [_truthy(r.get("overlap_flag")) for r in events]
    timing: dict[str, int] = {}
    for row in events:
        key = str(row.get("timestamp_evidence_status") or "not_recorded")
        timing[key] = timing.get(key, 0) + 1
    review = sum(1 for r in events if _truthy(r.get("timestamp_review_required")) is not False)

    intervals = []
    for row in datasets.get("clustered_strategy_inference.csv", []):
        lower = _number(row.get("mean_excess_return_pct_ci95_lower"))
        upper = _number(row.get("mean_excess_return_pct_ci95_upper"))
        if lower is None or upper is None:
            status = "not_available"
        else:
            status = "includes_zero" if lower <= 0 <= upper else "excludes_zero"
        intervals.append({"sample": row.get("sample"), "ci_lower_pp": lower, "ci_upper_pp": upper, "interval_status": status})
    return {
        "price_convention": price_convention_summary(manifest),
        "run": {
            "snapshot_id": manifest.get("snapshot_id"),
            "generated_utc": manifest.get("generated_utc"),
            "code_commit": manifest.get("code_commit"),
            "prices_sha256": (manifest.get("input_checksums_sha256") or {}).get("prices"),
        },
        "overlap": {
            "car_window_overlap_event_ids": car_overlap,
            "volatility_window_overlap_event_ids": vol_overlap,
            "car_window_overlap_pairs": sum(1 for r in pairs if (_number(r.get("overlapping_session_count")) or 0) > 0),
            "registry_flag_true": sum(1 for v in master if v is True),
            "registry_flag_false": sum(1 for v in master if v is False),
            "registry_flag_unset": sum(1 for v in master if v is None),
        },
        "timing": {"by_evidence_status": timing, "review_required": review, "events": len(events)},
        "strategy_intervals": intervals,
    }


def _utc_label(value: object) -> str:
    """Short, readable UTC time for the header; falls back to the raw value."""
    if not value:
        return "time not recorded"
    try:
        stamp = pd.Timestamp(str(value))
        stamp = stamp.tz_convert("UTC") if stamp.tzinfo else stamp
        return stamp.strftime("%Y-%m-%d %H:%M UTC")
    except (ValueError, TypeError):
        return str(value)


def _holdout_warning_html(intervals: list[dict]) -> str:
    """Render a visible warning when the holdout excess-return interval includes zero."""
    holdout = next((row for row in intervals if row.get("sample") == "Holdout"), None)
    if holdout is None or holdout["interval_status"] == "not_available":
        return '<div class="noteBox">Holdout cluster-bootstrap interval is not available in this run.</div>'
    lower, upper = holdout["ci_lower_pp"], holdout["ci_upper_pp"]
    if holdout["interval_status"] == "includes_zero":
        return (
            '<div class="noteBox"><strong>Warning: the holdout 95% interval includes zero</strong> '
            f"({lower:+.2f} to {upper:+.2f} pp). The data are consistent with no excess return; "
            "no reliable trading edge is established.</div>"
        )
    return (
        f'<div class="noteBox">The holdout 95% interval ({lower:+.2f} to {upper:+.2f} pp) excludes zero, '
        "but cluster counts are small and the event sample is hand-selected, so this remains exploratory evidence, not proof of an edge.</div>"
    )


def build_demo_dashboard(analysis_dir: str | Path, output_path: str | Path | None = None) -> Path:
    """Build a self-contained interactive dashboard from a verified historical analysis."""
    root = Path(analysis_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Analysis manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("input_data_classification") != "historical_market_data"
        or manifest.get("real_market_evidence") is not True
        or not manifest.get("snapshot_id")
    ):
        raise ValueError("Demo dashboard requires a frozen historical-market-data analysis manifest")

    names = (
        "event_master.csv", "multi_stock_event_study.csv", "multi_stock_car.csv",
        "multi_stock_significance.csv", "estimation_window_sensitivity_summary.csv", "robustness_evidence_ledger.csv", "cross_stock_consistency.csv",
        "event_window_audit.csv", "event_level_audit.csv", "event_pair_overlap_audit.csv",
        "event_relationship_highlights.csv", "event_timing_sensitivity.csv",
        "event_volatility.csv", "category_car_summary.csv", "category_volatility_summary.csv",
        "event_trading_results.csv", "sample_performance.csv", "clustered_strategy_inference.csv",
        "transaction_cost_sensitivity.csv", "covid_extreme_day_sensitivity.csv",
        "caar_event_time.csv", "caar_window_summary.csv", "window_choice_summary.csv", "window_choice_agreement.csv",
        "pattern_generalization.csv", "headline_classifier_metrics.csv", "headline_classifier_per_class.csv",
        "event_headline_classification.csv",
    )
    datasets: dict[str, list[dict]] = {}
    for name in names:
        path = root / name
        if path.is_file():
            try:
                frame = pd.read_csv(path)
                datasets[name] = frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")
            except pd.errors.EmptyDataError:
                datasets[name] = []

    evidence = dashboard_evidence(manifest, datasets)
    payload = json.dumps({"manifest": manifest, "datasets": datasets, "evidence": evidence}, default=str)
    payload = payload.replace("</", "<\\/")
    title = "EventEcho | Historical Event Study"
    template = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{color-scheme:dark;--bg:#0b1020;--panel:#121a2b;--panel2:#182338;--line:#293750;--muted:#9eacc3;--text:#edf3ff;--blue:#83b8ff;--green:#63d9b0;--red:#ff8d9a;--amber:#f4c879}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(ellipse at 10% 0%,#18294a 0,transparent 38%),var(--bg);color:var(--text);font:14px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1480px;margin:auto;padding:28px 22px 60px}.eyebrow{color:var(--blue);font-size:11px;font-weight:800;letter-spacing:.16em;text-transform:uppercase}
header{display:flex;gap:24px;align-items:flex-start;justify-content:space-between;margin-bottom:22px}h1{font-size:clamp(30px,4vw,46px);letter-spacing:-.04em;line-height:1.08;margin:8px 0 12px}h2{font-size:17px;margin:0 0 14px;letter-spacing:-.015em}h3{font-size:13px;margin:0 0 8px;color:#cbd7eb}.sub{max-width:760px;color:var(--muted);font-size:15px}.snapshot{border:1px solid #385078;background:#14233a;padding:13px 15px;border-radius:12px;min-width:220px}.snapshot strong{display:block;font-size:12px;color:var(--blue);margin-bottom:5px}.snapshot span{font-size:12px;color:#c5d1e5;overflow-wrap:anywhere}
.metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:22px 0}.metric,.panel{background:linear-gradient(145deg,var(--panel),#101828);border:1px solid var(--line);border-radius:15px}.metric{padding:17px}.metric .label{color:var(--muted);font-size:12px}.metric .value{font-size:27px;font-weight:750;letter-spacing:-.03em;margin:5px 0}.metric .note{font-size:11px;color:var(--muted)}
.controls{display:flex;flex-wrap:wrap;gap:12px;align-items:end;padding:16px;margin:16px 0}.field{display:flex;flex-direction:column;gap:6px;min-width:160px}.field label{font-size:11px;color:var(--muted);font-weight:700;text-transform:uppercase;letter-spacing:.06em}select{background:#0c1423;color:var(--text);border:1px solid #3a4b66;border-radius:8px;padding:10px 11px;font:inherit;min-width:160px}.layout{display:grid;grid-template-columns:minmax(0,1.3fr) minmax(320px,.7fr);gap:14px}.panel{padding:18px;min-width:0;margin-bottom:14px}.wide{grid-column:1/-1}.muted{color:var(--muted)}.small{font-size:12px;color:var(--muted)}.pill{display:inline-block;padding:3px 8px;border-radius:99px;background:#22314a;color:#dce8fa;font-size:11px;font-weight:700}.pill.warn{background:#4a351c;color:#ffdaa0}.pill.good{background:#143d37;color:#9bf1d5}
.tablewrap{overflow:auto;max-width:100%}table{width:100%;border-collapse:collapse;font-size:12px;white-space:nowrap}th,td{text-align:right;padding:9px 10px;border-bottom:1px solid #27334a}th{color:#aebed5;background:#141e30;font-size:10px;letter-spacing:.06em;text-transform:uppercase;position:sticky;top:0}th:first-child,td:first-child{text-align:left}tr:last-child td{border-bottom:0}.empty{padding:18px;color:var(--muted);background:#0e1726;border-radius:9px}
.chart{width:100%;height:auto;display:block;overflow:visible}.legend{display:flex;flex-wrap:wrap;gap:12px;font-size:11px;color:var(--muted);margin-top:7px}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px}.eventcard{padding:13px;border:1px solid #2c3a51;background:#101a2b;border-radius:10px;margin:8px 0}.eventtitle{font-weight:700}.eventmeta{font-size:11px;color:var(--muted);margin-top:4px}.noteBox{border-left:3px solid var(--amber);padding:11px 13px;background:#2a241a;color:#f0d9aa;border-radius:0 8px 8px 0;font-size:12px;margin-top:12px}.footer{margin-top:18px;padding-top:16px;border-top:1px solid var(--line);font-size:11px;color:var(--muted)}
@media(max-width:900px){header{display:block}.snapshot{margin-top:16px}.metrics{grid-template-columns:repeat(2,minmax(0,1fr))}.layout{grid-template-columns:1fr}.wide{grid-column:auto}}@media(max-width:520px){main{padding:18px 12px 40px}.metrics{gap:8px}.metric{padding:12px}.metric .value{font-size:22px}.field,select{width:100%}.controls{display:grid;grid-template-columns:1fr}}
</style></head><body><main>
<header><div><div class="eyebrow">Quantitative research workspace / India equities</div><h1>EventEcho</h1><div class="sub">Explore stock reactions around selected policy and macroeconomic events. Compare abnormal returns against a market-model expectation, inspect uncertainty, and audit event timing and overlap. This is an evidence explorer, not a prediction engine.</div></div><div class="snapshot"><strong>FROZEN HISTORICAL SNAPSHOT</strong><span id="snapshotId">__SNAPSHOT__</span><br><span id="period"></span><br><span id="source"></span><br><span id="runInfo">Generated __GENERATED__ · code __COMMIT__</span></div></header>
<div class="metrics" id="metrics"></div>
<section class="panel controls"><div class="field"><label for="eventSelect">Event</label><select id="eventSelect"></select></div><div class="field"><label for="stockSelect">Stock</label><select id="stockSelect"></select></div><div class="field"><label for="windowSelect">CAR window</label><select id="windowSelect"></select></div><div class="field"><label for="sampleSelect">Trading sample</label><select id="sampleSelect"><option value="all">All events</option><option value="Development">Development</option><option value="Holdout">Holdout</option></select></div><p class="small" id="definitions" style="flex-basis:100%;margin:4px 0 0"></p></section>
<div class="layout">
<section class="panel"><h2>CAR heatmap</h2><p class="small">Cumulative abnormal return by event and stock. Colors encode direction and magnitude; values are percentages.</p><div id="heatmap"></div></section>
<section class="panel"><h2>Selected event</h2><div id="eventDetail"></div><h3 style="margin-top:16px">Return response around Day 0</h3><div id="returnChart"></div><div class="legend"><span><i class="dot" style="background:#83b8ff"></i>Daily abnormal return</span><span><i class="dot" style="background:#63d9b0"></i>Cumulative abnormal return</span></div></section>
<section class="panel wide"><h2>Average reaction across events (CAAR)</h2><p class="small">For each event, abnormal returns are averaged over the chosen stocks, then averaged over events and added up from Day −5. The shaded band is a 95% cluster-bootstrap interval that resamples whole groups of related events. Groups with fewer than three independent clusters get no band. Pooled averages describe association, not cause.</p><div class="field" style="max-width:420px;margin-bottom:10px"><label for="caarGroup">Group</label><select id="caarGroup"></select></div><div id="caarChart"></div><div id="caarTable"></div></section>
<section class="panel wide"><h2>Compare event classes, sectors and stocks</h2><p class="small">Mean CAR across events for the CAR window chosen above. Sector and stock rows use all events. Most sectors contain one stock, so a sector row mostly shows that one stock. Sector labels are this project's broad grouping, not an official classification.</p><div id="groupComparison"></div></section>
<section class="panel wide"><h2>Inference & uncertainty</h2><p class="small">Confidence intervals and Holm-adjusted p-values are approximate under the current model assumptions. Statistical significance is not causality.</p><div id="inference"></div></section>
<section class="panel wide"><h2>Estimation-window robustness</h2><p class="small">The market model is re-fit with 60, 100, and 150 pre-event sessions while keeping the 20-session gap and event windows fixed. Holm correction is performed separately within each specification. Treat changes as sensitivity, not a search for the most significant result.</p><div id="estimationWindowSensitivity"></div></section>
<section class="panel wide"><h2>How the event-window choice changes conclusions</h2><p class="small">Same events and stocks, three window lengths. Wider windows can capture slower reactions but also add more days of unrelated news and noise.</p><div id="windowChoice"></div></section>
<section class="panel wide"><h2>Robustness evidence ledger</h2><p class="small">Lists findings that are HAC-Holm significant in at least one estimation-window specification. “Stable” means significant in all three tested lengths; event overlap, shared episodes, and timing/source cautions are shown separately. Stable significance does not establish causality.</p><div id="robustnessLedger"></div></section>
<section class="panel wide"><h2>Across-stock consistency</h2><p class="small">Do stocks move in the same direction around this event?</p><div id="consistency"></div></section>
<section class="panel wide"><h2>Hypothetical trading diagnostic</h2><p class="small">Hypothetical, exploratory rule; not an investable backtest and not live trading performance.</p><p class="small" id="tradingRule"></p><div id="trading"></div><div id="costs" style="margin-top:12px"></div></section>
<section class="panel wide"><h2>Holdout uncertainty</h2><p class="small">Cluster-bootstrap intervals resample dependent event clusters. A holdout interval that crosses zero does not establish a reliable excess-return edge. Excess returns are in percentage points (pp) versus NIFTY 50.</p><div id="holdoutWarning">__HOLDOUT_WARNING__</div><div id="clusteredInference"></div></section>
<section class="panel wide"><h2>Do patterns repeat on unseen events?</h2><p class="small">Patterns measured on the development events (2016 to Feb 2023) are checked on the later holdout events (Oct 2023 to 2025). Event classes have very few events in each sample, so these checks have little power: "no" means the pattern did not repeat, and "yes" is not proof.</p><div id="generalization"></div></section>
<section class="panel wide"><h2>Event headline classifier (NLP)</h2><p class="small">A TF-IDF + logistic regression model is trained on real, sourced news headlines to label an event headline as one of six classes. It is checked on held-out groups of related headlines and on this project's 21 registry headlines, which are never used for training. It is a cross-check on the hand-assigned categories, not a replacement.</p><div id="headlineClassifier"></div></section>
<section class="panel wide"><h2>Event register</h2><div id="eventRegister"></div></section>
<section class="panel wide"><h2>Event dependence & independence audit</h2><p class="small">Temporal overlap is calculated on trading sessions. Shared episode IDs and categories are metadata-based clustering warnings—not evidence of causality. Pending source/timing checks remain review cautions.</p><div id="eventDependenceAudit"></div></section>
<section class="panel wide"><h2>Event timing sensitivity</h2><p class="small">For unresolved timing, compare plausible session alignments. If the context indicates an after-market/evening release, the first scenario is strictly the next available session—never the same-day close that preceded publication. For other uncertain timestamps, compare the first session on/after the date with the following session. This is a robustness diagnostic, not an automatic date correction.</p><div id="eventTimingSensitivity"></div></section>
<section class="panel wide"><h2>Realized volatility before vs after events</h2><p class="small">Realized stock and benchmark volatility use non-overlapping trading-session windows −20:−1 and +1:+20, excluding Day 0. Annualization uses √252. Event-window market-model residual volatility is shown separately and is not the same as realized stock volatility.</p><div id="volatilityChart"></div><div id="eventVolatility"></div></section>
<section class="panel wide"><h2>Category-level evidence</h2><p class="small">Category summaries average stocks within each event and then aggregate within episodes. Overlapping event windows are excluded. A category needs at least three independent episodes to receive a numeric summary; “insufficient independent episodes” is an honest data limitation, not a zero effect.</p><div id="categoryEvidence"></div></section>
<section class="panel wide"><h2>Data provenance & interpretation limits</h2><div class="noteBox" id="priceConvention" data-status="__CONVENTION_STATUS__">__CONVENTION_NOTE__</div><div id="provenance"></div><div class="noteBox">Daily data cannot isolate intraday reactions to market-hours announcements. The sample contains only a small, hand-selected set of events; overlapping news and shared macro episodes weaken independence. Do not interpret these associations as causal effects or future returns.</div></section>
</div><div class="footer">EventEcho research prototype · Data provider and timestamps are recorded in the manifest. Confirm event timing and primary sources before presenting results. No investment advice.</div>
<script id="eventecho-data" type="application/json">__PAYLOAD__</script>
<script>
(function(){
"use strict";
var root=JSON.parse(document.getElementById("eventecho-data").textContent),d=root.datasets||{},m=root.manifest||{},ev=root.evidence||{};
var windowAudit=d["event_window_audit.csv"]||[];
var events=d["event_master.csv"]||[],cars=d["multi_stock_car.csv"]||[],study=d["multi_stock_event_study.csv"]||[],sig=d["multi_stock_significance.csv"]||[],cons=d["cross_stock_consistency.csv"]||[],trades=d["event_trading_results.csv"]||[],samples=d["sample_performance.csv"]||[],costs=d["transaction_cost_sensitivity.csv"]||[],clusteredInference=d["clustered_strategy_inference.csv"]||[],covid=d["covid_extreme_day_sensitivity.csv"]||[],estimationSensitivity=d["estimation_window_sensitivity_summary.csv"]||[],robustnessLedger=d["robustness_evidence_ledger.csv"]||[],caarPath=d["caar_event_time.csv"]||[],caarWin=d["caar_window_summary.csv"]||[],winSummary=d["window_choice_summary.csv"]||[],winAgree=d["window_choice_agreement.csv"]||[],generalization=d["pattern_generalization.csv"]||[],clsMetrics=d["headline_classifier_metrics.csv"]||[],clsPerClass=d["headline_classifier_per_class.csv"]||[],clsEvents=d["event_headline_classification.csv"]||[];
var $=function(id){return document.getElementById(id);};
var esc=function(v){return String(v==null?"—":v).replace(/[&<>"']/g,function(c){return {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c];});};
var num=function(v){if(v===null||v===undefined||(typeof v==="string"&&v.trim()===""))return null;var n=Number(v);return Number.isFinite(n)?n:null;};
var raw=function(h){return {__html:h};};
var x100=function(v){var n=num(v);return n===null?null:n*100;};
var pct=function(v,digits){var n=num(v);return n===null?"—":n.toFixed(digits==null?2:digits)+"%";};
var uniq=function(a){return Array.from(new Set(a.filter(function(x){return x!==null&&x!==undefined&&x!=="";})));};
var labelWindow=function(w){return String(w||"").replace("CAR_","").replace("_","").replace("_"," to ");};
var selected=function(){return {e:$("eventSelect").value,s:$("stockSelect").value,w:$("windowSelect").value,sample:$("sampleSelect").value};};
function option(select,values,allLabel){select.innerHTML="";if(allLabel){var o=document.createElement("option");o.value="all";o.textContent=allLabel;select.appendChild(o);}values.forEach(function(v){var o=document.createElement("option");o.value=v;o.textContent=v;select.appendChild(o);});}
option($("eventSelect"),events.map(function(x){return x.event_id;}));
option($("stockSelect"),uniq(cars.map(function(x){return x.stock;})),"All stocks");
option($("windowSelect"),uniq(cars.map(function(x){return x.window;})));
var groupTypeLabel={all:"All",sample:"Sample",event_class:"Event class",sector:"Sector",stock:"Stock"};
(function(){var s=$("caarGroup");uniq(caarPath.map(function(r){return r.group_type+"|"+r.group;})).forEach(function(k){var o=document.createElement("option"),p=k.split("|");o.value=k;o.textContent=(groupTypeLabel[p[0]]||p[0])+": "+p[1];s.appendChild(o);});})();
if(!$("eventSelect").options.length||!$("stockSelect").options.length||!$("windowSelect").options.length){$("metrics").innerHTML='<div class="metric">Analysis data is incomplete. Inspect the uploaded CSV artifacts and manifest.</div>';return;}
$("definitions").textContent="How to read: Day 0 is the first trading session on which the event could affect closing prices (after-close news maps to the next session). CAR windows count trading sessions around Day 0: CAR_-1_+1 covers 3 sessions, CAR_-3_+3 covers 7, CAR_-5_+5 covers 11. The market model is fitted on "+esc(m.estimation_sessions)+" sessions that end "+esc(m.gap_sessions)+" sessions before the window starts. Returns are in %; differences between returns (excess, CAR changes) are in percentage points (pp). All significance results are exploratory.";
$("period").textContent=(m.period_start||"—")+" → "+(m.period_end||"—");
$("source").textContent=((m.input_data_provenance||{}).source||"Recorded source unavailable")+" · "+(m.benchmark||"Benchmark not recorded");
function metric(label,value,note){return '<div class="metric"><div class="label">'+esc(label)+'</div><div class="value">'+esc(value)+'</div><div class="note">'+esc(note)+'</div></div>';}
function table(headers,rows){if(!rows.length)return '<div class="empty">No rows match these filters.</div>';return '<div class="tablewrap"><table><thead><tr>'+headers.map(function(h){return '<th>'+esc(h)+'</th>';}).join("")+'</tr></thead><tbody>'+rows.map(function(r){return '<tr>'+r.map(function(v,i){return '<td>'+ (v&&typeof v==="object"&&v.__html!==undefined?v.__html:i===0?String(v):esc(v))+'</td>';}).join("")+'</tr>';}).join("")+'</tbody></table></div>';}
function signed(v){var n=num(v);return n===null?"—":(n>0?"+":"")+n.toFixed(2)+"%";}
function selectedEvent(id){return events.find(function(x){return x.event_id===id;})||{};}
function filteredRows(file){var q=selected();return file.filter(function(r){return (!q.e||r.event_id===q.e)&&(!q.s||q.s==="all"||!r.stock||r.stock===q.s)&&(!q.w||r.window===q.w);});}
function heatmap(){
var q=selected(),rows=cars.filter(function(r){return (!q.e||r.event_id===q.e)&&(!q.s||q.s==="all"||r.stock===q.s)&&(!q.w||r.window===q.w);});
if(!rows.length){$("heatmap").innerHTML='<div class="empty">No CAR rows available.</div>';return;}
var stocks=uniq(rows.map(function(r){return r.stock;})),ids=uniq(rows.map(function(r){return r.event_id;}));
var html='<div class="tablewrap"><table><thead><tr><th>Event</th>'+stocks.map(function(s){return '<th>'+esc(s)+'</th>';}).join("")+'</tr></thead><tbody>';
ids.forEach(function(id){html+='<tr><td>'+esc(id)+'</td>';stocks.forEach(function(s){var r=rows.find(function(x){return x.event_id===id&&x.stock===s;});var v=r?x100(r.CAR):null;var alpha=v===null?0:Math.min(.78,Math.abs(v)/25);var color=v===null?"#202b3d":v>=0?"rgba(40,178,135,"+alpha+")":"rgba(226,87,111,"+alpha+")";html+='<td style="background:'+color+'">'+(v===null?"—":signed(v))+'</td>';});html+='</tr>';});html+='</tbody></table></div>';$("heatmap").innerHTML=html;
}
function auditFor(id){return windowAudit.find(function(r){return r.event_id===id;})||null;}
function registryFlag(v){var t=String(v==null?"":v).toLowerCase();return t==="true"?"True":t==="false"?"False":"Unset";}
function pp(v){var n=num(v);return n===null?"—":(n>0?"+":"")+n.toFixed(2)+" pp";}
function eventDetail(){
var e=selectedEvent(selected().e),w=auditFor(e.event_id),carPartners=w?String(w.event_window_overlap_event_ids||""):"";
var url=e.source_url?'<a href="'+esc(e.source_url)+'" target="_blank" rel="noopener noreferrer">Open recorded source ↗</a>':"No source URL recorded";
$("eventDetail").innerHTML='<div class="eventcard"><div class="eventtitle">'+esc(e.headline||e.event_id)+'</div><div class="eventmeta">'+esc(e.event_id)+' · '+esc(e.category)+' · Announced '+esc(e.event_date)+' · Effective '+esc(e.effective_event_date)+'</div><p>'+esc(e.notes||"No event note recorded.")+'</p><div>'+ (w===null?'<span class="pill">CAR-window overlap not audited</span>':carPartners?'<span class="pill warn">CAR window overlaps '+esc(carPartners.split(";").join(", "))+'</span>':'<span class="pill good">No CAR-window overlap</span>')+' <span class="pill">Registry flag: '+registryFlag(e.overlap_flag)+'</span> <span class="pill">'+esc(e.timing_confidence||"Timing ungraded")+'</span> <span class="pill">'+esc(e.timestamp_evidence_status||"Timestamp evidence not graded")+'</span></div><p class="small">'+url+'</p></div>';
}
function chart(){
var q=selected(),rows=study.filter(function(r){return r.event_id===q.e&&(q.s==="all"||r.stock===q.s);});
var shownStock=q.s;if(q.s==="all"){shownStock=uniq(cars.filter(function(c){return c.event_id===q.e;}).map(function(c){return c.stock;}))[0];}
if(q.s==="all"){rows=rows.filter(function(r){return r.stock===uniq(cars.filter(function(c){return c.event_id===q.e;}).map(function(c){return c.stock;}))[0];});}
rows.sort(function(a,b){return Number(a.event_day)-Number(b.event_day);});
if(!rows.length){$("returnChart").innerHTML='<div class="empty">No event-window daily returns.</div>';return;}
var width=520,height=245,left=42,right=14,top=18,bottom=30,plotW=width-left-right,plotH=height-top-bottom;
var ars=rows.map(function(r){return x100(r.abnormal_return);}),days=rows.map(function(r){return Number(r.event_day);}),cum=[],sum=0;
ars.forEach(function(v){sum+=v||0;cum.push(sum);});
var all=ars.concat(cum).filter(function(v){return v!==null&&Number.isFinite(v);}),min=Math.min.apply(null,all.concat([0])),max=Math.max.apply(null,all.concat([0]));if(max===min){max+=1;min-=1;}
var x=function(i){return left+(i/Math.max(1,rows.length-1))*plotW;},y=function(v){return top+(max-v)/(max-min)*plotH;};
var svg='<svg class="chart" viewBox="0 0 '+width+' '+height+'" role="img" aria-label="Abnormal and cumulative abnormal returns by event day">';
for(var g=0;g<5;g++){var v=min+(max-min)*g/4,yy=y(v);svg+='<line x1="'+left+'" y1="'+yy+'" x2="'+(width-right)+'" y2="'+yy+'" stroke="#2b3950"/><text x="'+(left-7)+'" y="'+(yy+4)+'" fill="#9eacc3" font-size="10" text-anchor="end">'+v.toFixed(1)+'</text>';}
var zeroIndex=days.indexOf(0);if(zeroIndex>=0)svg+='<line x1="'+x(zeroIndex)+'" y1="'+top+'" x2="'+x(zeroIndex)+'" y2="'+(height-bottom)+'" stroke="#f4c879" stroke-dasharray="4 4"/>';
svg+='<line x1="'+left+'" y1="'+y(0)+'" x2="'+(width-right)+'" y2="'+y(0)+'" stroke="#8797b0" stroke-dasharray="2 3"/>';
ars.forEach(function(v,i){var xx=x(i),yy=y(v||0),base=y(0),barW=Math.min(18,plotW/rows.length*.48);svg+='<rect x="'+(xx-barW/2)+'" y="'+Math.min(yy,base)+'" width="'+barW+'" height="'+Math.max(1,Math.abs(base-yy))+'" rx="2" fill="#83b8ff" opacity=".82"><title>Day '+days[i]+': '+(v==null?"n/a":v.toFixed(3))+'%</title></rect>';});
svg+='<polyline fill="none" stroke="#63d9b0" stroke-width="2.5" points="'+cum.map(function(v,i){return x(i)+","+y(v);}).join(" ")+'"/>';
cum.forEach(function(v,i){svg+='<circle cx="'+x(i)+'" cy="'+y(v)+'" r="3" fill="#63d9b0"><title>Day '+days[i]+': CAR '+v.toFixed(3)+'%</title></circle>';});
days.forEach(function(v,i){svg+='<text x="'+x(i)+'" y="'+(height-9)+'" fill="#9eacc3" font-size="10" text-anchor="middle">'+v+'</text>';});
svg+='<text x="12" y="12" fill="#9eacc3" font-size="10">Return (%)</text></svg>';var expected=2*(num(m.event_window_sessions)||5)+1;
$("returnChart").innerHTML='<p class="small">Stock shown: <strong>'+esc(shownStock)+'</strong>'+(q.s==="all"?' (pick a stock above to change it)':'')+'. The line adds abnormal returns from Day '+(rows.length?days[0]:"—")+'; it equals the ±'+esc(m.event_window_sessions)+' CAR only when all '+expected+' sessions are present.</p>'+svg+(rows.length<expected?'<div class="noteBox">Only '+rows.length+' of '+expected+' event-window sessions have data. Missing sessions are not plotted and are not treated as zero.</div>':'');
}
function inference(){
var q=selected(),rows=sig.filter(function(r){return r.event_id===q.e&&(q.s==="all"||r.stock===q.s)&&r.window===q.w;});
rows.sort(function(a,b){return (num(a.holm_p_value)||1)-(num(b.holm_p_value)||1);});
var p3=function(v){return num(v)===null?"—":Number(v).toPrecision(3);},yes=function(v){return String(v).toLowerCase()==="true"?raw('<span class="pill good">Yes</span>'):"No";};
$("inference").innerHTML=table(["Event","Stock","Window","CAR","95% CI (conventional)","Raw p","Holm p","Holm < 0.05","95% CI (HAC)","HAC raw p","HAC Holm p","HAC Holm < 0.05"],rows.map(function(r){return [esc(r.event_id),esc(r.stock),esc(r.window),signed(r.CAR_pct),signed(r.ci_lower_pct)+" to "+signed(r.ci_upper_pct),p3(r.raw_p_value),p3(r.holm_p_value),yes(r.significant_after_holm_5pct),signed(r.hac_ci_lower_pct)+" to "+signed(r.hac_ci_upper_pct),p3(r.hac_raw_p_value),p3(r.hac_holm_p_value),yes(r.hac_significant_after_holm_5pct)];}))+
'<p class="small">Holm family: all '+sig.length+' complete event × stock × window CAR tests, corrected separately for the conventional and HAC p-values. HAC is a robustness check for serial correlation in the estimation window, not a fix for event-induced variance or overlapping events. All tests are exploratory.</p>';
}
function consistency(){
var q=selected(),rows=cons.filter(function(r){return r.event_id===q.e&&r.window===q.w;});
$("consistency").innerHTML=table(["Window","Mean CAR","Median CAR","Across-stock SD","Positive stocks","N"],rows.map(function(r){return [esc(r.window),signed(r.mean_CAR_pct),signed(r.median_CAR_pct),pct(r.cross_stock_std_pct),pct(x100(r.positive_stock_fraction)),esc(r.stocks_observed)];}));
}
function trading(){
var q=selected(),rows=trades.filter(function(r){return (q.sample==="all"||r.sample===q.sample)&&(!q.e||r.event_id===q.e);});
$("tradingRule").textContent="Rule: buy each configured stock at the close of the first session after Day 0, sell "+(m.holding_sessions==null?"N":m.holding_sessions)+" sessions later at the close, equal-weight stocks within each event, and subtract "+(m.transaction_cost_bps_per_side==null?"the configured":m.transaction_cost_bps_per_side)+" bps per side (one entry and one exit). Benchmark: NIFTY 50 over the same sessions.";
$("trading").innerHTML=table(["Event","Sample","Net return","Benchmark return","Excess (pp)","Stocks"],rows.map(function(r){return [esc(r.event_id),esc(r.sample),signed(x100(r.portfolio_net_return)),signed(x100(r.benchmark_return)),pp(x100(r.excess_return)),esc(r.stocks_traded)];}));
var ss=samples.filter(function(r){return q.sample==="all"||r.sample===q.sample;});
$("costs").innerHTML='<h3>Chronological sample summary</h3>'+table(["Sample","Events","Mean net return","Mean excess (pp)","Share of events beating benchmark"],ss.map(function(r){return [esc(r.sample),esc(r.number_of_events),signed(r.mean_net_return_pct),pp(r.mean_excess_return_pct),pct(r.benchmark_outperformance_rate_pct)];}));
var cc=costs.map(function(r){return [esc(r.cost_bps_per_side),signed(r.avg_net_return_pct),pp(r.avg_excess_return_pct)];});
$("costs").innerHTML+='<h3 style="margin-top:12px">Transaction-cost sensitivity</h3>'+table(["Cost / side (bps)","Mean net return (all events)","Mean excess (pp)"],cc);
}
function register(){
var rows=events.map(function(e){var w=auditFor(e.event_id),partners=w?String(w.event_window_overlap_event_ids||""):null;return ['<strong>'+esc(e.event_id)+'</strong>',esc(e.event_date),esc(e.effective_event_date),esc(e.category),esc(e.episode_id),partners===null?"Not audited":partners?raw('<span class="pill warn">'+esc(partners)+'</span>'):"None",registryFlag(e.overlap_flag),esc(e.event_time||"Not recorded"),esc(e.timestamp_evidence_status),esc(e.verification_status)];});
$("eventRegister").innerHTML=table(["ID","Announced","Effective Day 0","Category","Episode","CAR-window overlap (computed)","Registry overlap flag (manual)","Recorded time","Timestamp evidence","Recorded status"],rows)+'<p class="small">“CAR-window overlap” is computed from shared trading sessions in the ±'+esc(m.event_window_sessions)+'-session windows. The registry flag is a broader manual review flag and is not the same measure. A source URL is provenance, not proof of an exact publication time.</p>';
}
function eventDependenceAudit(){
var q=selected(),levels=d["event_level_audit.csv"]||[],pairs=d["event_pair_overlap_audit.csv"]||[];
var eventRows=levels.filter(function(r){return !q.e||r.event_id===q.e;});
var ov=ev.overlap||{},tm=ev.timing||{};
var summary='<div class="metrics">'+metric("Events with CAR-window overlap",(ov.car_window_overlap_event_ids||[]).length,(ov.car_window_overlap_pairs||0)+" overlapping event pairs (computed)")+metric("Events with volatility-window overlap",(ov.volatility_window_overlap_event_ids||[]).length,"−20:−1 / +1:+20 windows (computed)")+metric("Registry overlap flag",(ov.registry_flag_true||0)+" / "+(ov.registry_flag_false||0)+" / "+(ov.registry_flag_unset||0),"True / False / unset (manual, broader)")+metric("Timestamp review required",(tm.review_required||0)+" of "+(tm.events||0),"Events without a verified exact release time")+'</div>';
$("eventDependenceAudit").innerHTML=summary+'<h3>Selected event diagnostics</h3>'+table(["Event","CAR-window partners","Shared episode partners","Same-category partners","Potential sourced developments","Review caution","Audit interpretation"],eventRows.map(function(r){return [esc(r.event_id),esc(r.event_window_overlap_event_ids||"none"),esc(r.shared_episode_partner_ids||"none"),esc(r.same_category_partner_ids||"none"),esc(r.documented_potential_development_in_window_count||0),String(r.unverified_possibility_flag).toLowerCase()==="true"?"Review needed":"No text-based caution detected",esc(r.potential_confounder_explanation||"")];}));
var related=function(r){return (num(r.overlapping_session_count)||0)>0||String(r.shared_episode_id).toLowerCase()==="true"||String(r.same_category).toLowerCase()==="true";};
var involving=pairs.filter(function(r){return !q.e||r.event_id_a===q.e||r.event_id_b===q.e;}),pairRows=involving.filter(related);
$("eventDependenceAudit").innerHTML+='<h3 style="margin-top:14px">Pairwise event relationships</h3>'+table(["Event A","Event B","Shared sessions","Shared episode","Same category","Flags","Explanation"],pairRows.map(function(r){return [esc(r.event_id_a),esc(r.event_id_b),esc(r.overlapping_session_count),String(r.shared_episode_id).toLowerCase()==="true"?"Yes":"No",String(r.same_category).toLowerCase()==="true"?"Yes":"No",esc(r.overlap_types),esc(r.explanation)];}))+'<p class="small">'+(involving.length-pairRows.length)+' other pair(s) involving this selection share no sessions, episode or category and are not listed; they remain in event_pair_overlap_audit.csv.</p>';
}
function eventTimingSensitivity(){
var q=selected(),rows=(d["event_timing_sensitivity.csv"]||[]).filter(function(r){return (!q.e||r.event_id===q.e)&&(!q.s||q.s==="all"||r.stock===q.s)&&(!q.w||r.window===q.w);});
$("eventTimingSensitivity").innerHTML=table(["Event","Stock","Window","Timing basis","Scenario","Scenario Day 0","Recorded Day 0","Scenario CAR","Recorded CAR","Delta vs recorded"],rows.map(function(r){return [esc(r.event_id),esc(r.stock),esc(r.window),esc(r.timing_basis),esc(r.scenario),esc(r.scenario_effective_date),esc(r.recorded_effective_event_date),signed(r.scenario_CAR_pct),signed(r.recorded_CAR_pct),pp(r.delta_vs_recorded_CAR_pct)];}));
}
function estimationWindowSensitivity(){
$("estimationWindowSensitivity").innerHTML=table(["Estimation sessions","Tests","Events","Stocks","Conventional Holm significant","HAC Holm significant","Median |CAR change vs 100|","Mean |CAR change vs 100|","CAR correlation vs 100"],estimationSensitivity.map(function(r){return [esc(r.estimation_sessions),esc(r.tests_available),esc(r.unique_events),esc(r.unique_stocks),esc(r.conventional_holm_significant_tests),esc(r.hac_holm_significant_tests),pp(r.median_absolute_CAR_delta_vs_100_pct).replace("+",""),pp(r.mean_absolute_CAR_delta_vs_100_pct).replace("+",""),num(r.CAR_correlation_vs_100_session)===null?"—":Number(r.CAR_correlation_vs_100_session).toFixed(3)];}));
}
function robustnessEvidenceLedger(){
var rows=robustnessLedger.filter(function(r){return Number(r.hac_holm_significant_window_count)>0;});
$("robustnessLedger").innerHTML='<div class="metrics">'+metric("Stable across all 3 windows",robustnessLedger.filter(function(r){return String(r.hac_significant_at_all_estimation_windows).toLowerCase()==="true";}).length.toString(),"HAC + Holm corrected")+
metric("HAC-significant in at least one window",rows.length.toString(),"Includes specification-sensitive findings")+
metric("Stable findings with attribution caveat",robustnessLedger.filter(function(r){return String(r.hac_significant_at_all_estimation_windows).toLowerCase()==="true"&&String(r.review_status).indexOf("caveat")>=0;}).length.toString(),"Overlap, shared episode or timing/source caution")+'</div>'+
table(["Event","Headline","Stock","Window","HAC-significant specs","Stable all specs","CAR direction stable","Event-window overlaps","Shared episode partners","Timing confidence","Median |CAR change|","Review status"],rows.map(function(r){return [esc(r.event_id),esc(r.headline),esc(r.stock),esc(r.window),esc(r.hac_holm_significant_window_count)+"/"+esc(r.estimation_windows_tested),String(r.hac_significant_at_all_estimation_windows).toLowerCase()==="true"?"Yes":"No",String(r.car_direction_stable_across_estimation_windows).toLowerCase()==="true"?"Yes":"No",esc(r.event_window_overlap_event_ids||"None flagged"),esc(r.shared_episode_partner_ids||"None flagged"),esc(r.timing_confidence),pct(r.median_absolute_CAR_change_vs_100_pct),esc(r.review_status)];}));
}
function eventVolatility(){
var q=selected(),rows=(d["event_volatility.csv"]||[]).filter(function(r){return (!q.e||r.event_id===q.e)&&(!q.s||q.s==="all"||r.stock===q.s);});function volPct(v){var n=num(v);return n===null?"—":pct(n*100);}
$("volatilityChart").innerHTML=volatilityChart(rows);
$("eventVolatility").innerHTML=table(["Event","Stock","Pre window","Post window","Pre annualized vol","Post annualized vol","Volatility change","Post/pre ratio","Pre/Post n","Comparison status","Benchmark pre/post vol","Event-window residual SD"],rows.map(function(r){return [esc(r.event_id),esc(r.stock),esc(r.pre_event_window),esc(r.post_event_window),volPct(r.pre_event_realized_volatility_annualized),volPct(r.post_event_realized_volatility_annualized),signed(num(r.realized_volatility_percentage_change)),num(r.post_to_pre_realized_volatility_ratio)===null?"—":Number(r.post_to_pre_realized_volatility_ratio).toFixed(2),esc(r.pre_event_observations)+"/"+esc(r.post_event_observations),esc(r.volatility_comparison_status),volPct(r.benchmark_pre_event_realized_volatility_annualized)+" / "+volPct(r.benchmark_post_event_realized_volatility_annualized),volPct(r.event_window_market_model_residual_volatility_daily)];}));
}
function categoryEvidence(){
var q=selected(),cars=(d["category_car_summary.csv"]||[]).filter(function(r){return !q.w||r.window===q.w;}),vol=(d["category_volatility_summary.csv"]||[]);
$("categoryEvidence").innerHTML='<h3>CAR by category and window</h3>'+table(["Category","Window","Status","Independent episodes","Events contributing","Mean CAR","Median CAR","Overlapping events excluded"],cars.map(function(r){return [esc(r.category),esc(r.window),esc(r.status),esc(r.independent_episodes),esc(r.events_contributing),signed(r.mean_event_CAR_pct),signed(r.median_event_CAR_pct),esc(r.excluded_overlapping_events)];}));
$("categoryEvidence").innerHTML+='<h3 style="margin-top:14px">Volatility by category</h3>'+table(["Category","Metric","Status","Independent episodes","Mean","Median","Overlapping events excluded","Events in master"],vol.map(function(r){var isPct=String(r.metric||"").indexOf("_pct")>=0;return [esc(r.category),esc(r.metric),esc(r.status),esc(r.independent_episodes),isPct?pct(num(r.mean_value)):num(r.mean_value)===null?"—":Number(r.mean_value).toFixed(2),isPct?pct(num(r.median_value)):num(r.median_value)===null?"—":Number(r.median_value).toFixed(2),esc(r.excluded_overlapping_events),esc(r.category_event_count_in_master)];}));
}

function wrap(text,width){return raw('<span style="white-space:normal;display:block;text-align:left;min-width:'+(width||260)+'px">'+esc(text)+'</span>');}
function ciText(lo,hi){var a=num(lo),b=num(hi);return a===null||b===null?"No interval":signed(a)+" to "+signed(b);}
function caar(){
var key=$("caarGroup").value;
if(!key){$("caarChart").innerHTML='<div class="empty">CAAR output is not present in this analysis artifact.</div>';$("caarTable").innerHTML="";return;}
var parts=key.split("|"),rows=caarPath.filter(function(r){return r.group_type===parts[0]&&r.group===parts[1];}).sort(function(a,b){return Number(a.event_day)-Number(b.event_day);});
var vals=rows.map(function(r){return num(r.CAAR_pct);}),lo=rows.map(function(r){return num(r.CAAR_ci95_lower_pct);}),hi=rows.map(function(r){return num(r.CAAR_ci95_upper_pct);});
if(!rows.length||vals.some(function(v){return v===null;})){$("caarChart").innerHTML='<div class="empty">No complete event paths for this group.</div>';$("caarTable").innerHTML="";return;}
var band=lo.every(function(v){return v!==null;})&&hi.every(function(v){return v!==null;});
var all=vals.concat(band?lo.concat(hi):[]).concat([0]),min=Math.min.apply(null,all),max=Math.max.apply(null,all);if(max===min){max+=1;min-=1;}
var width=720,height=270,left=48,right=16,top=28,bottom=30,pw=width-left-right,ph=height-top-bottom;
var x=function(i){return left+i/Math.max(1,rows.length-1)*pw;},y=function(v){return top+(max-v)/(max-min)*ph;};
var svg='<svg class="chart" style="max-width:860px" viewBox="0 0 '+width+' '+height+'" role="img" aria-label="Cumulative average abnormal return by event day">';
for(var g=0;g<5;g++){var v=min+(max-min)*g/4;svg+='<line x1="'+left+'" y1="'+y(v)+'" x2="'+(width-right)+'" y2="'+y(v)+'" stroke="#2b3950"/><text x="'+(left-6)+'" y="'+(y(v)+4)+'" fill="#9eacc3" font-size="10" text-anchor="end">'+v.toFixed(1)+'</text>';}
if(band){svg+='<polygon fill="#83b8ff" opacity=".2" points="'+hi.map(function(v,i){return x(i)+","+y(v);}).concat(lo.map(function(v,i){return x(i)+","+y(v);}).reverse()).join(" ")+'"/>';}
svg+='<line x1="'+left+'" y1="'+y(0)+'" x2="'+(width-right)+'" y2="'+y(0)+'" stroke="#8797b0" stroke-dasharray="2 3"/>';
var z=rows.findIndex(function(r){return Number(r.event_day)===0;});if(z>=0)svg+='<line x1="'+x(z)+'" y1="'+top+'" x2="'+x(z)+'" y2="'+(height-bottom)+'" stroke="#f4c879" stroke-dasharray="4 4"/>';
svg+='<polyline fill="none" stroke="#63d9b0" stroke-width="2.5" points="'+vals.map(function(v,i){return x(i)+","+y(v);}).join(" ")+'"/>';
vals.forEach(function(v,i){svg+='<circle cx="'+x(i)+'" cy="'+y(v)+'" r="3" fill="#63d9b0"><title>Day '+rows[i].event_day+': CAAR '+v.toFixed(2)+'%'+(band?' (95% CI '+lo[i].toFixed(2)+' to '+hi[i].toFixed(2)+')':'')+'</title></circle>';});
rows.forEach(function(r,i){svg+='<text x="'+x(i)+'" y="'+(height-9)+'" fill="#9eacc3" font-size="10" text-anchor="middle">'+esc(r.event_day)+'</text>';});
svg+='<text x="12" y="12" fill="#9eacc3" font-size="10">CAAR (%)</text></svg>';
var first=rows[0];
$("caarChart").innerHTML=svg+'<p class="small">'+esc(first.events)+' events, '+esc(first.dependency_clusters)+' independent clusters, '+esc(first.stocks)+' stock(s). '+(band?'Band: 95% cluster-bootstrap interval.':'No interval shown ('+esc(first.status)+').')+'</p>';
var wr=caarWin.filter(function(r){return r.group_type===parts[0]&&r.group===parts[1];});
$("caarTable").innerHTML=table(["Window","Events","Clusters","Mean CAR","95% CI","Median CAR","Share of events positive","Status"],wr.map(function(r){return [esc(r.window),esc(r.events),esc(r.dependency_clusters),signed(r.mean_CAR_pct),ciText(r.mean_CAR_ci95_lower_pct,r.mean_CAR_ci95_upper_pct),signed(r.median_CAR_pct),pct(r.share_events_positive_pct,0),esc(r.status)];}));
}
function groupComparison(){
var q=selected(),html="";
[["event_class","Event class"],["sector","Sector"],["stock","Stock"]].forEach(function(t){
var rows=caarWin.filter(function(r){return r.group_type===t[0]&&r.window===q.w;});
html+='<h3 style="margin-top:12px">'+esc(t[1])+' · '+esc(q.w)+'</h3>'+table([t[1],"Events","Clusters","Mean CAR","95% CI","Share of events positive","Status"],rows.map(function(r){return [esc(r.group),esc(r.events),esc(r.dependency_clusters),signed(r.mean_CAR_pct),ciText(r.mean_CAR_ci95_lower_pct,r.mean_CAR_ci95_upper_pct),pct(r.share_events_positive_pct,0),esc(r.status)];}));});
$("groupComparison").innerHTML=caarWin.length?html:'<div class="empty">Comparison output is not present in this analysis artifact.</div>';
}
function windowChoice(){
if(!winSummary.length){$("windowChoice").innerHTML='<div class="empty">Window-choice output is not present in this analysis artifact.</div>';return;}
var html=table(["Window","Sessions","Tests","Conventional Holm significant","HAC Holm significant","Mean CAR","Mean |CAR|","Median standard error","Share positive"],winSummary.map(function(r){return [esc(r.window),esc(r.sessions),esc(r.tests),esc(r.conventional_holm_significant),esc(r.hac_holm_significant),signed(r.mean_CAR_pct),pct(r.mean_absolute_CAR_pct),pct(r.median_standard_error_pct),pct(r.share_positive_pct,0)];}));
html+='<h3 style="margin-top:12px">Do results agree between windows?</h3>'+table(["Narrow","Wide","Event-stock pairs","Same CAR sign","CAR correlation","HAC-significant in both","Only narrow","Only wide","What it means"],winAgree.map(function(r){return [esc(r.narrow_window),esc(r.wide_window),esc(r.event_stock_pairs),pct(r.same_sign_share_pct,0),num(r.CAR_correlation)===null?"—":Number(r.CAR_correlation).toFixed(2),esc(r.hac_significant_in_both),esc(r.hac_significant_only_narrow),esc(r.hac_significant_only_wide),wrap(r.interpretation,360)];}));
$("windowChoice").innerHTML=html+'<p class="small">Standard errors grow with window length, so a wider window needs a bigger move to look unusual. A conclusion that holds in only one window should be treated as fragile.</p>';
}
function generalizationPanel(){
var q=selected(),rows=generalization.filter(function(r){return r.window===q.w;});
if(!generalization.length){$("generalization").innerHTML='<div class="empty">Replication output is not present in this analysis artifact.</div>';return;}
$("generalization").innerHTML=table(["Test","Group","Development","Holdout","Repeated?","Detail"],rows.map(function(r){var isRate=r.test==="predict_holdout_sign_from_development_class_mean",isRank=r.test==="stock_reaction_size_rank_persistence";return [esc(r.test.replace(/_/g," ")),esc(r.group),isRate||isRank?"—":signed(r.development_value)+" (n="+esc(r.development_n)+")",isRank?"ρ = "+(num(r.statistic)===null?"—":Number(r.statistic).toFixed(2)):isRate?pct(r.holdout_value,0)+" correct (n="+esc(r.holdout_n)+")":signed(r.holdout_value)+" (n="+esc(r.holdout_n)+")",String(r.replicated).toLowerCase()==="true"?raw('<span class="pill good">Yes</span>'):raw('<span class="pill warn">No</span>'),wrap(r.interpretation,380)];}))+'<p class="small">Window: '+esc(q.w)+'. Change the CAR window above to see the other windows.</p>';
}
function headlineClassifier(){
if(!clsMetrics.length){$("headlineClassifier").innerHTML='<div class="empty">Classifier output is not present in this analysis artifact.</div>';return;}
var html=table(["Evaluation","Model","Fair blind test?","Headlines","Held-out units","Accuracy","95% CI","Macro F1"],clsMetrics.map(function(r){return [esc(String(r.evaluation).replace(/_/g," ")),esc(String(r.model).replace(/_/g," ")),String(r.fair_blind_test).toLowerCase()==="true"?"Yes":"No (rules written after reading the data)",esc(r.headlines),esc(r.held_out_units),pct(x100(r.accuracy),0),num(r.accuracy_ci95_lower)===null?"—":pct(x100(r.accuracy_ci95_lower),0)+" to "+pct(x100(r.accuracy_ci95_upper),0),num(r.macro_f1)===null?"—":Number(r.macro_f1).toFixed(2)];}));
html+='<h3 style="margin-top:12px">Per class (grouped cross-validation)</h3>'+table(["Class","Precision","Recall","F1","Headlines"],clsPerClass.map(function(r){return [esc(r.event_class),Number(r.precision).toFixed(2),Number(r.recall).toFixed(2),Number(r.f1).toFixed(2),esc(r.support)];}));
html+='<h3 style="margin-top:12px">Registry headlines: model vs hand-assigned class</h3>'+table(["Event","Headline","Assigned class","Model prediction","Model probability","Keyword rules","Agrees?"],clsEvents.map(function(r){var ok=String(r.model_agrees_with_assigned_class).toLowerCase()==="true";return [esc(r.event_id),wrap(r.headline,260),esc(r.assigned_class),esc(r.predicted_class),pct(x100(r.predicted_probability),0),esc(r.keyword_rule_class),raw(ok?'<span class="pill good">Yes</span>':'<span class="pill warn">No</span>')];}));
$("headlineClassifier").innerHTML=html+'<p class="small">Training data: results/headline_corpus.csv (real headlines with source links). Low probabilities on short, formal registry headlines show the model is unsure; disagreements are flagged for review, not auto-corrected.</p>';
}
function volatilityChart(rows){
if(!rows.length)return "";
var items=rows.map(function(r){return {label:r.stock,pre:x100(r.pre_event_realized_volatility_annualized),post:x100(r.post_event_realized_volatility_annualized)};});
var b=rows[0];items.push({label:"NIFTY 50",pre:x100(b.benchmark_pre_event_realized_volatility_annualized),post:x100(b.benchmark_post_event_realized_volatility_annualized)});
items=items.filter(function(i){return i.pre!==null&&i.post!==null;});if(!items.length)return '<div class="empty">No complete volatility windows for this selection.</div>';
var max=Math.max.apply(null,items.map(function(i){return Math.max(i.pre,i.post);}))*1.1||1;
var width=Math.max(420,items.length*92),height=230,left=44,right=10,top=14,bottom=40,pw=width-left-right,ph=height-top-bottom,slot=pw/items.length,bw=Math.min(26,slot*.32);
var y=function(v){return top+(1-v/max)*ph;};
var svg='<div class="tablewrap"><svg class="chart" style="min-width:'+Math.min(width,640)+'px" viewBox="0 0 '+width+' '+height+'" role="img" aria-label="Annualized realized volatility before and after the event">';
for(var g=0;g<=4;g++){var v=max*g/4;svg+='<line x1="'+left+'" y1="'+y(v)+'" x2="'+(width-right)+'" y2="'+y(v)+'" stroke="#2b3950"/><text x="'+(left-6)+'" y="'+(y(v)+4)+'" fill="#9eacc3" font-size="10" text-anchor="end">'+v.toFixed(0)+'</text>';}
items.forEach(function(it,i){var cx=left+slot*i+slot/2;svg+='<rect x="'+(cx-bw-2)+'" y="'+y(it.pre)+'" width="'+bw+'" height="'+(y(0)-y(it.pre))+'" fill="#83b8ff"><title>'+esc(it.label)+' before: '+it.pre.toFixed(1)+'%</title></rect><rect x="'+(cx+2)+'" y="'+y(it.post)+'" width="'+bw+'" height="'+(y(0)-y(it.post))+'" fill="#f4c879"><title>'+esc(it.label)+' after: '+it.post.toFixed(1)+'%</title></rect><text x="'+cx+'" y="'+(height-22)+'" fill="#9eacc3" font-size="10" text-anchor="middle">'+esc(String(it.label).replace("_ENTERPRISES"," ENT").replace("_PORTS"," PORTS").replace("_BANK"," BANK"))+'</text>';});
svg+='<text x="12" y="10" fill="#9eacc3" font-size="10">Annualized volatility (%)</text></svg></div><div class="legend"><span><i class="dot" style="background:#83b8ff"></i>Sessions −20 to −1</span><span><i class="dot" style="background:#f4c879"></i>Sessions +1 to +20</span></div>';
return svg;
}
function provenance(){
var p=m.input_data_provenance||{},checks=m.input_checksums_sha256||{},pc=ev.price_convention||{};
var stockField=pc.stock_field||"Not recorded",benchmarkField=pc.benchmark_field||"Not recorded";
$("provenance").innerHTML='<div class="tablewrap"><table><tbody>'+
[['Snapshot ID',m.snapshot_id],['Period',(m.period_start||"—")+" to "+(m.period_end||"—")],['Provider',p.source||p.provider],['Retrieved UTC',p.retrieved_at_utc],['Benchmark',m.benchmark],['Price input',m.input_prices],['Stock price field',stockField],['Benchmark price field',benchmarkField],['Return convention',m.return_convention||"Not recorded"],['Analysis calendar',m.analysis_calendar||"Not recorded"],['Benchmark sessions used',m.benchmark_sessions_used],['Price SHA-256',checks.prices],['Generated UTC',m.generated_utc],['Code commit',m.code_commit||"Not recorded"],['Estimation sessions',m.estimation_sessions],['Gap sessions',m.gap_sessions],['Event-window radius',m.event_window_sessions],['Event sources independently verified',m.source_independently_verified]].map(function(r){return '<tr><td>'+esc(r[0])+'</td><td style="text-align:left;white-space:normal;overflow-wrap:anywhere">'+esc(r[1])+'</td></tr>';}).join("")+
'</tbody></table></div><p class="small">'+esc(m.provider_note||"")+' The checksum proves these are the same downloaded bytes; it does not prove the provider values are correct. A later download from the provider may differ. Event sources are recorded but not independently verified by this pipeline.</p>';
}
function renderClusteredInference(){
var rows=clusteredInference;
if(!rows.length){$("clusteredInference").innerHTML='<div class="empty">Cluster-bootstrap output is not present in this analysis artifact.</div>';return;}
var rendered=rows.map(function(r){
var lower=num(r.mean_excess_return_pct_ci95_lower),upper=num(r.mean_excess_return_pct_ci95_upper);
var status=lower===null||upper===null?"not_available":lower<=0&&upper>=0?"includes_zero":"excludes_zero";
return [esc(r.sample),esc(r.events_available),esc(r.independent_dependency_clusters),esc(r.largest_cluster_events),signed(r.mean_net_return_pct_point),pp(r.mean_excess_return_pct_point),pp(lower)+" to "+pp(upper),pct(r.benchmark_outperformance_rate_pct_point)+" ["+pct(r.benchmark_outperformance_rate_pct_ci95_lower,0)+" to "+pct(r.benchmark_outperformance_rate_pct_ci95_upper,0)+"]",raw(status==="includes_zero"?'<span class="pill warn">Includes zero</span>':status==="excludes_zero"?'<span class="pill good">Excludes zero</span>':'<span class="pill">Not available</span>')];
});
$("clusteredInference").innerHTML=table(["Sample","Events","Dependency clusters","Largest cluster","Mean net return","Mean excess (pp)","95% CI: mean excess","Share beating benchmark [95% CI]","Interval vs zero"],rendered)+
'<p class="small">Bootstrap replicates: '+esc(rows[0].bootstrap_replicates||"—")+'. Cluster counts are small, so interval estimates remain exploratory. These metrics are not evidence of causality or a deployable trading strategy.</p>';
}
function render(){
var q=selected(),eventRows=study.filter(function(r){return r.event_id===q.e;}),testRows=sig.filter(function(r){return !q.e||r.event_id===q.e;});
var significant=testRows.filter(function(r){return String(r.significant_after_holm_5pct).toLowerCase()==="true";}).length;
var obs=study.length,carCount=cars.filter(function(r){return num(r.CAR)!==null;}).length;
$("metrics").innerHTML=metric("Historical events",m.event_count||events.length,"Selected event master")+
metric("Stock-event-day rows",obs.toLocaleString(),"Actual, expected and abnormal returns")+
metric("Complete CAR estimates",carCount.toLocaleString(),"Of "+cars.length+" event × stock × window rows")+
metric("Holm-significant in selected event",significant.toString(),"Of "+testRows.length+" tests in current event filter");
heatmap();eventDetail();chart();caar();groupComparison();windowChoice();generalizationPanel();headlineClassifier();inference();estimationWindowSensitivity();robustnessEvidenceLedger();consistency();trading();renderClusteredInference();register();eventDependenceAudit();eventTimingSensitivity();eventVolatility();categoryEvidence();provenance();
}
["eventSelect","stockSelect","windowSelect","sampleSelect","caarGroup"].forEach(function(id){$(id).addEventListener("change",render);});
render();
})();
</script></main></body></html>'''
    html = (
        template.replace("__TITLE__", title)
        .replace("__SNAPSHOT__", html_lib.escape(str(manifest.get("snapshot_id", "unavailable"))))
        .replace("__GENERATED__", html_lib.escape(_utc_label(manifest.get("generated_utc"))))
        .replace("__COMMIT__", html_lib.escape(str(manifest.get("code_commit") or "not recorded")[:12]))
        .replace("__CONVENTION_STATUS__", html_lib.escape(evidence["price_convention"]["status"]))
        .replace("__CONVENTION_NOTE__", html_lib.escape(evidence["price_convention"]["note"]))
        .replace("__HOLDOUT_WARNING__", _holdout_warning_html(evidence["strategy_intervals"]))
        .replace("__PAYLOAD__", payload)
    )
    destination = Path(output_path) if output_path is not None else root / "eventecho_demo.html"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html, encoding="utf-8")
    return destination

def write_artifacts(artifacts: dict[str, pd.DataFrame], output_dir: Path, manifest: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    previous_manifest = output_dir / "manifest.json"
    if previous_manifest.is_file():
        try:
            old = json.loads(previous_manifest.read_text(encoding="utf-8"))
            current_names = set(artifacts)
            for record in old.get("artifacts", []):
                name = record.get("file")
                if name and name not in current_names:
                    stale = (output_dir / name).resolve()
                    if stale.parent == output_dir.resolve() and stale.is_file():
                        stale.unlink()
        except (OSError, ValueError, TypeError):
            # Preserve old files if a previous manifest cannot be trusted.
            pass
    records = []
    for filename, dataframe in artifacts.items():
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(f"Artifact {filename} must be a DataFrame")
        if dataframe.empty:
            raise ValueError(f"Refusing to write empty result artifact: {filename}")
        dataframe.to_csv(output_dir / filename, index=False)
        records.append({
            "file": filename,
            "description": ARTIFACT_DESCRIPTIONS.get(filename, "Generated analysis output."),
            "rows": len(dataframe),
            "columns": len(dataframe.columns),
        })
    payload = {
        "project": "EventEcho",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": "Generated by scripts/run_analysis.py from configured input data",
        "artifacts": records,
        **manifest,
    }
    (output_dir / "manifest.json").write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return output_dir / "manifest.json"