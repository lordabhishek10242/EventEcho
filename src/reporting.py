"""Stable result schemas, artifact writing, and provenance manifest."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ARTIFACT_DESCRIPTIONS = {
    "event_master.csv": "Event identifiers, source metadata, original and effective dates.",
    "multi_stock_event_study.csv": "Per-event, per-stock daily returns, expected returns, and abnormal returns.",
    "multi_stock_car.csv": "Complete cumulative abnormal returns for each event, stock, and window.",
    "cross_stock_consistency.csv": "Cross-stock CAR summaries by event and window.",
    "trade_candidates.csv": "Per-event and per-stock simulated entries, exits, costs, and benchmark returns.",
    "event_trading_results.csv": "Equal-weighted event portfolio returns and benchmark excess returns.",
    "sample_performance.csv": "Event-weighted development and holdout performance summaries.",
    "transaction_cost_sensitivity.csv": "Strategy summary metrics at configured per-side cost levels.",
    "final_strategy_assessment.csv": "Selected headline trading metrics at the configured base cost.",
    "multi_stock_significance.csv": "Approximate CAR inference with Holm-adjusted p-values.",
    "single_stock_event_study.csv": "Reliance subset of the generated multi-stock daily event-study output.",
    "covid_extreme_day_sensitivity.csv": "EVT002 CAR sensitivity after excluding high absolute market-return sessions.",
}



def build_demo_dashboard(analysis_dir: str | Path, output_path: str | Path | None = None) -> Path:
    """Build a self-contained, offline HTML report from verified historical analysis."""
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
        "multi_stock_significance.csv", "event_volatility.csv", "event_window_audit.csv",
        "event_level_audit.csv", "event_pair_overlap_audit.csv",
        "documented_confounder_audit.csv", "trade_candidates.csv",
        "event_trading_results.csv", "sample_performance.csv",
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

    payload = json.dumps({"manifest": manifest, "datasets": datasets}, default=str)
    payload = payload.replace("</", "<\\/")
    title = "EventEcho — Frozen Historical Event Study"
    html = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{title}</title><style>"
        "body{font:15px system-ui;max-width:1400px;margin:24px auto;padding:0 16px;background:#101827;color:#e7edf7}"
        "section,header{background:#192538;border:1px solid #34445b;border-radius:12px;padding:16px;margin:14px 0}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere;color:#c7d7eb}h1,h2{margin-top:0}.tag{color:#b7d3ff}"
        "</style></head><body><header><p class='tag'>HISTORICAL DATA SNAPSHOT</p>"
        f"<h1>{title}</h1><p>Abnormal returns, CAR, volatility, uncertainty and confounder diagnostics. "
        "Associations only; not causal claims or investment advice.</p>"
        f"<p>Snapshot: {manifest.get('snapshot_id')} | Period: {manifest.get('period_start')} to {manifest.get('period_end')} "
        f"| Events: {manifest.get('event_count')}</p></header>"
        "<section><h2>Event selector</h2><label>Event<select id='eventSelect'></select></label> "
        "<label>Stock<select id='stockSelect'></select></label> "
        "<label>CAR window<select id='windowSelect'></select></label></section>"
        "<section><h2>CAR heatmap</h2><div id='carHeatmap'></div></section>"
        "<section><h2>Event timeline</h2><div id='timeline'></div></section>"
        "<section><h2>Abnormal returns / cumulative CAR</h2><div id='cumulativeCar'></div></section>"
        "<section><h2>Volatility</h2><div id='volatility'></div></section>"
        "<section><h2>Hypothetical trading</h2><div id='trading'></div></section>"
        "<section><h2>Uncertainty and multiple testing</h2><div id='uncertainty'></div></section>"
        "<section><h2>Confounders and event overlap</h2><div id='confounders'></div></section>"
        "<section><h2>Embedded reproducibility data</h2><pre id='provenance'></pre>"
        f"<script id='eventecho-data' type='application/json'>{payload}</script></section>"
        "<script>const a=JSON.parse(document.getElementById('eventecho-data').textContent),d=a.datasets,m=a.manifest;"
        "const names=['eventSelect','stockSelect','windowSelect'];"
        "const vals=[...new Set((d['event_master.csv']||[]).map(x=>x.event_id).filter(Boolean))];"
        "document.getElementById(names[0]).innerHTML=vals.map(x=>'<option>'+x+'</option>').join('');"
        "document.getElementById(names[1]).innerHTML=[...new Set((d['multi_stock_car.csv']||[]).map(x=>x.stock).filter(Boolean))].map(x=>'<option>'+x+'</option>').join('');"
        "document.getElementById(names[2]).innerHTML=[...new Set((d['multi_stock_car.csv']||[]).map(x=>x.window).filter(Boolean))].map(x=>'<option>'+x+'</option>').join('');"
        "document.getElementById('provenance').textContent=JSON.stringify({snapshot:m.snapshot_id,prices:m.input_prices,checksums:m.input_checksums_sha256,source:m.input_data_provenance},null,2);"
        "function render(){const e=document.getElementById('eventSelect').value,s=document.getElementById('stockSelect').value,w=document.getElementById('windowSelect').value;"
        "for(const [id,file] of [['carHeatmap','multi_stock_car.csv'],['timeline','event_master.csv'],['cumulativeCar','multi_stock_event_study.csv'],['volatility','event_volatility.csv'],['trading','event_trading_results.csv'],['uncertainty','multi_stock_significance.csv']]){"
        "const rs=(d[file]||[]).filter(r=>(!e||r.event_id===e)&&(!s||!r.stock||r.stock===s)&&(!w||!r.window||r.window===w));"
        "document.getElementById(id).textContent=JSON.stringify(rs,null,2);}"
        "document.getElementById('confounders').textContent=JSON.stringify(['event_level_audit.csv','event_window_audit.csv','event_pair_overlap_audit.csv','documented_confounder_audit.csv'].flatMap(k=>d[k]||[]).filter(r=>!e||r.event_id===e||r.event_id_a===e||r.event_id_b===e),null,2);}"
        "names.forEach(id=>document.getElementById(id).addEventListener('change',render));render();</script>"
        "</body></html>"
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