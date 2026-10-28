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
    """Build a self-contained, offline HTML dashboard from frozen analysis outputs."""
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

    datasets: dict[str, list[dict]] = {}
    for filename in (
        "event_master.csv", "multi_stock_event_study.csv", "multi_stock_car.csv",
        "multi_stock_significance.csv", "event_volatility.csv", "event_window_audit.csv",
        "event_level_audit.csv", "event_pair_overlap_audit.csv",
        "documented_confounder_audit.csv", "trade_candidates.csv",
        "event_trading_results.csv", "sample_performance.csv",
    ):
        file_path = root / filename
        if file_path.is_file():
            try:
                datasets[filename] = pd.read_csv(file_path).where(pd.notna(pd.read_csv(file_path)), None).to_dict(orient="records")
            except pd.errors.EmptyDataError:
                datasets[filename] = []

    # Embed JSON safely for a portable, no-network dashboard.
    payload = json.dumps({"manifest": manifest, "datasets": datasets}, default=str)
    payload = payload.replace("</", "<\\/")
    document = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EventEcho — Frozen Historical Event Study</title>
<style>
:root{color-scheme:dark;font-family:Inter,system-ui,sans-serif;background:#101827;color:#e7edf7}
body{margin:0;padding:24px;max-width:1400px;margin-inline:auto}header,.panel{background:#192538;border:1px solid #34445b;border-radius:14px;padding:18px;margin-bottom:16px}
h1,h2{margin-top:0}p,.muted{color:#a9b8cc}.controls{display:flex;gap:12px;flex-wrap:wrap}label{display:grid;gap:6px;color:#a9b8cc}
select{background:#101827;color:#e7edf7;border:1px solid #50617a;border-radius:8px;padding:9px;min-width:150px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}.metric{background:#101827;border-radius:10px;padding:14px}
table{width:100%;border-collapse:collapse;font-size:13px;display:block;overflow:auto}th,td{text-align:left;padding:8px;border-bottom:1px solid #34445b;white-space:nowrap}
.tag{display:inline-block;border:1px solid #58708f;border-radius:99px;padding:3px 8px;color:#b7d3ff}
</style></head><body>
<header><span class="tag">HISTORICAL DATA SNAPSHOT</span><h1>EventEcho</h1>
<p>Evidence-oriented event study: abnormal returns, cumulative abnormal returns (CAR), volatility, overlap and confounder diagnostics. Results are associations, not causal claims or investment advice.</p>
<div class="grid" id="summary"></div><p class="muted" id="provenance"></p></header>
<section class="panel"><h2>Explore results</h2><div class="controls">
<label>Event<select id="eventSelect"></select></label><label>Stock<select id="stockSelect"></select></label><label>CAR window<select id="windowSelect"></select></label></div></section>
<section class="panel"><h2>CAR heatmap data</h2><div id="carHeatmap"></div></section>
<section class="panel"><h2>Event timeline</h2><div id="timeline"></div></section>
<section class="panel"><h2>Daily abnormal returns</h2><div id="cumulativeCar"></div></section>
<section class="panel"><h2>Volatility before / after</h2><div id="volatility"></div></section>
<section class="panel"><h2>Hypothetical trading simulation</h2><div id="trading"></div></section>
<section class="panel"><h2>Uncertainty and multiple testing</h2><div id="uncertainty"></div></section>
<section class="panel"><h2>Overlap and confounder audit</h2><div id="confounders"></div></section>
<script id="eventecho-data" type="application/json">""" + payload + """</script>
<script>
const app=JSON.parse(document.getElementById('eventecho-data').textContent), d=app.datasets, m=app.manifest;
const rows=(name)=>d[name]||[], esc=(x)=>String(x??'—').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function table(data){if(!data.length)return '<p class="muted">No artifact available for this section.</p>';const cols=Object.keys(data[0]);return '<table><thead><tr>'+cols.map(c=>'<th>'+esc(c)+'</th>').join('')+'</tr></thead><tbody>'+data.slice(0,300).map(r=>'<tr>'+cols.map(c=>'<td>'+esc(r[c])+'</td>').join('')+'</tr>').join('')+'</tbody></table>'}
function select(id,vals){const e=document.getElementById(id);e.innerHTML=vals.map(v=>'<option value="'+esc(v)+'">'+esc(v)+'</option>').join('')}
const events=rows('event_master.csv'), cars=rows('multi_stock_car.csv'), daily=rows('multi_stock_event_study.csv');
select('eventSelect',[...new Set(events.map(x=>x.event_id).filter(Boolean))]);select('stockSelect',[...new Set(cars.map(x=>x.stock).filter(Boolean))]);select('windowSelect',[...new Set(cars.map(x=>x.window).filter(Boolean))]);
document.getElementById('summary').innerHTML=[['Snapshot',m.snapshot_id],['Period',(m.period_start||'')+' → '+(m.period_end||'')],['Events',m.event_count],['Stocks',(m.stock_universe||[]).join(', ')],['Benchmark',m.benchmark||'NIFTY50']].map(([k,v])=>'<div class="metric"><p>'+esc(k)+'</p><strong>'+esc(v)+'</strong></div>').join('');
document.getElementById('provenance').textContent='Input: '+(m.input_prices||'snapshot prices')+' | SHA-256: '+JSON.stringify(m.input_checksums_sha256||{})+' | Source independently verified: '+String((m.input_data_provenance||{}).source_independently_verified??false);
function render(){const ev=document.getElementById('eventSelect').value, st=document.getElementById('stockSelect').value, win=document.getElementById('windowSelect').value;
document.getElementById('carHeatmap').innerHTML=table(cars.filter(r=>(!ev||r.event_id===ev)&&(!st||r.stock===st)&&(!win||r.window===win)));
document.getElementById('timeline').innerHTML=table(events.filter(r=>!ev||r.event_id===ev));
document.getElementById('cumulativeCar').innerHTML=table(daily.filter(r=>(!ev||r.event_id===ev)&&(!st||r.stock===st)));
document.getElementById('volatility').innerHTML=table(rows('event_volatility.csv').filter(r=>(!ev||r.event_id===ev)&&(!st||r.stock===st)));
document.getElementById('trading').innerHTML=table(rows('event_trading_results.csv').filter(r=>!ev||r.event_id===ev));
document.getElementById('uncertainty').innerHTML=table(rows('multi_stock_significance.csv').filter(r=>(!ev||r.event_id===ev)&&(!st||r.stock===st)&&(!win||r.window===win)));
document.getElementById('confounders').innerHTML=table([...rows('event_level_audit.csv'),...rows('event_window_audit.csv'),...rows('event_pair_overlap_audit.csv'),...rows('documented_confounder_audit.csv')].filter(r=>!ev||r.event_id===ev||r.event_id_a===ev||r.event_id_b===ev));
}
['eventSelect','stockSelect','windowSelect'].forEach(id=>document.getElementById(id).addEventListener('change',render));render();
</script></body></html>""";
    const destination = output_path ? Path(output_path) : root / "eventecho_demo.html";
    destination.parent.mkdir(parents=True, exist_ok=True);
    destination.write_text(document, encoding="utf-8");
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