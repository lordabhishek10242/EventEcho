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
}



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
<header><div><div class="eyebrow">Quantitative research workspace / India equities</div><h1>EventEcho</h1><div class="sub">Explore stock reactions around selected policy and macroeconomic events. Compare abnormal returns against a market-model expectation, inspect uncertainty, and audit event timing and overlap. This is an evidence explorer, not a prediction engine.</div></div><div class="snapshot"><strong>FROZEN HISTORICAL SNAPSHOT</strong><span id="snapshotId">__SNAPSHOT__</span><br><span id="period"></span><br><span id="source"></span></div></header>
<div class="metrics" id="metrics"></div>
<section class="panel controls"><div class="field"><label for="eventSelect">Event</label><select id="eventSelect"></select></div><div class="field"><label for="stockSelect">Stock</label><select id="stockSelect"></select></div><div class="field"><label for="windowSelect">CAR window</label><select id="windowSelect"></select></div><div class="field"><label for="sampleSelect">Trading sample</label><select id="sampleSelect"><option value="all">All events</option><option value="Development">Development</option><option value="Holdout">Holdout</option></select></div></section>
<div class="layout">
<section class="panel"><h2>CAR heatmap</h2><p class="small">Cumulative abnormal return by event and stock. Colors encode direction and magnitude; values are percentages.</p><div id="heatmap"></div></section>
<section class="panel"><h2>Selected event</h2><div id="eventDetail"></div><h3 style="margin-top:16px">Return response around Day 0</h3><div id="returnChart"></div><div class="legend"><span><i class="dot" style="background:#83b8ff"></i>Daily abnormal return</span><span><i class="dot" style="background:#63d9b0"></i>Cumulative abnormal return</span></div></section>
<section class="panel wide"><h2>Inference & uncertainty</h2><p class="small">Confidence intervals and Holm-adjusted p-values are approximate under the current model assumptions. Statistical significance is not causality.</p><div id="inference"></div></section>\n<section class="panel wide"><h2>Estimation-window robustness</h2><p class="small">The market model is re-fit with 60, 100, and 150 pre-event sessions while keeping the 20-session gap and event windows fixed. Holm correction is performed separately within each specification. Treat changes as sensitivity, not a search for the most significant result.</p><div id="estimationWindowSensitivity"></div></section>\n<section class="panel wide"><h2>Robustness evidence ledger</h2><p class="small">Lists findings that are HAC-Holm significant in at least one estimation-window specification. “Stable” means significant in all three tested lengths; event overlap, shared episodes, and timing/source cautions are shown separately. Stable significance does not establish causality.</p><div id="robustnessLedger"></div></section>
<section class="panel"><h2>Across-stock consistency</h2><p class="small">Do stocks move in the same direction around this event?</p><div id="consistency"></div></section>
<section class="panel"><h2>Hypothetical trading diagnostic</h2><p class="small">Exploratory close-to-close rule; not an investable backtest.</p><div id="trading"></div><div id="costs" style="margin-top:12px"></div></section>
<section class="panel wide"><h2>Event register</h2><div id="eventRegister"></div></section>\n<section class="panel wide"><h2>Event dependence & independence audit</h2><p class="small">Temporal overlap is calculated on trading sessions. Shared episode IDs and categories are metadata-based clustering warnings—not evidence of causality. Pending source/timing checks remain review cautions.</p><div id="eventDependenceAudit"></div></section>\n<section class="panel wide"><h2>Event timing sensitivity</h2><p class="small">For unresolved timing, compare plausible session alignments. If the context indicates an after-market/evening release, the first scenario is strictly the next available session—never the same-day close that preceded publication. For other uncertain timestamps, compare the first session on/after the date with the following session. This is a robustness diagnostic, not an automatic date correction.</p><div id="eventTimingSensitivity"></div></section>\n<section class="panel wide"><h2>Realized volatility before vs after events</h2><p class="small">Realized stock and benchmark volatility use non-overlapping trading-session windows −20:−1 and +1:+20, excluding Day 0. Annualization uses √252. Event-window market-model residual volatility is shown separately and is not the same as realized stock volatility.</p><div id="eventVolatility"></div></section>\n<section class="panel wide"><h2>Category-level evidence</h2><p class="small">Category summaries average stocks within each event and then aggregate within episodes. Overlapping event windows are excluded. A category needs at least three independent episodes to receive a numeric summary; “insufficient independent episodes” is an honest data limitation, not a zero effect.</p><div id="categoryEvidence"></div></section>
<section class="panel wide"><h2>Data provenance & interpretation limits</h2><div id="provenance"></div><div class="noteBox">Daily data cannot isolate intraday reactions to market-hours announcements. The sample contains only a small, hand-selected set of events; overlapping news and shared macro episodes weaken independence. Do not interpret these associations as causal effects or future returns.</div></section>
</div><div class="footer">EventEcho research prototype · Data provider and timestamps are recorded in the manifest. Confirm event timing and primary sources before presenting results. No investment advice.</div>
<script id="eventecho-data" type="application/json">__PAYLOAD__</script>
<script>
(function(){
"use strict";
var root=JSON.parse(document.getElementById("eventecho-data").textContent),d=root.datasets||{},m=root.manifest||{};
var events=d["event_master.csv"]||[],cars=d["multi_stock_car.csv"]||[],study=d["multi_stock_event_study.csv"]||[],sig=d["multi_stock_significance.csv"]||[],cons=d["cross_stock_consistency.csv"]||[],trades=d["event_trading_results.csv"]||[],samples=d["sample_performance.csv"]||[],costs=d["transaction_cost_sensitivity.csv"]||[],covid=d["covid_extreme_day_sensitivity.csv"]||[],estimationSensitivity=d["estimation_window_sensitivity_summary.csv"]||[],robustnessLedger=d["robustness_evidence_ledger.csv"]||[];
var $=function(id){return document.getElementById(id);};
var esc=function(v){return String(v==null?"—":v).replace(/[&<>"']/g,function(c){return {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c];});};
var num=function(v){var n=Number(v);return Number.isFinite(n)?n:null;};
var pct=function(v,digits){var n=num(v);return n===null?"—":n.toFixed(digits==null?2:digits)+"%";};
var uniq=function(a){return Array.from(new Set(a.filter(function(x){return x!==null&&x!==undefined&&x!=="";})));};
var labelWindow=function(w){return String(w||"").replace("CAR_","").replace("_","").replace("_"," to ");};
var selected=function(){return {e:$("eventSelect").value,s:$("stockSelect").value,w:$("windowSelect").value,sample:$("sampleSelect").value};};
function option(select,values,allLabel){select.innerHTML="";if(allLabel){var o=document.createElement("option");o.value="all";o.textContent=allLabel;select.appendChild(o);}values.forEach(function(v){var o=document.createElement("option");o.value=v;o.textContent=v;select.appendChild(o);});}
option($("eventSelect"),events.map(function(x){return x.event_id;}));
option($("stockSelect"),uniq(cars.map(function(x){return x.stock;})),"All stocks");
option($("windowSelect"),uniq(cars.map(function(x){return x.window;})));
if(!$("eventSelect").options.length||!$("stockSelect").options.length||!$("windowSelect").options.length){$("metrics").innerHTML='<div class="metric">Analysis data is incomplete. Inspect the uploaded CSV artifacts and manifest.</div>';return;}
$("period").textContent=(m.period_start||"—")+" → "+(m.period_end||"—");
$("source").textContent=((m.input_data_provenance||{}).source||"Recorded source unavailable")+" · "+(m.benchmark||"Benchmark not recorded");
function metric(label,value,note){return '<div class="metric"><div class="label">'+esc(label)+'</div><div class="value">'+esc(value)+'</div><div class="note">'+esc(note)+'</div></div>';}
function table(headers,rows){if(!rows.length)return '<div class="empty">No rows match these filters.</div>';return '<div class="tablewrap"><table><thead><tr>'+headers.map(function(h){return '<th>'+esc(h)+'</th>';}).join("")+'</tr></thead><tbody>'+rows.map(function(r){return '<tr>'+r.map(function(v,i){return '<td>'+ (i===0?String(v):esc(v))+'</td>';}).join("")+'</tr>';}).join("")+'</tbody></table></div>';}
function signed(v){var n=num(v);return n===null?"—":(n>0?"+":"")+n.toFixed(2)+"%";}
function selectedEvent(id){return events.find(function(x){return x.event_id===id;})||{};}
function filteredRows(file){var q=selected();return file.filter(function(r){return (!q.e||r.event_id===q.e)&&(!q.s||q.s==="all"||!r.stock||r.stock===q.s)&&(!q.w||r.window===q.w);});}
function heatmap(){
var q=selected(),rows=cars.filter(function(r){return (!q.e||r.event_id===q.e)&&(!q.s||q.s==="all"||r.stock===q.s)&&(!q.w||r.window===q.w);});
if(!rows.length){$("heatmap").innerHTML='<div class="empty">No CAR rows available.</div>';return;}
var stocks=uniq(rows.map(function(r){return r.stock;})),ids=uniq(rows.map(function(r){return r.event_id;}));
var html='<div class="tablewrap"><table><thead><tr><th>Event</th>'+stocks.map(function(s){return '<th>'+esc(s)+'</th>';}).join("")+'</tr></thead><tbody>';
ids.forEach(function(id){html+='<tr><td>'+esc(id)+'</td>';stocks.forEach(function(s){var r=rows.find(function(x){return x.event_id===id&&x.stock===s;});var v=r?num(r.CAR)*100:null;var alpha=v===null?0:Math.min(.78,Math.abs(v)/25);var color=v===null?"#202b3d":v>=0?"rgba(40,178,135,"+alpha+")":"rgba(226,87,111,"+alpha+")";html+='<td style="background:'+color+'">'+(v===null?"—":signed(v))+'</td>';});html+='</tr>';});html+='</tbody></table></div>';$("heatmap").innerHTML=html;
}
function eventDetail(){
var e=selectedEvent(selected().e),overlap=String(e.overlap_flag).toLowerCase()==="true";
var url=e.source_url?'<a href="'+esc(e.source_url)+'" target="_blank" rel="noopener noreferrer">Open recorded source ↗</a>':"No source URL recorded";
$("eventDetail").innerHTML='<div class="eventcard"><div class="eventtitle">'+esc(e.headline||e.event_id)+'</div><div class="eventmeta">'+esc(e.event_id)+' · '+esc(e.category)+' · Announced '+esc(e.event_date)+' · Effective '+esc(e.effective_event_date)+'</div><p>'+esc(e.notes||"No event note recorded.")+'</p><div>'+ (overlap?'<span class="pill warn">Overlap flagged</span>':'<span class="pill">No overlap flag recorded</span>')+' <span class="pill">'+esc(e.timing_confidence||"Timing ungraded")+'</span></div><p class="small">'+url+'</p></div>';
}
function chart(){
var q=selected(),rows=study.filter(function(r){return r.event_id===q.e&&(q.s==="all"||r.stock===q.s);});
if(q.s==="all"){rows=rows.filter(function(r){return r.stock===uniq(cars.filter(function(c){return c.event_id===q.e;}).map(function(c){return c.stock;}))[0];});}
rows.sort(function(a,b){return Number(a.event_day)-Number(b.event_day);});
if(!rows.length){$("returnChart").innerHTML='<div class="empty">No event-window daily returns.</div>';return;}
var width=520,height=245,left=42,right=14,top=18,bottom=30,plotW=width-left-right,plotH=height-top-bottom;
var ars=rows.map(function(r){return num(r.abnormal_return)*100;}),days=rows.map(function(r){return Number(r.event_day);}),cum=[],sum=0;
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
svg+='<text x="12" y="12" fill="#9eacc3" font-size="10">Return (%)</text></svg>';$("returnChart").innerHTML=svg;
}
function inference(){
var q=selected(),rows=sig.filter(function(r){return r.event_id===q.e&&(q.s==="all"||r.stock===q.s)&&r.window===q.w;});
rows.sort(function(a,b){return (num(a.holm_p_value)||1)-(num(b.holm_p_value)||1);});
$("inference").innerHTML=table(["Event","Stock","Window","CAR","95% CI","Raw p","Holm p","Holm < 0.05"],rows.map(function(r){return [esc(r.event_id),esc(r.stock),esc(r.window),signed(r.CAR_pct),signed(r.ci_lower_pct)+" to "+signed(r.ci_upper_pct),num(r.raw_p_value)===null?"—":Number(r.raw_p_value).toPrecision(3),num(r.holm_p_value)===null?"—":Number(r.holm_p_value).toPrecision(3),String(r.significant_after_holm_5pct).toLowerCase()==="true"?'<span class="pill good">Yes</span>':'No'];}));
}
function consistency(){
var q=selected(),rows=cons.filter(function(r){return r.event_id===q.e&&r.window===q.w;});
$("consistency").innerHTML=table(["Window","Mean CAR","Median CAR","Across-stock SD","Positive stocks","N"],rows.map(function(r){return [esc(r.window),signed(r.mean_CAR_pct),signed(r.median_CAR_pct),pct(r.cross_stock_std_pct),pct(num(r.positive_stock_fraction)*100),esc(r.stocks_observed)];}));
}
function trading(){
var q=selected(),rows=trades.filter(function(r){return (q.sample==="all"||r.sample===q.sample)&&(!q.e||r.event_id===q.e);});
$("trading").innerHTML=table(["Event","Sample","Net return","Benchmark","Excess","Stocks"],rows.map(function(r){return [esc(r.event_id),esc(r.sample),signed(num(r.portfolio_net_return)*100),signed(num(r.benchmark_return)*100),signed(num(r.excess_return)*100),esc(r.stocks_traded)];}));
var ss=samples.filter(function(r){return q.sample==="all"||r.sample===q.sample;});
$("costs").innerHTML='<h3>Chronological sample summary</h3>'+table(["Sample","Events","Mean net","Mean excess","Benchmark beat rate"],ss.map(function(r){return [esc(r.sample),esc(r.number_of_events),signed(r.mean_net_return_pct),signed(r.mean_excess_return_pct),pct(r.benchmark_outperformance_rate_pct)];}));
var cc=costs.map(function(r){return [esc(r.cost_bps_per_side),signed(r.avg_net_return_pct),signed(r.avg_excess_return_pct)];});
$("costs").innerHTML+='<h3 style="margin-top:12px">Transaction-cost sensitivity</h3>'+table(["Cost / side (bps)","Mean net return","Mean excess return"],cc);
}
function register(){
var rows=events.map(function(e){var flag=String(e.overlap_flag).toLowerCase()==="true";return ['<strong>'+esc(e.event_id)+'</strong>',esc(e.event_date),esc(e.effective_event_date),esc(e.category),esc(e.episode_id),flag?'<span class="pill warn">Flagged</span>':'No flag',esc(e.verification_status)];});
$("eventRegister").innerHTML=table(["ID","Announced","Effective","Category","Episode","Overlap","Recorded status"],rows);
}
function eventDependenceAudit(){
var q=selected(),levels=d["event_level_audit.csv"]||[],pairs=d["event_pair_overlap_audit.csv"]||[];
var eventRows=levels.filter(function(r){return !q.e||r.event_id===q.e;});
$("eventDependenceAudit").innerHTML='<h3>Selected event diagnostics</h3>'+table(["Event","CAR-window partners","Shared episode partners","Same-category partners","Potential sourced developments","Review caution","Audit interpretation"],eventRows.map(function(r){return [esc(r.event_id),esc(r.event_window_overlap_event_ids||"none"),esc(r.shared_episode_partner_ids||"none"),esc(r.same_category_partner_ids||"none"),esc(r.documented_potential_development_in_window_count||0),String(r.unverified_possibility_flag).toLowerCase()==="true"?"Review needed":"No text-based caution detected",esc(r.potential_confounder_explanation||"")];}));
var pairRows=pairs.filter(function(r){return !q.e||r.event_id_a===q.e||r.event_id_b===q.e;});
$("eventDependenceAudit").innerHTML+='<h3 style="margin-top:14px">Pairwise event relationships</h3>'+table(["Event A","Event B","Shared sessions","Shared episode","Same category","Flags","Explanation"],pairRows.map(function(r){return [esc(r.event_id_a),esc(r.event_id_b),esc(r.overlapping_session_count),String(r.shared_episode_id).toLowerCase()==="true"?"Yes":"No",String(r.same_category).toLowerCase()==="true"?"Yes":"No",esc(r.overlap_types),esc(r.explanation)];}));
}
function eventTimingSensitivity(){
var q=selected(),rows=(d["event_timing_sensitivity.csv"]||[]).filter(function(r){return (!q.e||r.event_id===q.e)&&(!q.s||q.s==="all"||r.stock===q.s)&&(!q.w||r.window===q.w);});
$("eventTimingSensitivity").innerHTML=table(["Event","Stock","Window","Timing basis","Scenario","Scenario Day 0","Recorded Day 0","Scenario CAR","Recorded CAR","Delta vs recorded"],rows.map(function(r){return [esc(r.event_id),esc(r.stock),esc(r.window),esc(r.timing_basis),esc(r.scenario),esc(r.scenario_effective_date),esc(r.recorded_effective_event_date),signed(r.scenario_CAR_pct),signed(r.recorded_CAR_pct),signed(r.delta_vs_recorded_CAR_pct)];}));
}
function estimationWindowSensitivity(){
$("estimationWindowSensitivity").innerHTML=table(["Estimation sessions","Tests","Events","Stocks","Conventional Holm significant","HAC Holm significant","Median |CAR change vs 100|","Mean |CAR change vs 100|","CAR correlation vs 100"],estimationSensitivity.map(function(r){return [esc(r.estimation_sessions),esc(r.tests_available),esc(r.unique_events),esc(r.unique_stocks),esc(r.conventional_holm_significant_tests),esc(r.hac_holm_significant_tests),pct(r.median_absolute_CAR_delta_vs_100_pct),pct(r.mean_absolute_CAR_delta_vs_100_pct),num(r.CAR_correlation_vs_100_session)===null?"—":Number(r.CAR_correlation_vs_100_session).toFixed(3)];}));
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
$("eventVolatility").innerHTML=table(["Event","Stock","Pre window","Post window","Pre annualized vol","Post annualized vol","Volatility change","Post/pre ratio","Pre/Post n","Comparison status","Benchmark pre/post vol","Event-window residual SD"],rows.map(function(r){return [esc(r.event_id),esc(r.stock),esc(r.pre_event_window),esc(r.post_event_window),volPct(r.pre_event_realized_volatility_annualized),volPct(r.post_event_realized_volatility_annualized),signed(num(r.realized_volatility_percentage_change)),num(r.post_to_pre_realized_volatility_ratio)===null?"—":Number(r.post_to_pre_realized_volatility_ratio).toFixed(2),esc(r.pre_event_observations)+"/"+esc(r.post_event_observations),esc(r.volatility_comparison_status),volPct(r.benchmark_pre_event_realized_volatility_annualized)+" / "+volPct(r.benchmark_post_event_realized_volatility_annualized),volPct(r.event_window_market_model_residual_volatility_daily)];}));
}
function categoryEvidence(){
var q=selected(),cars=(d["category_car_summary.csv"]||[]).filter(function(r){return !q.w||r.window===q.w;}),vol=(d["category_volatility_summary.csv"]||[]);
$("categoryEvidence").innerHTML='<h3>CAR by category and window</h3>'+table(["Category","Window","Status","Independent episodes","Events contributing","Mean CAR","Median CAR","Overlapping events excluded"],cars.map(function(r){return [esc(r.category),esc(r.window),esc(r.status),esc(r.independent_episodes),esc(r.events_contributing),signed(r.mean_event_CAR_pct),signed(r.median_event_CAR_pct),esc(r.excluded_overlapping_events)];}));
$("categoryEvidence").innerHTML+='<h3 style="margin-top:14px">Volatility by category</h3>'+table(["Category","Metric","Status","Independent episodes","Mean","Median","Overlapping events excluded","Events in master"],vol.map(function(r){var isPct=String(r.metric||"").indexOf("_pct")>=0;return [esc(r.category),esc(r.metric),esc(r.status),esc(r.independent_episodes),isPct?pct(num(r.mean_value)):num(r.mean_value)===null?"—":Number(r.mean_value).toFixed(2),isPct?pct(num(r.median_value)):num(r.median_value)===null?"—":Number(r.median_value).toFixed(2),esc(r.excluded_overlapping_events),esc(r.category_event_count_in_master)];}));
}
function provenance(){
var p=m.input_data_provenance||{},checks=m.input_checksums_sha256||{};
$("provenance").innerHTML='<div class="tablewrap"><table><tbody>'+
[['Snapshot ID',m.snapshot_id],['Period',(m.period_start||"—")+" to "+(m.period_end||"—")],['Provider',p.source||p.provider],['Retrieved UTC',p.retrieved_at_utc],['Benchmark',m.benchmark],['Price input',m.input_prices],['Price SHA-256',checks.prices],['Estimation sessions',m.estimation_sessions],['Gap sessions',m.gap_sessions],['Event-window radius',m.event_window_sessions],['Event sources independently verified',m.source_independently_verified]].map(function(r){return '<tr><td>'+esc(r[0])+'</td><td style="text-align:left;white-space:normal;overflow-wrap:anywhere">'+esc(r[1])+'</td></tr>';}).join("")+
'</tbody></table></div><p class="small">Stock returns use provider adjusted-close fields; the Nifty benchmark uses a price-index close, so dividend conventions differ. The manifest records provenance but does not independently validate every event source.</p>';
}
function render(){
var q=selected(),eventRows=study.filter(function(r){return r.event_id===q.e;}),testRows=sig.filter(function(r){return !q.e||r.event_id===q.e;});
var significant=testRows.filter(function(r){return String(r.significant_after_holm_5pct).toLowerCase()==="true";}).length;
var obs=study.length,carCount=cars.length;
$("metrics").innerHTML=metric("Historical events",m.event_count||events.length,"Selected event master")+
metric("Stock-event-day rows",obs.toLocaleString(),"Actual, expected and abnormal returns")+
metric("CAR estimates",carCount.toLocaleString(),"Three windows where complete")+
metric("Holm-significant in selected event",significant.toString(),"Of "+testRows.length+" tests in current event filter");
heatmap();eventDetail();chart();inference();estimationWindowSensitivity();robustnessEvidenceLedger();consistency();trading();register();eventDependenceAudit();eventTimingSensitivity();eventVolatility();categoryEvidence();provenance();
}
["eventSelect","stockSelect","windowSelect","sampleSelect"].forEach(function(id){$(id).addEventListener("change",render);});
render();
})();
</script></main></body></html>'''
    html = (
        template.replace("__TITLE__", title)
        .replace("__SNAPSHOT__", str(manifest.get("snapshot_id", "unavailable")))
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