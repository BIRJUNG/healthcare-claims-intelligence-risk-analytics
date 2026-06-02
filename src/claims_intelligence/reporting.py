from __future__ import annotations

import json
from html import escape
from pathlib import Path

import pandas as pd

from .models import ModelOutputs


def render_dashboard(
    base_tables: dict[str, pd.DataFrame],
    marts: dict[str, pd.DataFrame],
    model_outputs: ModelOutputs,
    quality_report: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _payload(base_tables, marts, model_outputs, quality_report)
    html = DASHBOARD_TEMPLATE.replace("__PAYLOAD__", json.dumps(payload, default=str))
    output_path.write_text(html, encoding="utf-8")


def write_executive_summary(
    base_tables: dict[str, pd.DataFrame],
    marts: dict[str, pd.DataFrame],
    model_outputs: ModelOutputs,
    quality_report: pd.DataFrame,
    output_path: Path,
) -> None:
    claim = base_tables["fact_claim"]
    pmpm = marts["mart_member_month_pmpm"]
    providers = marts["mart_provider_performance"]
    high_cost = marts["mart_high_cost_member"]
    failures = int((quality_report["status"] == "FAIL").sum())
    lines = [
        "# Executive Summary",
        "",
        f"- Claims analyzed: {len(claim):,}",
        f"- Members analyzed: {base_tables['dim_member']['member_key'].nunique():,}",
        f"- Total paid amount: ${claim['paid_amount'].sum():,.0f}",
        f"- Weighted PMPM: ${pmpm['paid_amount'].sum() / max(pmpm['member_months'].sum(), 1):,.2f}",
        f"- Inpatient episodes: {len(base_tables['fact_inpatient_episode']):,}",
        f"- High-cost members in queue: {len(high_cost):,}",
        f"- Providers in intervention tier: {(providers['performance_tier'].astype(str) == 'Intervention').sum():,}",
        f"- Quality failures: {failures}",
        "",
        "## Recommended Actions",
        "",
        "1. Use the high-cost queue for complex care management prioritization.",
        "2. Review providers in the intervention tier for peer benchmarking and payment integrity follow-up.",
        "3. Use the readmission queue for 7-day follow-up workflows.",
        "4. Use the HCC gap table for chart review and chronic condition recapture planning.",
        "",
        "## Model Metrics",
        "",
    ]
    for name, metrics in model_outputs.metrics.items():
        lines.append(f"### {name}")
        for key, value in metrics.items():
            lines.append(f"- `{key}`: {value}")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _payload(
    base_tables: dict[str, pd.DataFrame],
    marts: dict[str, pd.DataFrame],
    model_outputs: ModelOutputs,
    quality_report: pd.DataFrame,
) -> dict[str, object]:
    claim = base_tables["fact_claim"]
    member = base_tables["dim_member"]
    pmpm = marts["mart_member_month_pmpm"]
    readmission = marts["mart_readmission_queue"]
    provider = marts["mart_provider_performance"]
    high_cost = marts["mart_high_cost_member"]
    fwa = marts["mart_fwa_queue"]
    hcc = marts["mart_hcc_risk"]
    weighted_pmpm = float(pmpm["paid_amount"].sum() / max(pmpm["member_months"].sum(), 1))
    readmission_rate = 0.0
    if not base_tables["fact_inpatient_episode"].empty:
        readmission_rate = float(base_tables["fact_inpatient_episode"]["readmission_within_30_days"].mean())
    kpis = {
        "claims": int(len(claim)),
        "members": int(len(member)),
        "paid": float(claim["paid_amount"].sum()),
        "pmpm": weighted_pmpm,
        "readmissionRate": readmission_rate,
        "complexMembers": int((high_cost["priority_tier"].astype(str) == "Complex care").sum()),
        "providerInterventions": int((provider["performance_tier"].astype(str) == "Intervention").sum()),
        "fwaCases": int(len(fwa)),
        "qualityFailures": int((quality_report["status"] == "FAIL").sum()),
    }
    pmpm_trend = pmpm.groupby("month_key", as_index=False).agg(pmpm=("pmpm", "mean"), paid_amount=("paid_amount", "sum"), member_months=("member_months", "sum"))
    util = marts["mart_utilization"].copy()
    risk_cost = high_cost.groupby("risk_segment", as_index=False).agg(paid_12m=("paid_12m", "sum"), members=("member_key", "nunique"))
    provider_chart = provider.head(150)[["provider_name", "specialty_group", "payment_per_member", "services_per_member", "provider_outlier_score", "performance_tier"]]
    return {
        "kpis": kpis,
        "pmpmTrend": _records(pmpm_trend),
        "utilization": _records(util),
        "riskCost": _records(risk_cost),
        "providerChart": _records(provider_chart),
        "highCost": _records(high_cost.head(400)),
        "providers": _records(provider.head(250)),
        "readmissions": _records(readmission.head(350)),
        "fwa": _records(fwa.head(350)),
        "hcc": _records(hcc.head(350)),
        "quality": _records(quality_report),
        "metrics": model_outputs.metrics,
    }


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    clean = frame.copy()
    for col in clean.select_dtypes(include=["category"]).columns:
        clean[col] = clean[col].astype(str)
    clean = clean.replace({pd.NA: None})
    return clean.to_dict(orient="records")


DASHBOARD_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Healthcare Claims Intelligence & Risk Analytics</title>
  <link rel="preconnect" href="https://cdn.plot.ly">
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {
      color-scheme: dark;
      --bg: #08111f;
      --panel: rgba(255, 255, 255, 0.08);
      --panel-strong: rgba(255, 255, 255, 0.13);
      --text: #f7fbff;
      --muted: rgba(229, 240, 255, 0.68);
      --line: rgba(255, 255, 255, 0.16);
      --brand: #6ee7f9;
      --brand-2: #a78bfa;
      --brand-3: #34d399;
      --danger: #fb7185;
      --warn: #fbbf24;
      --ok: #34d399;
      --shadow: 0 24px 80px rgba(0, 0, 0, 0.34);
      --radius: 8px;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    [data-theme="light"] {
      color-scheme: light;
      --bg: #eef6ff;
      --panel: rgba(255, 255, 255, 0.72);
      --panel-strong: rgba(255, 255, 255, 0.9);
      --text: #0d1726;
      --muted: rgba(30, 41, 59, 0.70);
      --line: rgba(15, 23, 42, 0.12);
      --shadow: 0 24px 80px rgba(31, 41, 55, 0.16);
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      background:
        radial-gradient(circle at 12% 8%, rgba(110,231,249,0.20), transparent 28rem),
        radial-gradient(circle at 82% 12%, rgba(167,139,250,0.20), transparent 30rem),
        radial-gradient(circle at 50% 86%, rgba(52,211,153,0.14), transparent 30rem),
        linear-gradient(135deg, var(--bg), #0d1829 48%, var(--bg));
      overflow-x: hidden;
    }
    body::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background-image:
        linear-gradient(rgba(255,255,255,0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.04) 1px, transparent 1px);
      background-size: 58px 58px;
      mask-image: linear-gradient(to bottom, rgba(0,0,0,.8), transparent);
    }
    a { color: inherit; }
    .shell { width: min(1480px, calc(100% - 32px)); margin: 0 auto; }
    header {
      position: sticky;
      top: 0;
      z-index: 20;
      backdrop-filter: blur(24px);
      background: rgba(8, 17, 31, 0.66);
      border-bottom: 1px solid var(--line);
    }
    [data-theme="light"] header { background: rgba(238,246,255,.72); }
    .nav { display: flex; align-items: center; justify-content: space-between; min-height: 74px; gap: 18px; }
    .brand { display: flex; align-items: center; gap: 12px; font-weight: 800; letter-spacing: 0; }
    .mark {
      width: 38px; height: 38px; border-radius: 8px;
      background: linear-gradient(135deg, var(--brand), var(--brand-2), var(--brand-3));
      box-shadow: 0 0 34px rgba(110,231,249,.36);
    }
    .links { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; justify-content: flex-end; }
    .links a, button, .chip {
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--text);
      border-radius: var(--radius);
      padding: 10px 12px;
      font: inherit;
      text-decoration: none;
      cursor: pointer;
      backdrop-filter: blur(18px);
      transition: transform .18s ease, border-color .18s ease, background .18s ease;
    }
    .links a:hover, button:hover { transform: translateY(-1px); border-color: rgba(110,231,249,.55); background: var(--panel-strong); }
    .hero { padding: 54px 0 28px; display: grid; grid-template-columns: minmax(0, 1.05fr) minmax(340px, .95fr); gap: 24px; align-items: stretch; }
    .hero-copy, .panel, .metric {
      border: 1px solid var(--line);
      background: linear-gradient(145deg, var(--panel-strong), var(--panel));
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      backdrop-filter: blur(24px);
    }
    .hero-copy { padding: clamp(24px, 5vw, 56px); position: relative; overflow: hidden; }
    .hero-copy::after {
      content: "";
      position: absolute;
      inset: 1px;
      border-radius: var(--radius);
      background: linear-gradient(120deg, rgba(255,255,255,.18), transparent 28%, rgba(255,255,255,.05));
      pointer-events: none;
    }
    .eyebrow { color: var(--brand); font-weight: 800; text-transform: uppercase; font-size: 12px; letter-spacing: .14em; }
    h1 { font-size: clamp(36px, 6vw, 76px); line-height: 0.96; margin: 14px 0 18px; letter-spacing: 0; }
    .subtitle { color: var(--muted); font-size: clamp(16px, 2vw, 20px); line-height: 1.6; max-width: 760px; }
    .hero-actions { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 24px; }
    .primary { background: linear-gradient(135deg, rgba(110,231,249,.95), rgba(167,139,250,.92)); color: #06101e; font-weight: 800; }
    .grid { display: grid; gap: 14px; }
    .metrics { grid-template-columns: repeat(4, minmax(0, 1fr)); margin: 18px 0 24px; }
    .metric { padding: 18px; min-height: 118px; }
    .metric span { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .12em; font-weight: 800; }
    .metric strong { display: block; margin-top: 12px; font-size: clamp(24px, 3vw, 36px); letter-spacing: 0; }
    .metric small { color: var(--muted); display: block; margin-top: 8px; }
    .section { padding: 30px 0; }
    .section-head { display: flex; justify-content: space-between; gap: 16px; align-items: end; margin-bottom: 14px; }
    h2 { margin: 0; font-size: clamp(24px, 3vw, 38px); letter-spacing: 0; }
    .section-head p { color: var(--muted); max-width: 760px; line-height: 1.6; margin: 8px 0 0; }
    .two { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .three { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    .panel { padding: 18px; overflow: hidden; }
    .panel h3 { margin: 0 0 12px; font-size: 18px; }
    .chart { height: 360px; }
    .toolbar { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 12px; align-items: center; }
    input, select {
      width: min(100%, 340px);
      border: 1px solid var(--line);
      background: rgba(255,255,255,.08);
      color: var(--text);
      border-radius: var(--radius);
      padding: 11px 12px;
      font: inherit;
      outline: none;
    }
    option { color: #0f172a; }
    .table-wrap { overflow: auto; border: 1px solid var(--line); border-radius: var(--radius); max-height: 520px; }
    table { width: 100%; border-collapse: collapse; min-width: 980px; }
    th, td { padding: 12px 13px; text-align: left; border-bottom: 1px solid var(--line); font-size: 13px; vertical-align: top; }
    th { color: var(--muted); text-transform: uppercase; letter-spacing: .08em; font-size: 11px; position: sticky; top: 0; background: rgba(15, 23, 42, .82); backdrop-filter: blur(14px); cursor: pointer; z-index: 1; }
    [data-theme="light"] th { background: rgba(255,255,255,.88); }
    tr:hover td { background: rgba(110,231,249,.06); }
    .pill { display: inline-flex; align-items: center; border: 1px solid var(--line); border-radius: 999px; padding: 4px 8px; background: rgba(255,255,255,.08); white-space: nowrap; }
    .modal { position: fixed; inset: 0; display: none; align-items: center; justify-content: center; z-index: 50; background: rgba(3, 7, 18, .62); padding: 18px; }
    .modal.open { display: flex; }
    .modal-card { width: min(760px, 100%); max-height: min(720px, 92vh); overflow: auto; background: rgba(12, 23, 40, .92); border: 1px solid var(--line); border-radius: var(--radius); box-shadow: var(--shadow); backdrop-filter: blur(24px); padding: 20px; }
    [data-theme="light"] .modal-card { background: rgba(255,255,255,.94); }
    .detail-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
    .detail { border: 1px solid var(--line); border-radius: var(--radius); padding: 10px; }
    .detail span { color: var(--muted); display: block; font-size: 12px; margin-bottom: 4px; }
    footer { padding: 36px 0 48px; color: var(--muted); text-align: center; }
    @media (max-width: 980px) {
      .hero, .two, .three, .metrics { grid-template-columns: 1fr; }
      .links { justify-content: flex-start; }
      .nav { align-items: flex-start; flex-direction: column; padding: 14px 0; }
      table { min-width: 860px; }
    }
  </style>
</head>
<body>
  <script>const DATA = __PAYLOAD__;</script>
  <header>
    <div class="shell nav">
      <div class="brand"><div class="mark"></div><div>Healthcare Claims Intelligence</div></div>
      <nav class="links">
        <a href="#overview">Overview</a>
        <a href="#providers">Providers</a>
        <a href="#queues">Queues</a>
        <a href="#risk">HCC/RAF</a>
        <a href="#governance">Governance</a>
        <button id="themeToggle" type="button">Light Mode</button>
      </nav>
    </div>
  </header>
  <main class="shell">
    <section class="hero" id="overview">
      <div class="hero-copy">
        <div class="eyebrow">Medicare-style payer analytics platform</div>
        <h1>Claims, risk, utilization, and intervention intelligence.</h1>
        <p class="subtitle">A deployable analytics product for PMPM trend, high-cost members, readmissions, provider outliers, FWA review, and HCC/RAF-style risk operations.</p>
        <div class="hero-actions">
          <a class="links primary" href="#queues">Open Work Queues</a>
          <a class="links chip" href="#governance">View Governance</a>
        </div>
      </div>
      <div class="panel">
        <h3>Portfolio Scope</h3>
        <div id="riskChart" class="chart"></div>
      </div>
    </section>
    <section class="grid metrics" id="metrics"></section>
    <section class="section">
      <div class="section-head">
        <div>
          <h2>Financial And Utilization Trend</h2>
          <p>PMPM and utilization signals for executive review and care management planning.</p>
        </div>
      </div>
      <div class="grid two">
        <article class="panel"><h3>PMPM Trend</h3><div id="pmpmChart" class="chart"></div></article>
        <article class="panel"><h3>Utilization Per 1,000</h3><div id="utilChart" class="chart"></div></article>
      </div>
    </section>
    <section class="section" id="providers">
      <div class="section-head">
        <div>
          <h2>Provider Performance</h2>
          <p>Peer benchmarking for cost, utilization, quality, and intervention prioritization.</p>
        </div>
      </div>
      <div class="grid two">
        <article class="panel"><h3>Cost vs Utilization Outlier Matrix</h3><div id="providerChart" class="chart"></div></article>
        <article class="panel">
          <h3>Provider Scorecard</h3>
          <div class="toolbar"><input id="providerSearch" type="search" placeholder="Search provider, specialty, action"><button data-export="providerTable">Export CSV</button></div>
          <div class="table-wrap"><table id="providerTable"></table></div>
        </article>
      </div>
    </section>
    <section class="section" id="queues">
      <div class="section-head">
        <div>
          <h2>Operational Work Queues</h2>
          <p>High-cost member, readmission, and payment-integrity queues with local triage actions.</p>
        </div>
      </div>
      <div class="grid">
        <article class="panel">
          <h3>High-Cost Member Queue</h3>
          <div class="toolbar"><input id="memberSearch" type="search" placeholder="Search member, state, risk, intervention"><button data-export="memberTable">Export CSV</button></div>
          <div class="table-wrap"><table id="memberTable"></table></div>
        </article>
        <article class="panel">
          <h3>Readmission Queue</h3>
          <div class="toolbar"><input id="readmissionSearch" type="search" placeholder="Search member, provider, diagnosis"><button data-export="readmissionTable">Export CSV</button></div>
          <div class="table-wrap"><table id="readmissionTable"></table></div>
        </article>
        <article class="panel">
          <h3>FWA Review Queue</h3>
          <div class="toolbar"><input id="fwaSearch" type="search" placeholder="Search claim, provider, reason"><button data-export="fwaTable">Export CSV</button></div>
          <div class="table-wrap"><table id="fwaTable"></table></div>
        </article>
      </div>
    </section>
    <section class="section" id="risk">
      <div class="section-head">
        <div>
          <h2>HCC/RAF Risk Adjustment</h2>
          <p>Risk segmentation and suspected documentation gap prioritization.</p>
        </div>
      </div>
      <article class="panel">
        <div class="toolbar"><input id="hccSearch" type="search" placeholder="Search member, segment, action"><button data-export="hccTable">Export CSV</button></div>
        <div class="table-wrap"><table id="hccTable"></table></div>
      </article>
    </section>
    <section class="section" id="governance">
      <div class="section-head">
        <div>
          <h2>Governance</h2>
          <p>Quality checks, model metrics, and simulation boundaries.</p>
        </div>
      </div>
      <div class="grid two">
        <article class="panel"><h3>Data Quality</h3><div class="table-wrap"><table id="qualityTable"></table></div></article>
        <article class="panel"><h3>Model Metrics</h3><div id="modelMetrics"></div></article>
      </div>
    </section>
  </main>
  <div class="modal" id="detailModal" role="dialog" aria-modal="true">
    <div class="modal-card">
      <div class="section-head"><h2 id="modalTitle">Details</h2><button id="closeModal">Close</button></div>
      <div id="modalBody" class="detail-grid"></div>
    </div>
  </div>
  <footer class="shell">Synthetic or de-identified data only. Do not publish PHI in a public repository.</footer>
  <script>
    const fmt = new Intl.NumberFormat("en-US");
    const money = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });
    const pct = new Intl.NumberFormat("en-US", { style: "percent", maximumFractionDigits: 1 });
    const tableState = {};
    const triageKey = "claims-intelligence-triage";
    const triage = JSON.parse(localStorage.getItem(triageKey) || "{}");

    function saveTriage() { localStorage.setItem(triageKey, JSON.stringify(triage)); }
    function pretty(k) { return k.replaceAll("_", " ").replace(/([A-Z])/g, " $1").replace(/\b\w/g, c => c.toUpperCase()); }
    function formatValue(key, value) {
      if (value === null || value === undefined || value === "") return "NA";
      if (/amount|paid|allowed|pmpm|payment|responsibility|savings/i.test(key)) return money.format(Number(value) || 0);
      if (/rate|percentile|score|probability/i.test(key)) return Number(value) <= 1.5 ? pct.format(Number(value) || 0) : Number(value).toFixed(2);
      return String(value);
    }
    function renderMetrics() {
      const items = [
        ["Claims", fmt.format(DATA.kpis.claims), "claim-level rows"],
        ["Members", fmt.format(DATA.kpis.members), "covered lives"],
        ["Paid Amount", money.format(DATA.kpis.paid), "synthetic reimbursement"],
        ["PMPM", money.format(DATA.kpis.pmpm), "member-month cost"],
        ["Readmission Rate", pct.format(DATA.kpis.readmissionRate), "30-day episode rate"],
        ["Complex Members", fmt.format(DATA.kpis.complexMembers), "care management tier"],
        ["Provider Interventions", fmt.format(DATA.kpis.providerInterventions), "peer outlier tier"],
        ["FWA Cases", fmt.format(DATA.kpis.fwaCases), "payment integrity queue"],
      ];
      document.querySelector("#metrics").innerHTML = items.map(([label, value, note]) => `<article class="metric"><span>${label}</span><strong>${value}</strong><small>${note}</small></article>`).join("");
    }
    function table(id, rows, columns, searchId, keyField) {
      tableState[id] = { rows, columns, filtered: rows, sortKey: null, dir: 1, keyField };
      const search = searchId ? document.querySelector(`#${searchId}`) : null;
      if (search) search.addEventListener("input", () => drawTable(id));
      drawTable(id);
    }
    function drawTable(id) {
      const state = tableState[id];
      const search = document.querySelector(`#${id.replace("Table", "Search")}`);
      const needle = search ? search.value.trim().toLowerCase() : "";
      let rows = state.rows.filter(row => !needle || Object.values(row).join(" ").toLowerCase().includes(needle));
      if (state.sortKey) rows = [...rows].sort((a,b) => String(a[state.sortKey]).localeCompare(String(b[state.sortKey]), undefined, { numeric: true }) * state.dir);
      state.filtered = rows;
      const head = `<thead><tr>${state.columns.map(c => `<th data-sort="${c.key}">${c.label}</th>`).join("")}<th>Actions</th></tr></thead>`;
      const body = rows.map(row => {
        const rowKey = state.keyField ? row[state.keyField] : Object.values(row)[0];
        const t = triage[rowKey] || {};
        const cells = state.columns.map(c => `<td>${c.pill ? `<span class="pill">${formatValue(c.key, row[c.key])}</span>` : formatValue(c.key, row[c.key])}</td>`).join("");
        const actions = `<td><button data-detail="${id}" data-key="${rowKey}">Details</button> <button data-flag="${rowKey}">${t.flagged ? "Unflag" : "Flag"}</button> <button data-resolve="${rowKey}">${t.resolved ? "Reopen" : "Resolve"}</button></td>`;
        return `<tr>${cells}${actions}</tr>`;
      }).join("");
      document.querySelector(`#${id}`).innerHTML = head + `<tbody>${body}</tbody>`;
      document.querySelectorAll(`#${id} th[data-sort]`).forEach(th => th.addEventListener("click", () => {
        const key = th.dataset.sort;
        state.dir = state.sortKey === key ? state.dir * -1 : 1;
        state.sortKey = key;
        drawTable(id);
      }));
      document.querySelectorAll(`#${id} [data-detail]`).forEach(btn => btn.addEventListener("click", () => openDetail(id, btn.dataset.key)));
      document.querySelectorAll(`#${id} [data-flag]`).forEach(btn => btn.addEventListener("click", () => { triage[btn.dataset.flag] = { ...(triage[btn.dataset.flag] || {}), flagged: !(triage[btn.dataset.flag] || {}).flagged }; saveTriage(); drawTable(id); }));
      document.querySelectorAll(`#${id} [data-resolve]`).forEach(btn => btn.addEventListener("click", () => { triage[btn.dataset.resolve] = { ...(triage[btn.dataset.resolve] || {}), resolved: !(triage[btn.dataset.resolve] || {}).resolved }; saveTriage(); drawTable(id); }));
    }
    function openDetail(id, key) {
      const state = tableState[id];
      const row = state.rows.find(r => String(r[state.keyField]) === String(key)) || {};
      document.querySelector("#modalTitle").textContent = `${pretty(id.replace("Table", ""))} Details`;
      document.querySelector("#modalBody").innerHTML = Object.entries(row).map(([k,v]) => `<div class="detail"><span>${pretty(k)}</span><strong>${formatValue(k, v)}</strong></div>`).join("");
      document.querySelector("#detailModal").classList.add("open");
    }
    function exportTable(id) {
      const state = tableState[id];
      const rows = state.filtered || state.rows;
      const cols = state.columns.map(c => c.key);
      const csv = [cols.join(","), ...rows.map(row => cols.map(c => `"${String(row[c] ?? "").replaceAll('"', '""')}"`).join(","))].join("\n");
      const blob = new Blob([csv], { type: "text/csv" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `${id}.csv`;
      a.click();
      URL.revokeObjectURL(a.href);
    }
    function charts() {
      const layout = { paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: { color: getComputedStyle(document.documentElement).getPropertyValue("--text") }, margin: { t: 18, r: 18, b: 42, l: 56 }, xaxis: { gridcolor: "rgba(148,163,184,.18)" }, yaxis: { gridcolor: "rgba(148,163,184,.18)" } };
      Plotly.newPlot("pmpmChart", [{ x: DATA.pmpmTrend.map(d => d.month_key), y: DATA.pmpmTrend.map(d => d.pmpm), type: "scatter", mode: "lines+markers", line: { color: "#6ee7f9", width: 4 }, fill: "tozeroy" }], layout, { responsive: true, displayModeBar: false });
      Plotly.newPlot("utilChart", [
        { x: DATA.utilization.map(d => d.month_key), y: DATA.utilization.map(d => d.admissions_per_1000), name: "Admissions", type: "scatter", mode: "lines", line: { color: "#a78bfa", width: 3 } },
        { x: DATA.utilization.map(d => d.month_key), y: DATA.utilization.map(d => d.ed_visits_per_1000), name: "ED visits", type: "scatter", mode: "lines", line: { color: "#34d399", width: 3 } },
      ], layout, { responsive: true, displayModeBar: false });
      Plotly.newPlot("riskChart", [{ labels: DATA.riskCost.map(d => d.risk_segment), values: DATA.riskCost.map(d => d.paid_12m), type: "pie", hole: .62, marker: { colors: ["#6ee7f9", "#a78bfa", "#34d399", "#fbbf24"] } }], { ...layout, showlegend: true }, { responsive: true, displayModeBar: false });
      Plotly.newPlot("providerChart", [{ x: DATA.providerChart.map(d => d.payment_per_member), y: DATA.providerChart.map(d => d.services_per_member), text: DATA.providerChart.map(d => d.provider_name), mode: "markers", type: "scatter", marker: { color: DATA.providerChart.map(d => d.provider_outlier_score), colorscale: "Turbo", size: 11, opacity: .86, line: { color: "rgba(255,255,255,.55)", width: 1 } } }], { ...layout, xaxis: { title: "Payment per member" }, yaxis: { title: "Services per member" } }, { responsive: true, displayModeBar: false });
    }
    function initTables() {
      table("providerTable", DATA.providers, [
        { key: "provider_name", label: "Provider" }, { key: "specialty_group", label: "Specialty", pill: true }, { key: "payment_per_member", label: "Payment/member" }, { key: "peer_cost_percentile", label: "Cost pct" }, { key: "provider_outlier_score", label: "Outlier" }, { key: "performance_tier", label: "Tier", pill: true }, { key: "recommended_action", label: "Action" }
      ], "providerSearch", "provider_id");
      table("memberTable", DATA.highCost, [
        { key: "member_id", label: "Member" }, { key: "state_code", label: "State", pill: true }, { key: "risk_segment", label: "Risk", pill: true }, { key: "paid_12m", label: "Paid 12m" }, { key: "admissions", label: "Admits" }, { key: "ed_visits", label: "ED" }, { key: "risk_priority_score", label: "Priority" }, { key: "recommended_intervention", label: "Intervention" }
      ], "memberSearch", "member_id");
      table("readmissionTable", DATA.readmissions, [
        { key: "member_id", label: "Member" }, { key: "provider_name", label: "Provider" }, { key: "diagnosis_group", label: "Diagnosis", pill: true }, { key: "length_of_stay", label: "LOS" }, { key: "readmission_within_30_days", label: "Readmit" }, { key: "readmission_priority_score", label: "Priority" }, { key: "recommended_action", label: "Action" }
      ], "readmissionSearch", "episode_key");
      table("fwaTable", DATA.fwa, [
        { key: "claim_id", label: "Claim" }, { key: "provider_name", label: "Provider" }, { key: "service_line", label: "Service", pill: true }, { key: "paid_amount", label: "Paid" }, { key: "fwa_score", label: "FWA score" }, { key: "review_reason", label: "Reason" }, { key: "recommended_action", label: "Action" }
      ], "fwaSearch", "claim_id");
      table("hccTable", DATA.hcc, [
        { key: "member_id", label: "Member" }, { key: "risk_segment", label: "Risk", pill: true }, { key: "raf_like_score", label: "RAF-like" }, { key: "hcc_count", label: "HCCs" }, { key: "suspected_hcc_gap_count", label: "Gaps" }, { key: "hcc_gap_priority", label: "Priority" }, { key: "risk_adjustment_action", label: "Action" }
      ], "hccSearch", "member_id");
      table("qualityTable", DATA.quality, [
        { key: "check_name", label: "Check" }, { key: "status", label: "Status", pill: true }, { key: "value", label: "Value" }, { key: "threshold", label: "Threshold" }
      ], "", "check_name");
    }
    function modelMetrics() {
      document.querySelector("#modelMetrics").innerHTML = Object.entries(DATA.metrics).map(([name, metrics]) => `<div class="panel" style="margin-bottom:12px;box-shadow:none;"><h3>${pretty(name)}</h3>${Object.entries(metrics).map(([k,v]) => `<div class="detail"><span>${pretty(k)}</span><strong>${formatValue(k, v)}</strong></div>`).join("")}</div>`).join("");
    }
    document.querySelector("#themeToggle").addEventListener("click", () => {
      const light = document.documentElement.dataset.theme !== "light";
      document.documentElement.dataset.theme = light ? "light" : "dark";
      document.querySelector("#themeToggle").textContent = light ? "Dark Mode" : "Light Mode";
      setTimeout(charts, 0);
    });
    document.querySelector("#closeModal").addEventListener("click", () => document.querySelector("#detailModal").classList.remove("open"));
    document.querySelector("#detailModal").addEventListener("click", e => { if (e.target.id === "detailModal") e.currentTarget.classList.remove("open"); });
    document.querySelectorAll("[data-export]").forEach(btn => btn.addEventListener("click", () => exportTable(btn.dataset.export)));
    renderMetrics();
    charts();
    initTables();
    modelMetrics();
  </script>
</body>
</html>"""
