//! Self-contained zero-CDN offline HTML Drift Receipt generator.

use crate::drift::{DriftReport, Severity};

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn badge(sev: Severity) -> &'static str {
    match sev {
        Severity::Stable => r#"<span class="pill ok">STABLE</span>"#,
        Severity::Warn => r#"<span class="pill warn">WARN</span>"#,
        Severity::Drift => r#"<span class="pill bad">DRIFT</span>"#,
    }
}

pub fn render_drift_html(report: &DriftReport) -> String {
    let max_bar_psi = report
        .features
        .iter()
        .map(|f| f.psi)
        .fold(0.25_f64, f64::max);

    let mut psi_bars = String::new();
    for f in report.features.iter().take(12) {
        let pct = ((f.psi / max_bar_psi) * 100.0).clamp(1.0, 100.0);
        let fill = match f.severity {
            Severity::Stable => "#10b981",
            Severity::Warn => "#f59e0b",
            Severity::Drift => "#ef4444",
        };
        psi_bars.push_str(&format!(
            r#"<div class="bar-row"><div class="bar-lbl">{}</div><div class="bar-track"><div class="bar-fill" style="width:{pct:.1}%;background:{fill}"></div></div><div class="bar-val">{:.3}</div></div>"#,
            esc(&f.feature),
            f.psi
        ));
    }

    let mut rows_html = String::new();
    for f in &report.features {
        let ks_s = f.ks_stat.map(|v| format!("{v:.3}")).unwrap_or_else(|| "—".to_string());
        let z_s = f.mean_z_shift.map(|v| format!("{v:+.2}σ")).unwrap_or_else(|| "—".to_string());
        let notes = if f.reasons.is_empty() {
            "Within bounds".to_string()
        } else {
            esc(&f.reasons.join("; "))
        };
        rows_html.push_str(&format!(
            r#"<tr>
              <td><strong>{}</strong> <span class="muted">({})</span></td>
              <td>{}</td>
              <td class="num">{:.3}</td>
              <td class="num">{}</td>
              <td class="num">{:.3}</td>
              <td class="num">{:+.1}pp</td>
              <td class="num">{}</td>
              <td class="note">{}</td>
            </tr>"#,
            esc(&f.feature),
            esc(&f.kind),
            badge(f.severity),
            f.psi,
            ks_s,
            f.js_divergence,
            f.missing_delta * 100.0,
            z_s,
            notes
        ));
    }

    let pred_section = if let Some(pd) = &report.prediction_drift {
        format!(
            r#"<div class="card"><h2>Prediction Score Drift — {} {}</h2>
            <p class="sub">PSI: <strong>{:.3}</strong> · KS: <strong>{}</strong> · JS: <strong>{:.3}</strong> · Mean shift: <strong>{}</strong></p></div>"#,
            esc(&pd.feature),
            badge(pd.severity),
            pd.psi,
            pd.ks_stat.map(|v| format!("{v:.3}")).unwrap_or_else(|| "—".to_string()),
            pd.js_divergence,
            pd.mean_z_shift.map(|v| format!("{v:+.2}σ")).unwrap_or_else(|| "—".to_string()),
        )
    } else {
        String::new()
    };

    let cal_section = if let Some(cd) = &report.calibration_drift {
        format!(
            r#"<div class="card"><h2>Calibration &amp; Concept Drift {}</h2>
            <p class="sub">Evaluated rows: <strong>{}</strong> · Current ECE: <strong>{:.4}</strong> · Brier: <strong>{:.4}</strong> · Observed rate: <strong>{:.1}%</strong> vs Mean pred: <strong>{:.1}%</strong> (gap {:+.1}pp)</p></div>"#,
            badge(cd.severity),
            cd.current.n_eval,
            cd.current.ece,
            cd.current.brier_score,
            cd.current.base_rate * 100.0,
            cd.current.mean_predicted * 100.0,
            cd.current.calibration_gap * 100.0,
        )
    } else {
        String::new()
    };

    let recs_html: String = report
        .recommendations
        .iter()
        .map(|r| format!("<li>{}</li>", esc(r)))
        .collect();

    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>driftkeel — Production ML Drift Receipt</title>
<style>
  :root {{ --bg:#0b1120; --card:#111a2e; --border:#1f2d47; --fg:#e2e8f0; --muted:#94a3b8; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; padding:28px; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; background:var(--bg); color:var(--fg); line-height:1.5; }}
  .wrap {{ max-width:1120px; margin:0 auto; }}
  .hdr {{ display:flex; justify-content:space-between; align-items:flex-start; gap:16px; flex-wrap:wrap; margin-bottom:20px; }}
  h1 {{ margin:0 0 6px; font-size:22px; }}
  h2 {{ margin:0 0 12px; font-size:15px; color:#f8fafc; }}
  .sub {{ color:var(--muted); font-size:13px; margin:0; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin-bottom:18px; }}
  .kpi {{ background:var(--card); border:1px solid var(--border); border-radius:10px; padding:14px 16px; }}
  .kpi .lbl {{ font-size:11px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); }}
  .kpi .val {{ font-size:22px; font-weight:700; margin-top:4px; }}
  .card {{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:18px 20px; margin-bottom:16px; }}
  .pill {{ display:inline-block; padding:3px 9px; border-radius:999px; font-size:11px; font-weight:700; letter-spacing:.04em; }}
  .pill.ok {{ background:rgba(16,185,129,.16); color:#34d399; border:1px solid rgba(16,185,129,.35); }}
  .pill.warn {{ background:rgba(245,158,11,.16); color:#fbbf24; border:1px solid rgba(245,158,11,.35); }}
  .pill.bad {{ background:rgba(239,68,68,.18); color:#f87171; border:1px solid rgba(239,68,68,.4); }}
  .bar-row {{ display:grid; grid-template-columns:200px 1fr 56px; align-items:center; gap:10px; margin-bottom:8px; font-size:13px; }}
  .bar-lbl {{ overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
  .bar-track {{ height:10px; background:#1e293b; border-radius:999px; overflow:hidden; }}
  .bar-fill {{ height:100%; border-radius:999px; }}
  .bar-val {{ text-align:right; font-family:ui-monospace,monospace; font-size:12px; color:#cbd5e1; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th, td {{ padding:9px 10px; border-bottom:1px solid var(--border); text-align:left; }}
  th {{ color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.05em; }}
  .num {{ text-align:right; font-family:ui-monospace,monospace; }}
  .muted {{ color:var(--muted); font-size:11px; }}
  .note {{ font-size:12px; color:#cbd5e1; }}
  ul {{ margin:0; padding-left:18px; font-size:13px; color:#cbd5e1; }}
  .foot {{ margin-top:14px; font-size:11px; color:var(--muted); font-family:ui-monospace,monospace; }}
</style>
</head>
<body>
<div class="wrap">
  <div class="hdr">
    <div>
      <h1>driftkeel — Production ML Drift Receipt {}</h1>
      <p class="sub">Reference: <strong>{}</strong> ({} rows) → Current: <strong>{}</strong> ({} rows)</p>
    </div>
    <div class="sub">Generated {}</div>
  </div>

  <div class="grid">
    <div class="kpi"><div class="lbl">Features Checked</div><div class="val">{}</div></div>
    <div class="kpi"><div class="lbl">Drifted / Warned</div><div class="val">{} / {}</div></div>
    <div class="kpi"><div class="lbl">Max PSI</div><div class="val">{:.3}</div></div>
    <div class="kpi"><div class="lbl">Max KS</div><div class="val">{:.3}</div></div>
    <div class="kpi"><div class="lbl">Max Missing Δ</div><div class="val">{:+.1}pp</div></div>
  </div>

  {pred_section}
  {cal_section}

  <div class="card">
    <h2>Top Features by Population Stability Index (PSI)</h2>
    {psi_bars}
  </div>

  <div class="card">
    <h2>Actionable Recommendations</h2>
    <ul>{recs_html}</ul>
  </div>

  <div class="card">
    <h2>Full Feature Drift Breakdown</h2>
    <table>
      <thead>
        <tr>
          <th>Feature</th><th>Status</th><th class="num">PSI</th><th class="num">KS</th><th class="num">JS</th><th class="num">Δ Missing</th><th class="num">Mean Shift</th><th>Diagnostics</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>

  <div class="foot">driftkeel v{} · receipt sha256: {}</div>
</div>
</body>
</html>"#,
        badge(report.verdict),
        esc(&report.reference_source),
        report.reference_rows,
        esc(&report.current_source),
        report.current_rows,
        esc(&report.generated_at),
        report.features_checked,
        report.features_drifted,
        report.features_warned,
        report.max_psi,
        report.max_ks,
        report.max_missing_delta * 100.0,
        esc(&report.version),
        esc(&report.sha256),
    )
}
