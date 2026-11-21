//! `driftkeel` — Zero-Python Rust ML feature, prediction, and concept drift monitor.

mod cli;
mod drift;
mod html;
mod io_data;
mod skill;

use std::fs;
use std::path::Path;
use std::process::ExitCode;

use clap::Parser;
use cli::{Cli, Command, FailOn, ResolvedFormat};
use drift::{BaselineSnapshot, DriftThresholds, Severity, build_snapshot, evaluate_drift};
use io_data::read_table;
use owo_colors::OwoColorize;

fn main() -> ExitCode {
    let cli = Cli::parse();
    match dispatch(cli) {
        Ok(code) => ExitCode::from(code),
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::from(2)
        }
    }
}

fn try_load_baseline_json(path: &Path) -> Result<Option<BaselineSnapshot>, String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    if ext != "json" {
        return Ok(None);
    }
    let content =
        fs::read_to_string(path).map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    if let Ok(snap) = serde_json::from_str::<BaselineSnapshot>(&content)
        && snap.tool == "driftkeel"
    {
        return Ok(Some(snap));
    }
    Ok(None)
}

fn dispatch(cli: Cli) -> Result<u8, String> {
    match cli.command {
        Command::Skill { install, format: _ } => {
            skill::handle_skill(install.as_deref())?;
            Ok(0)
        }
        Command::Snapshot {
            input,
            out,
            bins,
            score_col,
            label_col,
            ignore,
            categorical,
            format,
        } => {
            let table = read_table(&input)?;
            let snap = build_snapshot(
                &table,
                &input.display().to_string(),
                bins,
                score_col.as_deref(),
                label_col.as_deref(),
                &ignore,
                &categorical,
            );
            if let Some(parent) = out.parent()
                && !parent.as_os_str().is_empty()
            {
                fs::create_dir_all(parent).map_err(|e| format!("failed creating dir: {e}"))?;
            }
            let json_str = serde_json::to_string_pretty(&snap)
                .map_err(|e| format!("failed serializing baseline: {e}"))?;
            fs::write(&out, &json_str)
                .map_err(|e| format!("failed writing {}: {e}", out.display()))?;

            let summary = serde_json::json!({
                "tool": "driftkeel",
                "version": env!("CARGO_PKG_VERSION"),
                "action": "snapshot",
                "source": input.display().to_string(),
                "out": out.display().to_string(),
                "row_count": snap.row_count,
                "columns_captured": snap.columns.len(),
                "bins": snap.bins,
                "sha256": snap.sha256,
            });
            match format.resolve() {
                ResolvedFormat::Json => println!("{}", serde_json::to_string_pretty(&summary).unwrap()),
                ResolvedFormat::Agent => println!("{}", serde_json::to_string(&summary).unwrap()),
                ResolvedFormat::Human => {
                    println!(
                        "{} snapshot {} ({} rows, {} features) -> {} [sha256: {}]",
                        "driftkeel".bold().green(),
                        input.display(),
                        snap.row_count,
                        snap.columns.len(),
                        out.display(),
                        &snap.sha256[..12]
                    );
                }
            }
            Ok(0)
        }
        Command::Check {
            reference,
            current,
            bins,
            score_col,
            label_col,
            ignore,
            categorical,
            psi_warn,
            psi_fail,
            ks_warn,
            ks_fail,
            js_warn,
            js_fail,
            missing_warn,
            missing_fail,
            ece_fail,
            fail_on,
            html: html_path,
            format,
        } => {
            let (baseline, raw_ref) = if let Some(snap) = try_load_baseline_json(&reference)? {
                (snap, None)
            } else {
                let ref_tbl = read_table(&reference)?;
                let snap = build_snapshot(
                    &ref_tbl,
                    &reference.display().to_string(),
                    bins,
                    score_col.as_deref(),
                    label_col.as_deref(),
                    &ignore,
                    &categorical,
                );
                (snap, Some(ref_tbl))
            };
            let cur_tbl = read_table(&current)?;
            let thresholds = DriftThresholds {
                psi_warn,
                psi_fail,
                ks_warn,
                ks_fail,
                js_warn,
                js_fail,
                missing_warn,
                missing_fail,
                ece_fail,
            };

            let report = evaluate_drift(
                &baseline,
                raw_ref.as_ref(),
                &cur_tbl,
                &current.display().to_string(),
                score_col.as_deref(),
                label_col.as_deref(),
                &ignore,
                thresholds,
            );

            if let Some(hp) = &html_path {
                let rendered = html::render_drift_html(&report);
                if let Some(parent) = hp.parent()
                    && !parent.as_os_str().is_empty()
                {
                    fs::create_dir_all(parent).map_err(|e| format!("failed creating dir: {e}"))?;
                }
                fs::write(hp, rendered)
                    .map_err(|e| format!("failed writing {}: {e}", hp.display()))?;
                eprintln!("wrote drift receipt -> {}", hp.display());
            }

            match format.resolve() {
                ResolvedFormat::Json => {
                    println!("{}", serde_json::to_string_pretty(&report).unwrap());
                }
                ResolvedFormat::Agent => {
                    println!("{}", serde_json::to_string(&report).unwrap());
                }
                ResolvedFormat::Human => {
                    let verdict_label = match report.verdict {
                        Severity::Stable => "STABLE".green().bold().to_string(),
                        Severity::Warn => "WARN".yellow().bold().to_string(),
                        Severity::Drift => "DRIFT".red().bold().to_string(),
                    };
                    println!(
                        "{} check {} -> {} | verdict: {} (drifted={}, warned={}, max_psi={:.3}, max_ks={:.3})",
                        "driftkeel".bold(),
                        report.reference_source,
                        report.current_source,
                        verdict_label,
                        report.features_drifted,
                        report.features_warned,
                        report.max_psi,
                        report.max_ks
                    );
                    for f in &report.features {
                        let tag = match f.severity {
                            Severity::Stable => "ok   ".green().to_string(),
                            Severity::Warn => "warn ".yellow().to_string(),
                            Severity::Drift => "DRIFT".red().bold().to_string(),
                        };
                        let ks_s = f
                            .ks_stat
                            .map(|v| format!("{v:.3}"))
                            .unwrap_or_else(|| "  -  ".to_string());
                        println!(
                            "  [{tag}] {:<24} psi={:.3}  ks={}  js={:.3}  miss={:+.1}pp  {}",
                            f.feature,
                            f.psi,
                            ks_s,
                            f.js_divergence,
                            f.missing_delta * 100.0,
                            f.reasons.join("; ")
                        );
                    }
                    if let Some(pd) = &report.prediction_drift {
                        println!(
                            "  [PRED ] {:<24} psi={:.3}  ks={}  js={:.3}  {}",
                            pd.feature,
                            pd.psi,
                            pd.ks_stat.map(|v| format!("{v:.3}")).unwrap_or_else(|| "-".into()),
                            pd.js_divergence,
                            pd.reasons.join("; ")
                        );
                    }
                    if let Some(cd) = &report.calibration_drift {
                        println!(
                            "  [CAL  ] ece={:.4}  brier={:.4}  base_rate={:.1}%  mean_pred={:.1}%  {}",
                            cd.current.ece,
                            cd.current.brier_score,
                            cd.current.base_rate * 100.0,
                            cd.current.mean_predicted * 100.0,
                            cd.reasons.join("; ")
                        );
                    }
                }
            }

            let failed = match fail_on {
                FailOn::Error => report.verdict == Severity::Drift,
                FailOn::Warn => report.verdict != Severity::Stable,
            };
            Ok(if failed { 1 } else { 0 })
        }
    }
}
