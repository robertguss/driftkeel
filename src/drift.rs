//! Core ML drift statistics: PSI, exact two-sample KS + asymptotic p-value,
//! base-2 Jensen-Shannon divergence, missingness deltas, unseen categories,
//! prediction score drift, and post-hoc calibration (ECE / Brier) drift.

use std::collections::{HashMap, HashSet};

use chrono::Utc;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::io_data::DataTable;

const EPS: f64 = 1e-6;
const SKETCH_SIZE: usize = 256;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineSnapshot {
    pub schema_version: u32,
    pub tool: String,
    pub tool_version: String,
    pub created_at: String,
    pub source_name: String,
    pub row_count: usize,
    pub bins: usize,
    pub sha256: String,
    pub score_col: Option<String>,
    pub label_col: Option<String>,
    pub columns: IndexMap<String, ColumnBaseline>,
    pub calibration_baseline: Option<CalibrationSummary>,
}

impl BaselineSnapshot {
    /// Validate invariants of a deserialized baseline JSON snapshot.
    pub fn validate(&self) -> Result<(), String> {
        if self.tool != "driftkeel" {
            return Err(format!("invalid baseline tool `{}`", self.tool));
        }
        if self.bins < 2 {
            return Err(format!(
                "invalid baseline bins `{}` (must be >= 2)",
                self.bins
            ));
        }
        for (col, base) in &self.columns {
            match base {
                ColumnBaseline::Numeric {
                    missing_rate,
                    bin_edges,
                    bin_props,
                    quantile_sketch,
                    ..
                } => {
                    if !missing_rate.is_finite() || !(*missing_rate >= 0.0 && *missing_rate <= 1.0)
                    {
                        return Err(format!("invalid missing_rate on baseline column `{col}`"));
                    }
                    if bin_props.len() != bin_edges.len() + 1 {
                        return Err(format!(
                            "corrupt baseline column `{col}`: bin_props.len() ({}) != bin_edges.len() + 1 ({})",
                            bin_props.len(),
                            bin_edges.len() + 1
                        ));
                    }
                    if bin_edges.iter().any(|v| !v.is_finite())
                        || bin_edges.windows(2).any(|w| w[0] > w[1])
                    {
                        return Err(format!(
                            "corrupt baseline column `{col}`: bin_edges must be finite and non-decreasing"
                        ));
                    }
                    if bin_props.iter().any(|v| !v.is_finite() || *v < 0.0) {
                        return Err(format!(
                            "corrupt baseline column `{col}`: bin_props must be finite and >= 0"
                        ));
                    }
                    if quantile_sketch.iter().any(|v| !v.is_finite())
                        || quantile_sketch.windows(2).any(|w| w[0] > w[1])
                    {
                        return Err(format!(
                            "corrupt baseline column `{col}`: quantile_sketch must be finite and non-decreasing"
                        ));
                    }
                }
                ColumnBaseline::Categorical {
                    missing_rate,
                    frequencies,
                    ..
                } => {
                    if !missing_rate.is_finite() || !(*missing_rate >= 0.0 && *missing_rate <= 1.0)
                    {
                        return Err(format!("invalid missing_rate on baseline column `{col}`"));
                    }
                    if frequencies.values().any(|v| !v.is_finite() || *v < 0.0) {
                        return Err(format!(
                            "corrupt baseline column `{col}`: categorical frequencies must be finite and >= 0"
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ColumnBaseline {
    Numeric {
        non_null_count: usize,
        missing_rate: f64,
        mean: f64,
        std: f64,
        min: f64,
        p05: f64,
        median: f64,
        p95: f64,
        max: f64,
        /// Interior bin cutpoints (length `k - 1` for `k` bins).
        bin_edges: Vec<f64>,
        /// Reference proportions across the `k` bins (sums to 1.0).
        bin_props: Vec<f64>,
        /// Sorted 256-point quantile sketch for fast accurate KS checks.
        quantile_sketch: Vec<f64>,
    },
    Categorical {
        non_null_count: usize,
        missing_rate: f64,
        cardinality: usize,
        frequencies: IndexMap<String, f64>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationSummary {
    pub n_eval: usize,
    pub base_rate: f64,
    pub mean_predicted: f64,
    pub calibration_gap: f64,
    pub brier_score: f64,
    pub ece: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct DriftThresholds {
    pub psi_warn: f64,
    pub psi_fail: f64,
    pub ks_warn: f64,
    pub ks_fail: f64,
    pub js_warn: f64,
    pub js_fail: f64,
    pub missing_warn: f64,
    pub missing_fail: f64,
    pub ece_fail: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Stable,
    Warn,
    Drift,
}

impl Severity {
    pub fn rank(self) -> u8 {
        match self {
            Self::Stable => 0,
            Self::Warn => 1,
            Self::Drift => 2,
        }
    }
    pub fn max(self, other: Self) -> Self {
        if self.rank() >= other.rank() {
            self
        } else {
            other
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct FeatureDrift {
    pub feature: String,
    pub kind: String,
    pub severity: Severity,
    pub psi: f64,
    pub ks_stat: Option<f64>,
    pub ks_p_value: Option<f64>,
    pub js_divergence: f64,
    pub ref_missing_rate: f64,
    pub cur_missing_rate: f64,
    pub missing_delta: f64,
    pub ref_mean: Option<f64>,
    pub cur_mean: Option<f64>,
    pub mean_z_shift: Option<f64>,
    pub unseen_categories: Vec<String>,
    pub reasons: Vec<String>,
    pub bin_comparison: Vec<BinComparison>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BinComparison {
    pub label: String,
    pub ref_pct: f64,
    pub cur_pct: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CalibrationDrift {
    pub severity: Severity,
    pub current: CalibrationSummary,
    pub reference: Option<CalibrationSummary>,
    pub ece_delta: Option<f64>,
    pub brier_delta: Option<f64>,
    pub base_rate_delta: Option<f64>,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DriftReport {
    pub tool: String,
    pub version: String,
    pub generated_at: String,
    pub reference_source: String,
    pub current_source: String,
    pub reference_rows: usize,
    pub current_rows: usize,
    pub verdict: Severity,
    pub features_checked: usize,
    pub features_drifted: usize,
    pub features_warned: usize,
    pub max_psi: f64,
    pub max_ks: f64,
    pub max_js: f64,
    pub max_missing_delta: f64,
    pub sha256: String,
    pub features: Vec<FeatureDrift>,
    pub prediction_drift: Option<FeatureDrift>,
    pub calibration_drift: Option<CalibrationDrift>,
    pub recommendations: Vec<String>,
}

fn r4(v: f64) -> f64 {
    (v * 10_000.0).round() / 10_000.0
}

fn r6(v: f64) -> f64 {
    (v * 1_000_000.0).round() / 1_000_000.0
}

fn parse_numeric_col(vals: &[Option<String>]) -> (Vec<f64>, usize) {
    let mut nums = Vec::new();
    let mut non_null = 0usize;
    for s in vals.iter().flatten() {
        non_null += 1;
        if let Ok(v) = s.parse::<f64>()
            && v.is_finite()
        {
            nums.push(v);
        }
    }
    (nums, non_null)
}

fn is_numeric_column(name: &str, vals: &[Option<String>], forced_cat: &HashSet<String>) -> bool {
    if forced_cat.contains(name) {
        return false;
    }
    let (nums, non_null) = parse_numeric_col(vals);
    if non_null == 0 {
        return false;
    }
    (nums.len() as f64) / (non_null as f64) >= 0.90
}

fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let pos = q.clamp(0.0, 1.0) * ((sorted.len() - 1) as f64);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let a = sorted[lo];
    let b = sorted[hi];
    if lo == hi || a == b {
        a
    } else {
        let w = pos - (lo as f64);
        (a + (b - a) * w).clamp(a, b)
    }
}

fn build_quantile_edges(sorted: &[f64], bins: usize) -> Vec<f64> {
    let b = bins.clamp(2, 50);
    if sorted.is_empty() {
        return Vec::new();
    }
    // Zero-variance / constant reference column: emit tight guard edges `[c - 1e-9, c + 1e-9]`
    // so identical values stay in the center bin (`PSI = 0`) while out-of-point shifts trigger drift!
    if sorted.len() < 2 || (sorted[0] - sorted[sorted.len() - 1]).abs() <= 1e-12 {
        let c = sorted[0];
        let delta = (c.abs() * 1e-9).max(1e-9);
        return vec![c - delta, c + delta];
    }
    let mut edges: Vec<f64> = Vec::new();
    for i in 1..b {
        let q = (i as f64) / (b as f64);
        let e = quantile_sorted(sorted, q);
        if edges.last().is_none_or(|&prev| e > prev + 1e-12) {
            edges.push(e);
        }
    }
    edges
}

fn bin_proportions(values: &[f64], edges: &[f64]) -> Vec<f64> {
    let k = edges.len() + 1;
    let mut counts = vec![0usize; k];
    for &v in values {
        let idx = edges.partition_point(|&e| v > e);
        counts[idx.min(k - 1)] += 1;
    }
    let total = values.len().max(1) as f64;
    counts.into_iter().map(|c| (c as f64) / total).collect()
}

fn build_sketch(sorted: &[f64]) -> Vec<f64> {
    if sorted.is_empty() {
        return Vec::new();
    }
    if sorted.len() <= SKETCH_SIZE {
        return sorted.to_vec();
    }
    (0..SKETCH_SIZE)
        .map(|i| {
            let q = (i as f64) / ((SKETCH_SIZE - 1) as f64);
            quantile_sorted(sorted, q)
        })
        .collect()
}

pub fn compute_psi(p: &[f64], q: &[f64]) -> f64 {
    let mut total = 0.0_f64;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        let ps = pi.max(EPS);
        let qs = qi.max(EPS);
        total += (qs - ps) * (qs / ps).ln();
    }
    total.max(0.0)
}

/// Base-2 Jensen-Shannon divergence bounded in [0.0, 1.0].
pub fn compute_js_divergence(p: &[f64], q: &[f64]) -> f64 {
    let sum_p: f64 = p.iter().sum::<f64>().max(1e-12);
    let sum_q: f64 = q.iter().sum::<f64>().max(1e-12);
    let mut kl_pm = 0.0_f64;
    let mut kl_qm = 0.0_f64;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        let pn = pi / sum_p;
        let qn = qi / sum_q;
        let m = 0.5 * (pn + qn);
        if pn > 0.0 && m > 0.0 {
            kl_pm += pn * (pn / m).log2();
        }
        if qn > 0.0 && m > 0.0 {
            kl_qm += qn * (qn / m).log2();
        }
    }
    (0.5 * (kl_pm + kl_qm)).clamp(0.0, 1.0)
}

/// Exact two-sample Kolmogorov-Smirnov supremum statistic D between two sorted slices.
pub fn ks_two_sample(ref_sorted: &[f64], cur_sorted: &[f64]) -> f64 {
    if ref_sorted.is_empty() || cur_sorted.is_empty() {
        return 0.0;
    }
    let n1 = ref_sorted.len() as f64;
    let n2 = cur_sorted.len() as f64;
    let mut i = 0usize;
    let mut j = 0usize;
    let mut max_d = 0.0_f64;

    while i < ref_sorted.len() && j < cur_sorted.len() {
        let v1 = ref_sorted[i];
        let v2 = cur_sorted[j];
        let x = v1.min(v2);
        while i < ref_sorted.len() && ref_sorted[i] <= x {
            i += 1;
        }
        while j < cur_sorted.len() && cur_sorted[j] <= x {
            j += 1;
        }
        let d = ((i as f64) / n1 - (j as f64) / n2).abs();
        if d > max_d {
            max_d = d;
        }
    }
    max_d
}

/// Asymptotic two-sample KS p-value via Kolmogorov distribution series.
pub fn ks_p_value(d: f64, n1: usize, n2: usize) -> f64 {
    if d <= 0.0 || n1 == 0 || n2 == 0 {
        return 1.0;
    }
    let en = ((n1 as f64) * (n2 as f64) / ((n1 + n2) as f64)).sqrt();
    let lambda = (en + 0.12 + 0.11 / en) * d;
    if lambda <= 0.0 {
        return 1.0;
    }
    let mut sum = 0.0_f64;
    for k in 1..=100 {
        let kf = k as f64;
        let term = (-2.0 * kf * kf * lambda * lambda).exp();
        if k % 2 == 1 {
            sum += term;
        } else {
            sum -= term;
        }
        if term < 1e-12 {
            break;
        }
    }
    (2.0 * sum).clamp(0.0, 1.0)
}

pub fn compute_calibration(
    table: &DataTable,
    score_col: &str,
    label_col: &str,
) -> Option<CalibrationSummary> {
    let scores = table.columns.get(score_col)?;
    let labels = table.columns.get(label_col)?;
    let mut pairs: Vec<(f64, f64)> = Vec::new();
    for (s_opt, y_opt) in scores.iter().zip(labels.iter()) {
        if let (Some(ss), Some(ys)) = (s_opt, y_opt)
            && let (Ok(p), Ok(y)) = (ss.parse::<f64>(), ys.parse::<f64>())
            && p.is_finite()
            && y.is_finite()
        {
            pairs.push((p.clamp(0.0, 1.0), if y >= 0.5 { 1.0 } else { 0.0 }));
        }
    }
    if pairs.is_empty() {
        return None;
    }
    let n = pairs.len() as f64;
    let base_rate = pairs.iter().map(|&(_, y)| y).sum::<f64>() / n;
    let mean_predicted = pairs.iter().map(|&(p, _)| p).sum::<f64>() / n;
    let brier = pairs.iter().map(|&(p, y)| (p - y).powi(2)).sum::<f64>() / n;

    // 10 equal-width bins on [0, 1] for ECE
    let mut bin_count = [0usize; 10];
    let mut bin_p_sum = [0.0_f64; 10];
    let mut bin_y_sum = [0.0_f64; 10];
    for &(p, y) in &pairs {
        let b = ((p * 10.0).floor() as usize).min(9);
        bin_count[b] += 1;
        bin_p_sum[b] += p;
        bin_y_sum[b] += y;
    }
    let mut ece = 0.0_f64;
    for b in 0..10 {
        if bin_count[b] > 0 {
            let cnt = bin_count[b] as f64;
            ece += (cnt / n) * ((bin_p_sum[b] / cnt) - (bin_y_sum[b] / cnt)).abs();
        }
    }

    Some(CalibrationSummary {
        n_eval: pairs.len(),
        base_rate: r6(base_rate),
        mean_predicted: r6(mean_predicted),
        calibration_gap: r6(mean_predicted - base_rate),
        brier_score: r6(brier),
        ece: r6(ece),
    })
}

pub fn build_snapshot(
    table: &DataTable,
    source_name: &str,
    bins: usize,
    score_col: Option<&str>,
    label_col: Option<&str>,
    ignore: &[String],
    categorical: &[String],
) -> BaselineSnapshot {
    let ignore_set: HashSet<String> = ignore.iter().cloned().collect();
    let forced_cat: HashSet<String> = categorical.iter().cloned().collect();
    let mut columns = IndexMap::new();

    for (col_name, vals) in &table.columns {
        if ignore_set.contains(col_name) {
            continue;
        }
        let n_rows = table.row_count.max(1);
        if is_numeric_column(col_name, vals, &forced_cat) {
            let (mut nums, _non_null) = parse_numeric_col(vals);
            nums.sort_by(|a, b| a.total_cmp(b));
            // Count only finite valid numeric values toward completeness so corrupted/Inf cells count as missing
            let missing_rate = ((n_rows.saturating_sub(nums.len())) as f64) / (n_rows as f64);
            let mean = if nums.is_empty() {
                0.0
            } else {
                nums.iter().sum::<f64>() / (nums.len() as f64)
            };
            let var = if nums.len() > 1 {
                nums.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (nums.len() as f64)
            } else {
                0.0
            };
            let std = var.sqrt();
            let edges = build_quantile_edges(&nums, bins);
            let props = bin_proportions(&nums, &edges);
            let sketch = build_sketch(&nums);
            columns.insert(
                col_name.clone(),
                ColumnBaseline::Numeric {
                    non_null_count: nums.len(),
                    missing_rate: r6(missing_rate),
                    mean: r6(mean),
                    std: r6(std),
                    min: r6(nums.first().copied().unwrap_or(0.0)),
                    p05: r6(quantile_sorted(&nums, 0.05)),
                    median: r6(quantile_sorted(&nums, 0.50)),
                    p95: r6(quantile_sorted(&nums, 0.95)),
                    max: r6(nums.last().copied().unwrap_or(0.0)),
                    bin_edges: edges,
                    bin_props: props,
                    quantile_sketch: sketch,
                },
            );
        } else {
            let mut counts: HashMap<String, usize> = HashMap::new();
            let mut non_null = 0usize;
            for v in vals.iter().flatten() {
                non_null += 1;
                *counts.entry(v.clone()).or_insert(0) += 1;
            }
            let missing_rate = ((n_rows.saturating_sub(non_null)) as f64) / (n_rows as f64);
            let mut entries: Vec<(String, usize)> = counts.into_iter().collect();
            entries.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            let denom = non_null.max(1) as f64;
            let cardinality = entries.len();
            let mut freqs = IndexMap::new();
            let mut other_sum = 0usize;
            const MAX_TRACKED_CATEGORIES: usize = 2000;
            for (idx, (cat, cnt)) in entries.into_iter().enumerate() {
                if idx < MAX_TRACKED_CATEGORIES {
                    freqs.insert(cat, r6((cnt as f64) / denom));
                } else {
                    other_sum += cnt;
                }
            }
            if other_sum > 0 {
                freqs.insert("__OTHER__".to_string(), r6((other_sum as f64) / denom));
            }
            columns.insert(
                col_name.clone(),
                ColumnBaseline::Categorical {
                    non_null_count: non_null,
                    missing_rate: r6(missing_rate),
                    cardinality,
                    frequencies: freqs,
                },
            );
        }
    }

    let calibration_baseline = match (score_col, label_col) {
        (Some(sc), Some(lc)) => compute_calibration(table, sc, lc),
        _ => None,
    };

    let mut snap = BaselineSnapshot {
        schema_version: 1,
        tool: "driftkeel".to_string(),
        tool_version: env!("CARGO_PKG_VERSION").to_string(),
        created_at: Utc::now().to_rfc3339(),
        source_name: source_name.to_string(),
        row_count: table.row_count,
        bins: bins.clamp(2, 50),
        sha256: String::new(),
        score_col: score_col.map(|s| s.to_string()),
        label_col: label_col.map(|s| s.to_string()),
        columns,
        calibration_baseline,
    };
    let mut hasher = Sha256::new();
    hasher.update(
        serde_json::to_string(&snap.columns)
            .unwrap_or_default()
            .as_bytes(),
    );
    snap.sha256 = hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    snap
}

#[allow(clippy::too_many_arguments)]
pub fn evaluate_drift(
    baseline: &BaselineSnapshot,
    raw_ref_table: Option<&DataTable>,
    current: &DataTable,
    current_source: &str,
    score_col: Option<&str>,
    label_col: Option<&str>,
    ignore: &[String],
    thresholds: DriftThresholds,
) -> DriftReport {
    let ignore_set: HashSet<String> = ignore.iter().cloned().collect();
    let eff_score = score_col.or(baseline.score_col.as_deref());
    let eff_label = label_col.or(baseline.label_col.as_deref());

    let mut features = Vec::new();
    let mut prediction_drift = None;

    for (col_name, col_base) in &baseline.columns {
        if ignore_set.contains(col_name) {
            continue;
        }
        if eff_label == Some(col_name.as_str()) {
            continue;
        }
        let Some(cur_vals) = current.columns.get(col_name) else {
            let fd = FeatureDrift {
                feature: col_name.clone(),
                kind: match col_base {
                    ColumnBaseline::Numeric { .. } => "numeric".to_string(),
                    ColumnBaseline::Categorical { .. } => "categorical".to_string(),
                },
                severity: Severity::Drift,
                psi: 1.0,
                ks_stat: None,
                ks_p_value: None,
                js_divergence: 1.0,
                ref_missing_rate: match col_base {
                    ColumnBaseline::Numeric { missing_rate, .. } => *missing_rate,
                    ColumnBaseline::Categorical { missing_rate, .. } => *missing_rate,
                },
                cur_missing_rate: 1.0,
                missing_delta: 1.0,
                ref_mean: None,
                cur_mean: None,
                mean_z_shift: None,
                unseen_categories: Vec::new(),
                reasons: vec!["COLUMN_MISSING: feature absent from current dataset".to_string()],
                bin_comparison: Vec::new(),
            };
            if eff_score == Some(col_name.as_str()) {
                prediction_drift = Some(fd);
            } else {
                features.push(fd);
            }
            continue;
        };

        let raw_ref_col = raw_ref_table.and_then(|t| t.columns.get(col_name));
        let fd = compare_column(
            col_name,
            col_base,
            raw_ref_col.map(|v| v.as_slice()),
            cur_vals,
            current.row_count,
            thresholds,
        );
        if eff_score == Some(col_name.as_str()) {
            prediction_drift = Some(fd);
        } else {
            features.push(fd);
        }
    }

    features.sort_by(|a, b| {
        b.severity
            .rank()
            .cmp(&a.severity.rank())
            .then_with(|| b.psi.total_cmp(&a.psi))
            .then_with(|| a.feature.cmp(&b.feature))
    });

    let calibration_drift = match (eff_score, eff_label) {
        (Some(sc), Some(lc)) => compute_calibration(current, sc, lc).map(|cur_cal| {
            let mut sev = Severity::Stable;
            let mut reasons = Vec::new();
            if cur_cal.ece >= thresholds.ece_fail {
                sev = Severity::Drift;
                reasons.push(format!(
                    "CALIBRATION_ECE: current ECE {:.4} >= {:.4}",
                    cur_cal.ece, thresholds.ece_fail
                ));
            } else if cur_cal.ece >= thresholds.ece_fail * 0.65 {
                sev = sev.max(Severity::Warn);
                reasons.push(format!(
                    "CALIBRATION_ECE_WARN: current ECE {:.4}",
                    cur_cal.ece
                ));
            }
            let (ece_delta, brier_delta, base_rate_delta) =
                if let Some(ref_cal) = &baseline.calibration_baseline {
                    let ed = r6(cur_cal.ece - ref_cal.ece);
                    let bd = r6(cur_cal.brier_score - ref_cal.brier_score);
                    let brd = r6(cur_cal.base_rate - ref_cal.base_rate);
                    if brd.abs() >= 0.08 {
                        sev = sev.max(Severity::Warn);
                        reasons.push(format!(
                            "BASE_RATE_SHIFT: observed positive rate shifted {:+.1}pp",
                            brd * 100.0
                        ));
                    }
                    (Some(ed), Some(bd), Some(brd))
                } else {
                    (None, None, None)
                };
            CalibrationDrift {
                severity: sev,
                current: cur_cal,
                reference: baseline.calibration_baseline.clone(),
                ece_delta,
                brier_delta,
                base_rate_delta,
                reasons,
            }
        }),
        _ => None,
    };

    let mut verdict = Severity::Stable;
    let mut features_drifted = 0usize;
    let mut features_warned = 0usize;
    let mut max_psi = 0.0_f64;
    let mut max_ks = 0.0_f64;
    let mut max_js = 0.0_f64;
    let mut max_missing_delta = 0.0_f64;

    for f in &features {
        verdict = verdict.max(f.severity);
        match f.severity {
            Severity::Drift => features_drifted += 1,
            Severity::Warn => features_warned += 1,
            Severity::Stable => {}
        }
        max_psi = max_psi.max(f.psi);
        max_ks = max_ks.max(f.ks_stat.unwrap_or(0.0));
        max_js = max_js.max(f.js_divergence);
        max_missing_delta = max_missing_delta.max(f.missing_delta);
    }
    if let Some(pd) = &prediction_drift {
        verdict = verdict.max(pd.severity);
        max_psi = max_psi.max(pd.psi);
        max_ks = max_ks.max(pd.ks_stat.unwrap_or(0.0));
        max_js = max_js.max(pd.js_divergence);
        max_missing_delta = max_missing_delta.max(pd.missing_delta);
    }
    if let Some(cd) = &calibration_drift {
        verdict = verdict.max(cd.severity);
    }

    let mut recommendations = Vec::new();
    if verdict == Severity::Stable {
        recommendations.push(
            "All monitored features and distributions are within healthy bounds; safe to score."
                .to_string(),
        );
    } else {
        for f in features
            .iter()
            .filter(|f| f.severity != Severity::Stable)
            .take(5)
        {
            if f.missing_delta >= thresholds.missing_warn {
                recommendations.push(format!(
                    "Audit upstream ETL join for `{}`: missingness jumped {:+.1}pp ({:.1}% -> {:.1}%).",
                    f.feature,
                    f.missing_delta * 100.0,
                    f.ref_missing_rate * 100.0,
                    f.cur_missing_rate * 100.0
                ));
            } else if !f.unseen_categories.is_empty() {
                recommendations.push(format!(
                    "Map or retrain for unseen `{}` categories: {}.",
                    f.feature,
                    f.unseen_categories.join(", ")
                ));
            } else {
                recommendations.push(format!(
                    "Investigate `{}` distribution shift (PSI={:.3}, JS={:.3}).",
                    f.feature, f.psi, f.js_divergence
                ));
            }
        }
        if let Some(pd) = &prediction_drift
            && pd.severity != Severity::Stable
        {
            recommendations.push(format!(
                "Prediction score `{}` drifted (PSI={:.3}); verify threshold cutoffs and advisor alert volume.",
                pd.feature, pd.psi
            ));
        }
        if let Some(cd) = &calibration_drift
            && cd.severity != Severity::Stable
        {
            recommendations.push(format!(
                "Recalibrate model probabilities (Platt/isotonic): current ECE={:.4}, calibration gap={:+.3}.",
                cd.current.ece, cd.current.calibration_gap
            ));
        }
    }

    let mut hasher = Sha256::new();
    hasher.update(baseline.sha256.as_bytes());
    hasher.update(current_source.as_bytes());
    hasher.update(format!("{}:{:.6}:{:.6}", current.row_count, max_psi, max_ks).as_bytes());
    let sha256: String = hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();

    DriftReport {
        tool: "driftkeel".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        generated_at: Utc::now().to_rfc3339(),
        reference_source: baseline.source_name.clone(),
        current_source: current_source.to_string(),
        reference_rows: baseline.row_count,
        current_rows: current.row_count,
        verdict,
        features_checked: features.len(),
        features_drifted,
        features_warned,
        max_psi: r4(max_psi),
        max_ks: r4(max_ks),
        max_js: r4(max_js),
        max_missing_delta: r4(max_missing_delta),
        sha256,
        features,
        prediction_drift,
        calibration_drift,
        recommendations,
    }
}

fn compare_column(
    name: &str,
    base: &ColumnBaseline,
    raw_ref_vals: Option<&[Option<String>]>,
    cur_vals: &[Option<String>],
    cur_row_count: usize,
    t: DriftThresholds,
) -> FeatureDrift {
    let mut severity = Severity::Stable;
    let mut reasons = Vec::new();
    let n_cur = cur_row_count.max(1);

    match base {
        ColumnBaseline::Numeric {
            non_null_count: ref_n,
            missing_rate: ref_miss,
            mean: ref_mean,
            std: ref_std,
            bin_edges,
            bin_props,
            quantile_sketch,
            ..
        } => {
            let (mut cur_nums, cur_non_null) = parse_numeric_col(cur_vals);
            let invalid_numeric = cur_non_null.saturating_sub(cur_nums.len());
            cur_nums.sort_by(|a, b| a.total_cmp(b));
            // Count non-finite / unparseable numeric cells as missing so corrupted upstream strings trigger alerts!
            let cur_miss = ((n_cur.saturating_sub(cur_nums.len())) as f64) / (n_cur as f64);
            let miss_delta = cur_miss - ref_miss;
            if invalid_numeric > 0 {
                let bad_rate = (invalid_numeric as f64) / (n_cur as f64);
                reasons.push(format!(
                    "INVALID_NUMERIC: {invalid_numeric} unparseable/non-finite value(s) ({:.1}% of rows)",
                    bad_rate * 100.0
                ));
                if bad_rate >= t.missing_fail {
                    severity = severity.max(Severity::Drift);
                } else if bad_rate >= t.missing_warn {
                    severity = severity.max(Severity::Warn);
                }
            }

            let cur_props = bin_proportions(&cur_nums, bin_edges);
            let psi = if cur_nums.is_empty() {
                1.0
            } else {
                compute_psi(bin_props, &cur_props)
            };
            let js = if cur_nums.is_empty() {
                1.0
            } else {
                compute_js_divergence(bin_props, &cur_props)
            };

            let ref_sample: Vec<f64> = if let Some(raw_ref) = raw_ref_vals {
                let (mut v, _) = parse_numeric_col(raw_ref);
                v.sort_by(|a, b| a.total_cmp(b));
                v
            } else {
                quantile_sketch.clone()
            };
            let ks = ks_two_sample(&ref_sample, &cur_nums);
            let ks_p = ks_p_value(ks, *ref_n, cur_nums.len());

            let cur_mean = if cur_nums.is_empty() {
                0.0
            } else {
                cur_nums.iter().sum::<f64>() / (cur_nums.len() as f64)
            };
            let z_shift = if *ref_std > 1e-9 {
                (cur_mean - ref_mean) / ref_std
            } else {
                0.0
            };

            if psi >= t.psi_fail {
                severity = severity.max(Severity::Drift);
                reasons.push(format!("PSI={psi:.3} >= {:.2}", t.psi_fail));
            } else if psi >= t.psi_warn {
                severity = severity.max(Severity::Warn);
                reasons.push(format!("PSI={psi:.3} >= {:.2}", t.psi_warn));
            }
            if ks >= t.ks_fail {
                severity = severity.max(Severity::Drift);
                reasons.push(format!("KS={ks:.3} >= {:.2}", t.ks_fail));
            } else if ks >= t.ks_warn {
                severity = severity.max(Severity::Warn);
                reasons.push(format!("KS={ks:.3} >= {:.2}", t.ks_warn));
            }
            if js >= t.js_fail {
                severity = severity.max(Severity::Drift);
                reasons.push(format!("JS={js:.3} >= {:.2}", t.js_fail));
            } else if js >= t.js_warn {
                severity = severity.max(Severity::Warn);
                reasons.push(format!("JS={js:.3} >= {:.2}", t.js_warn));
            }
            if miss_delta >= t.missing_fail {
                severity = severity.max(Severity::Drift);
                reasons.push(format!(
                    "MISSING_SPIKE={:+.1}pp >= {:.1}pp",
                    miss_delta * 100.0,
                    t.missing_fail * 100.0
                ));
            } else if miss_delta >= t.missing_warn {
                severity = severity.max(Severity::Warn);
                reasons.push(format!(
                    "MISSING_WARN={:+.1}pp >= {:.1}pp",
                    miss_delta * 100.0,
                    t.missing_warn * 100.0
                ));
            }

            let mut bin_comp = Vec::new();
            for (idx, (&rp, &cp)) in bin_props.iter().zip(cur_props.iter()).enumerate() {
                let lbl = if bin_edges.is_empty() {
                    "all".to_string()
                } else if idx == 0 {
                    format!("<= {:.2}", bin_edges[0])
                } else if idx == bin_edges.len() {
                    format!("> {:.2}", bin_edges[idx - 1])
                } else {
                    format!("{:.2}..{:.2}", bin_edges[idx - 1], bin_edges[idx])
                };
                bin_comp.push(BinComparison {
                    label: lbl,
                    ref_pct: r4(rp * 100.0),
                    cur_pct: r4(cp * 100.0),
                });
            }

            FeatureDrift {
                feature: name.to_string(),
                kind: "numeric".to_string(),
                severity,
                psi: r4(psi),
                ks_stat: Some(r4(ks)),
                ks_p_value: Some(r4(ks_p)),
                js_divergence: r4(js),
                ref_missing_rate: r4(*ref_miss),
                cur_missing_rate: r4(cur_miss),
                missing_delta: r4(miss_delta),
                ref_mean: Some(r4(*ref_mean)),
                cur_mean: Some(r4(cur_mean)),
                mean_z_shift: Some(r4(z_shift)),
                unseen_categories: Vec::new(),
                reasons,
                bin_comparison: bin_comp,
            }
        }
        ColumnBaseline::Categorical {
            missing_rate: ref_miss,
            frequencies: ref_freqs,
            ..
        } => {
            let mut cur_counts: HashMap<String, usize> = HashMap::new();
            let mut cur_non_null = 0usize;
            for v in cur_vals.iter().flatten() {
                cur_non_null += 1;
                *cur_counts.entry(v.clone()).or_insert(0) += 1;
            }
            let cur_miss = ((n_cur.saturating_sub(cur_non_null)) as f64) / (n_cur as f64);
            let miss_delta = cur_miss - ref_miss;
            let denom = cur_non_null.max(1) as f64;

            let has_other_bucket = ref_freqs.contains_key("__OTHER__");
            let mut all_keys: Vec<String> = ref_freqs.keys().cloned().collect();
            let mut unseen = Vec::new();
            let mut other_cur_count = 0usize;
            for (k, &cnt) in &cur_counts {
                if !ref_freqs.contains_key(k) {
                    if has_other_bucket {
                        // Reference truncated tail categories into `__OTHER__`: fold tail into
                        // `__OTHER__` instead of false-flagging reference tail levels as unseen!
                        other_cur_count += cnt;
                    } else {
                        unseen.push(k.clone());
                        all_keys.push(k.clone());
                    }
                }
            }
            unseen.sort();

            let mut p_vec = Vec::with_capacity(all_keys.len());
            let mut q_vec = Vec::with_capacity(all_keys.len());
            let mut bin_comp = Vec::new();
            for k in &all_keys {
                let rp = ref_freqs.get(k).copied().unwrap_or(0.0);
                let cp = if has_other_bucket && k == "__OTHER__" {
                    let direct = cur_counts.get(k).copied().unwrap_or(0);
                    ((direct + other_cur_count) as f64) / denom
                } else {
                    cur_counts
                        .get(k)
                        .map(|&c| (c as f64) / denom)
                        .unwrap_or(0.0)
                };
                p_vec.push(rp);
                q_vec.push(cp);
                bin_comp.push(BinComparison {
                    label: k.clone(),
                    ref_pct: r4(rp * 100.0),
                    cur_pct: r4(cp * 100.0),
                });
            }
            bin_comp.sort_by(|a, b| {
                b.cur_pct
                    .max(b.ref_pct)
                    .total_cmp(&a.cur_pct.max(a.ref_pct))
            });
            bin_comp.truncate(12);

            let psi = compute_psi(&p_vec, &q_vec);
            let js = compute_js_divergence(&p_vec, &q_vec);

            if psi >= t.psi_fail {
                severity = severity.max(Severity::Drift);
                reasons.push(format!("PSI={psi:.3} >= {:.2}", t.psi_fail));
            } else if psi >= t.psi_warn {
                severity = severity.max(Severity::Warn);
                reasons.push(format!("PSI={psi:.3} >= {:.2}", t.psi_warn));
            }
            if js >= t.js_fail {
                severity = severity.max(Severity::Drift);
                reasons.push(format!("JS={js:.3} >= {:.2}", t.js_fail));
            } else if js >= t.js_warn {
                severity = severity.max(Severity::Warn);
                reasons.push(format!("JS={js:.3} >= {:.2}", t.js_warn));
            }
            if miss_delta >= t.missing_fail {
                severity = severity.max(Severity::Drift);
                reasons.push(format!(
                    "MISSING_SPIKE={:+.1}pp >= {:.1}pp",
                    miss_delta * 100.0,
                    t.missing_fail * 100.0
                ));
            } else if miss_delta >= t.missing_warn {
                severity = severity.max(Severity::Warn);
                reasons.push(format!(
                    "MISSING_WARN={:+.1}pp >= {:.1}pp",
                    miss_delta * 100.0,
                    t.missing_warn * 100.0
                ));
            }
            if !unseen.is_empty() {
                let unseen_share: f64 = unseen
                    .iter()
                    .map(|k| cur_counts.get(k).copied().unwrap_or(0) as f64 / denom)
                    .sum();
                if unseen_share >= 0.05 {
                    severity = severity.max(Severity::Drift);
                } else {
                    severity = severity.max(Severity::Warn);
                }
                reasons.push(format!(
                    "UNSEEN_CATEGORIES({}): {} ({:.1}% of rows)",
                    unseen.len(),
                    unseen
                        .iter()
                        .take(4)
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(", "),
                    unseen_share * 100.0
                ));
            }

            FeatureDrift {
                feature: name.to_string(),
                kind: "categorical".to_string(),
                severity,
                psi: r4(psi),
                ks_stat: None,
                ks_p_value: None,
                js_divergence: r4(js),
                ref_missing_rate: r4(*ref_miss),
                cur_missing_rate: r4(cur_miss),
                missing_delta: r4(miss_delta),
                ref_mean: None,
                cur_mean: None,
                mean_z_shift: None,
                unseen_categories: unseen,
                reasons,
                bin_comparison: bin_comp,
            }
        }
    }
}
