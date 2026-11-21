# `driftkeel`

**Part of the [Keel Data Stack](https://github.com/robertguss/keel-stack) (Tool #15).**

Zero-Python, single-binary Rust ML **feature, prediction, and concept drift monitor** built for production scoring pipelines and AI agents (DX + AX).

## Why `driftkeel`?
ML models fail silently when an upstream LMS table drops nulls differently, a new degree/program code appears, or an enrollment cohort shifts. `driftkeel` compares live scoring batches (`CSV`, `TSV`, `JSONL`, or `Apache Parquet`) against either a compact saved `baseline.json` snapshot or a raw reference dataset in milliseconds—no Python, Evidently, or SaaS dependencies required.

## What It Checks
- **Population Stability Index (PSI):** Industry-standard binned divergence (`< 0.10` stable, `0.10–0.25` warn, `>= 0.25` significant shift).
- **Two-Sample Kolmogorov-Smirnov (`KS`):** Exact ECDF supremum `D` (from raw reference files or 256-quantile sketch in `baseline.json`).
- **Jensen-Shannon Divergence (`JS`):** Bounded `[0, 1]` symmetric divergence (base-2) for both numeric and categorical features.
- **Missingness Spikes (`Δ missing`):** Immediately catches broken upstream joins or API null floods.
- **Unseen Categories:** Flags brand-new categorical levels never observed in training.
- **Mean Z-Shift:** Standardized mean drift in units of reference standard deviation.
- **Prediction & Calibration/Concept Drift (`--score-col`, `--label-col`):** Tracks predicted score PSI/KS plus post-hoc **ECE (Expected Calibration Error)**, **Brier score**, and positive base-rate shift when outcomes land.
- **Offline HTML Drift Receipt (`--html`):** Self-contained, zero-CDN shareable drift dashboard with SVG distribution bars and SHA-256 fingerprint.

## Quick Start

```bash
# 1. Snapshot your training or reference panel into a compact JSON baseline
driftkeel snapshot train_features.parquet --out baseline.json --score-col risk_prob

# 2. Gate a live scoring batch (exit 0 = stable, exit 1 = drift, exit 2 = error)
driftkeel check baseline.json live_week3.parquet --format agent --html drift_receipt.html

# 3. Direct dataset-to-dataset check with outcome calibration drift
driftkeel check ref_scored.csv prod_scored.csv --score-col risk_prob --label-col had_dfw
```

## Install

```bash
cargo install --git https://github.com/robertguss/driftkeel --locked
```

## License
Dual-licensed under MIT or Apache-2.0.
