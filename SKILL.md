---
name: driftkeel
description: Zero-Python Rust ML feature, prediction, and concept drift monitor computing PSI, two-sample KS, Jensen-Shannon divergence, missingness spikes, unseen categories, and calibration ECE drift with offline HTML receipts.
---

# `driftkeel` — Production ML Feature, Prediction & Concept Drift Monitor

## When to Use
- Detect silent upstream ETL breakage (missing-rate spikes, unseen categorical codes, covariate distribution shifts) **before** stale predictions reach advisors or dashboards.
- Snapshot a training or reference dataset (`CSV`, `TSV`, `JSONL`, `Parquet`) into a compact versioned JSON baseline (`driftkeel snapshot`).
- Gate nightly or weekly inference batches in CI / cron (`driftkeel check baseline.json live_batch.parquet --format agent`) with deterministic exit codes (`0 = stable`, `1 = drift`, `2 = error`).
- Monitor prediction score drift (`--score-col`) and post-hoc calibration/concept drift (`--score-col` + `--label-col`) with ECE & Brier deltas.

## Command Matrix

| Task | Command |
| --- | --- |
| Snapshot a reference dataset into a compact baseline JSON | `driftkeel snapshot train_features.parquet --out baseline.json` |
| Check a live batch against a saved baseline JSON | `driftkeel check baseline.json live_week3.parquet --format agent` |
| Direct dataset-to-dataset drift comparison | `driftkeel check train.csv prod.csv --score-col risk_prob --html drift_report.html` |
| Include calibration / concept drift when labels arrive | `driftkeel check train.csv scored_labeled.csv --score-col risk_prob --label-col had_dfw` |
| Strict CI gate (fail on warnings too) | `driftkeel check baseline.json live.jsonl --fail-on warn --psi-warn 0.10 --psi-fail 0.25` |
| Print / install this skill | `driftkeel skill` / `driftkeel skill --install` |

## Exit Codes
- `0`: Distribution & calibration stable (within configured thresholds).
- `1`: Drift threshold exceeded (at least one feature, score distribution, or calibration check breached `--fail-on`).
- `2`: I/O, schema, or invocation error.
