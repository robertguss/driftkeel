use std::io::IsTerminal;
use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum OutputFormat {
    Auto,
    Human,
    Json,
    Agent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedFormat {
    Human,
    Json,
    Agent,
}

impl OutputFormat {
    pub fn resolve(self) -> ResolvedFormat {
        match self {
            Self::Human => ResolvedFormat::Human,
            Self::Json => ResolvedFormat::Json,
            Self::Agent => ResolvedFormat::Agent,
            Self::Auto => {
                if std::io::stdout().is_terminal() {
                    ResolvedFormat::Human
                } else {
                    ResolvedFormat::Json
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum FailOn {
    Error,
    Warn,
}

#[derive(Debug, Parser)]
#[command(
    name = "driftkeel",
    version,
    about = "Zero-Python Rust ML feature, prediction, and concept drift monitor (PSI, KS, JS, missingness, calibration ECE & HTML receipts)."
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Snapshot a reference dataset (CSV, TSV, JSONL, Parquet) into a compact baseline JSON
    Snapshot {
        /// Reference dataset file (.csv, .tsv, .jsonl, .ndjson, .parquet)
        input: PathBuf,
        /// Output baseline JSON path
        #[arg(short, long)]
        out: PathBuf,
        /// Number of quantile bins for numeric features
        #[arg(long, default_value_t = 10)]
        bins: usize,
        /// Optional prediction score column name
        #[arg(long)]
        score_col: Option<String>,
        /// Optional binary ground-truth label column (0/1) for baseline calibration stats
        #[arg(long)]
        label_col: Option<String>,
        /// Comma-separated columns to ignore (e.g. IDs, timestamps)
        #[arg(long, value_delimiter = ',')]
        ignore: Vec<String>,
        /// Comma-separated columns to force as categorical even if numeric-looking
        #[arg(long, value_delimiter = ',')]
        categorical: Vec<String>,
        /// Output format (auto, human, json, agent)
        #[arg(long, value_enum, default_value_t = OutputFormat::Auto)]
        format: OutputFormat,
    },

    /// Compare a live batch against a saved baseline JSON or raw reference dataset
    Check {
        /// Saved baseline .json OR raw reference dataset (.csv/.tsv/.jsonl/.parquet)
        reference: PathBuf,
        /// Current/production dataset (.csv/.tsv/.jsonl/.parquet)
        current: PathBuf,
        /// Number of quantile bins if building baseline from raw reference dataset
        #[arg(long, default_value_t = 10)]
        bins: usize,
        /// Optional prediction score column name
        #[arg(long)]
        score_col: Option<String>,
        /// Optional binary ground-truth label column (0/1) for concept/calibration drift
        #[arg(long)]
        label_col: Option<String>,
        /// Comma-separated columns to ignore
        #[arg(long, value_delimiter = ',')]
        ignore: Vec<String>,
        /// Comma-separated columns to force as categorical
        #[arg(long, value_delimiter = ',')]
        categorical: Vec<String>,
        /// PSI warning threshold
        #[arg(long, default_value_t = 0.10)]
        psi_warn: f64,
        /// PSI failure threshold
        #[arg(long, default_value_t = 0.25)]
        psi_fail: f64,
        /// Two-sample KS warning threshold
        #[arg(long, default_value_t = 0.10)]
        ks_warn: f64,
        /// Two-sample KS failure threshold
        #[arg(long, default_value_t = 0.15)]
        ks_fail: f64,
        /// Jensen-Shannon divergence (base-2) warning threshold
        #[arg(long, default_value_t = 0.08)]
        js_warn: f64,
        /// Jensen-Shannon divergence (base-2) failure threshold
        #[arg(long, default_value_t = 0.15)]
        js_fail: f64,
        /// Absolute missing-rate increase warning threshold
        #[arg(long, default_value_t = 0.03)]
        missing_warn: f64,
        /// Absolute missing-rate increase failure threshold
        #[arg(long, default_value_t = 0.05)]
        missing_fail: f64,
        /// Calibration ECE failure threshold (when --score-col and --label-col are provided)
        #[arg(long, default_value_t = 0.08)]
        ece_fail: f64,
        /// Exit code 1 threshold: error (default) or warn
        #[arg(long, value_enum, default_value_t = FailOn::Error)]
        fail_on: FailOn,
        /// Write self-contained zero-CDN offline HTML drift receipt
        #[arg(long)]
        html: Option<PathBuf>,
        /// Output format (auto, human, json, agent)
        #[arg(long, value_enum, default_value_t = OutputFormat::Auto)]
        format: OutputFormat,
    },

    /// Print or install the bundled SKILL.md for AI coding agents
    Skill {
        /// Install SKILL.md into <dir>/driftkeel/SKILL.md (defaults to .agents/skills)
        #[arg(long, num_args = 0..=1, default_missing_value = ".agents/skills")]
        install: Option<PathBuf>,
        /// Output format (kept for Keel stack CLI consistency)
        #[arg(long, value_enum, default_value_t = OutputFormat::Auto)]
        format: OutputFormat,
    },
}
