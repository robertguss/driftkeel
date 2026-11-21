use std::fs;

use assert_cmd::Command;
use tempfile::tempdir;

fn make_reference_csv(rows: usize) -> String {
    let mut out = String::from("gpa,miss_rate,program,risk_prob,had_dfw\n");
    let progs = ["MDiv", "MAC", "MAR", "ThM"];
    for i in 0..rows {
        let gpa = 2.6 + ((i % 15) as f64) * 0.1;
        let miss = ((i % 10) as f64) * 0.03;
        let prog = progs[i % progs.len()];
        // Calibrated probabilities: 0.20 for low-miss half, 0.50 for high-miss half
        let prob = if i % 2 == 0 { 0.20 } else { 0.50 };
        let y = if i % 2 == 0 {
            if (i / 2) % 5 == 0 { 1 } else { 0 } // exact 20% positive rate
        } else if (i / 2) % 2 == 0 {
            1
        } else {
            0
        }; // exact 50% positive rate
        out.push_str(&format!("{gpa:.2},{miss:.3},{prog},{prob:.3},{y}\n"));
    }
    out
}

fn make_drifted_csv(rows: usize) -> String {
    let mut out = String::from("gpa,miss_rate,program,risk_prob,had_dfw\n");
    let progs = ["MDiv", "MAC", "MABS_NEW", "MABS_NEW"];
    for i in 0..rows {
        // Huge mean shift in gpa, 30% missingness spike in miss_rate, unseen category MABS_NEW, broken calibration!
        let gpa = 1.4 + ((i % 10) as f64) * 0.08;
        let miss = if i % 3 == 0 {
            "".to_string()
        } else {
            format!("{:.3}", 0.35 + ((i % 10) as f64) * 0.04)
        };
        let prog = progs[i % progs.len()];
        let prob = 0.05; // severly under-predicting
        let y = if i % 2 == 0 { 1 } else { 0 }; // 50% actual DFW vs 5% predicted -> huge ECE!
        out.push_str(&format!("{gpa:.2},{miss},{prog},{prob:.3},{y}\n"));
    }
    out
}

#[test]
fn snapshot_and_stable_check_pass_with_exit_zero() {
    let dir = tempdir().unwrap();
    let ref_path = dir.path().join("ref.csv");
    let cur_path = dir.path().join("cur.csv");
    let base_path = dir.path().join("baseline.json");

    fs::write(&ref_path, make_reference_csv(200)).unwrap();
    fs::write(&cur_path, make_reference_csv(200)).unwrap();

    Command::cargo_bin("driftkeel")
        .unwrap()
        .args([
            "snapshot",
            ref_path.to_str().unwrap(),
            "--out",
            base_path.to_str().unwrap(),
            "--score-col",
            "risk_prob",
            "--label-col",
            "had_dfw",
            "--format",
            "agent",
        ])
        .assert()
        .success();

    let out = Command::cargo_bin("driftkeel")
        .unwrap()
        .args([
            "check",
            base_path.to_str().unwrap(),
            cur_path.to_str().unwrap(),
            "--format",
            "agent",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let parsed: serde_json::Value = serde_json::from_slice(&out).unwrap();
    assert_eq!(parsed["verdict"], "stable");
    assert_eq!(parsed["features_drifted"], 0);
    assert!(parsed["max_psi"].as_f64().unwrap() < 0.05);
}

#[test]
fn drifted_batch_triggers_exit_one_and_catches_all_failure_modes() {
    let dir = tempdir().unwrap();
    let ref_path = dir.path().join("ref.csv");
    let cur_path = dir.path().join("drifted.csv");
    let html_path = dir.path().join("receipt.html");

    fs::write(&ref_path, make_reference_csv(240)).unwrap();
    fs::write(&cur_path, make_drifted_csv(240)).unwrap();

    let out = Command::cargo_bin("driftkeel")
        .unwrap()
        .args([
            "check",
            ref_path.to_str().unwrap(),
            cur_path.to_str().unwrap(),
            "--score-col",
            "risk_prob",
            "--label-col",
            "had_dfw",
            "--html",
            html_path.to_str().unwrap(),
            "--format",
            "agent",
        ])
        .assert()
        .code(1)
        .get_output()
        .stdout
        .clone();

    let parsed: serde_json::Value = serde_json::from_slice(&out).unwrap();
    assert_eq!(parsed["verdict"], "drift");
    assert!(parsed["features_drifted"].as_u64().unwrap() >= 3);
    assert!(parsed["max_missing_delta"].as_f64().unwrap() > 0.25);
    assert_eq!(parsed["calibration_drift"]["severity"], "drift");
    assert!(parsed["calibration_drift"]["current"]["ece"].as_f64().unwrap() > 0.30);

    let prog_feat = parsed["features"]
        .as_array()
        .unwrap()
        .iter()
        .find(|f| f["feature"] == "program")
        .unwrap();
    assert!(
        prog_feat["unseen_categories"]
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v == "MABS_NEW")
    );

    let html = fs::read_to_string(&html_path).unwrap();
    assert!(html.contains("driftkeel"));
    assert!(html.contains("MABS_NEW"));
}

#[test]
fn skill_prints_and_installs() {
    let dir = tempdir().unwrap();
    Command::cargo_bin("driftkeel")
        .unwrap()
        .args(["skill", "--install", dir.path().to_str().unwrap(), "--format", "human"])
        .assert()
        .success();
    let installed = dir.path().join("driftkeel/SKILL.md");
    assert!(installed.exists());
    assert!(fs::read_to_string(installed).unwrap().contains("driftkeel"));
}
