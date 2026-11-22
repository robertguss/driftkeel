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
fn handles_high_cardinality_corrupted_numeric_constant_col_na_and_duplicate_headers() {
    let dir = tempdir().unwrap();

    // 1. 120-category self-comparison MUST be STABLE (exit 0)
    let cats_path = dir.path().join("cats120.csv");
    let mut cats_csv = String::from("code,region\n");
    for i in 0..120 {
        let reg = if i % 2 == 0 { "NA" } else { "None" };
        cats_csv.push_str(&format!("C{i:03},{reg}\n"));
    }
    fs::write(&cats_path, cats_csv).unwrap();
    let cat_out = Command::cargo_bin("driftkeel")
        .unwrap()
        .args([
            "check",
            cats_path.to_str().unwrap(),
            cats_path.to_str().unwrap(),
            "--format",
            "agent",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let cat_json: serde_json::Value = serde_json::from_slice(&cat_out).unwrap();
    assert_eq!(cat_json["verdict"], "stable");
    assert_eq!(cat_json["max_psi"], 0.0);
    // Verify "NA" and "None" were NOT wiped as missing!
    let reg_feat = cat_json["features"]
        .as_array()
        .unwrap()
        .iter()
        .find(|f| f["feature"] == "region")
        .unwrap();
    assert_eq!(reg_feat["ref_missing_rate"], 0.0);

    // 2. Corrupted "ERROR"/"inf" numeric cells trigger missing_delta + DRIFT
    let ref_num = dir.path().join("ref_num.csv");
    let cur_num = dir.path().join("cur_num.csv");
    let mut r_csv = String::from("x,const_col\n");
    let mut c_csv = String::from("x,const_col\n");
    for i in 0..100 {
        r_csv.push_str(&format!("{i},0.0\n"));
        if i < 50 {
            c_csv.push_str("ERROR,1000.0\n");
        } else {
            c_csv.push_str(&format!("{i},1000.0\n"));
        }
    }
    fs::write(&ref_num, r_csv).unwrap();
    fs::write(&cur_num, c_csv).unwrap();
    let num_out = Command::cargo_bin("driftkeel")
        .unwrap()
        .args([
            "check",
            ref_num.to_str().unwrap(),
            cur_num.to_str().unwrap(),
            "--format",
            "agent",
        ])
        .assert()
        .code(1)
        .get_output()
        .stdout
        .clone();
    let nj: serde_json::Value = serde_json::from_slice(&num_out).unwrap();
    assert_eq!(nj["verdict"], "drift");
    assert!(nj["max_missing_delta"].as_f64().unwrap() >= 0.49);
    let const_f = nj["features"]
        .as_array()
        .unwrap()
        .iter()
        .find(|f| f["feature"] == "const_col")
        .unwrap();
    assert!(const_f["psi"].as_f64().unwrap() > 1.0);

    // 3. Duplicate headers rejected with exit 2
    let dup_csv = dir.path().join("dup.csv");
    fs::write(&dup_csv, "x,x\n1,2\n").unwrap();
    Command::cargo_bin("driftkeel")
        .unwrap()
        .args(["check", dup_csv.to_str().unwrap(), dup_csv.to_str().unwrap()])
        .assert()
        .code(2);

    // 4. Corrupt baseline JSON rejected cleanly (not falling through to read_json_array)
    let bad_base = dir.path().join("bad_baseline.json");
    fs::write(&bad_base, r#"{"tool":"driftkeel","schema_version":1,"bins":0}"#).unwrap();
    Command::cargo_bin("driftkeel")
        .unwrap()
        .args(["check", bad_base.to_str().unwrap(), cats_path.to_str().unwrap()])
        .assert()
        .code(2);
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
