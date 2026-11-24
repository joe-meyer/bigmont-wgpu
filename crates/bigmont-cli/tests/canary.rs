//! `bigmont canary` pass/fail logic: the pure decision function, then the real binary's exit codes.

use bigmont_cli::canary::{failures, length_failure};
use bigmont_cli::engine::EngineReport;
use bigmont_cli::formats;
use bigmont_oracle::{XorShift64, fermat_base2, random_odd_modulus};
use num_bigint::BigUint;
use std::path::PathBuf;
use std::process::{Command, Output};

fn healthy_gpu_report() -> EngineReport {
    EngineReport { gpu_tests: 400, rechecks: 12, ..EngineReport::default() }
}

#[test]
fn matching_verdicts_and_a_healthy_gpu_pass() {
    assert!(failures(&[0, 1, 0], &[0, 1, 0], &healthy_gpu_report(), true).is_empty());
}

#[test]
fn a_cpu_run_passes_without_gpu_tests() {
    // --backend cpu (or no GPU in Auto): the GPU-usage condition does not apply.
    let r = EngineReport { cpu_tests: 400, ..EngineReport::default() };
    assert!(failures(&[0, 1], &[0, 1], &r, false).is_empty());
}

#[test]
fn a_differing_verdict_fails_and_names_the_index() {
    let f = failures(&[0, 1, 0], &[0, 0, 0], &healthy_gpu_report(), true);
    assert_eq!(f.len(), 1, "{f:?}");
    assert!(f[0].contains("1 of 3") && f[0].contains("indices: 1"), "{f:?}");
}

#[test]
fn a_length_mismatch_fails() {
    let f = failures(&[0, 1], &[0, 1, 0], &healthy_gpu_report(), true);
    assert_eq!(f.len(), 1, "{f:?}");
    assert!(f[0].contains("2 candidates") && f[0].contains("3 expected"), "{f:?}");
    assert!(length_failure(2, 3).is_some());
    assert!(length_failure(3, 3).is_none());
}

#[test]
fn every_gpu_disagreement_or_error_fails_even_when_the_final_verdicts_match() {
    // The CPU corrects GPU PRPs and sampled composites, and a failed GPU bucket falls back to the
    // CPU, so the final verdicts can all match while the GPU is wrong.
    for (word, r) in [
        ("prp", EngineReport { prp_disagreements: 1, ..healthy_gpu_report() }),
        ("composite", EngineReport { composite_disagreements: 1, ..healthy_gpu_report() }),
        ("error", EngineReport { gpu_errors: 1, ..healthy_gpu_report() }),
    ] {
        let f = failures(&[0, 1], &[0, 1], &r, true);
        assert_eq!(f.len(), 1, "{word}: {f:?}");
        assert!(f[0].to_lowercase().contains(word), "{word}: {f:?}");
        // The same report fails on a CPU-resolved run too (those counters cannot be non-zero there).
        assert_eq!(failures(&[0, 1], &[0, 1], &r, false).len(), 1, "{word}");
    }
}

#[test]
fn a_gpu_run_that_tested_nothing_on_the_gpu_fails() {
    let r = EngineReport { cpu_tests: 400, ..EngineReport::default() };
    let f = failures(&[0, 1], &[0, 1], &r, true);
    assert_eq!(f.len(), 1, "{f:?}");
    assert!(f[0].contains("no candidate was tested on the GPU"), "{f:?}");
}

#[test]
fn all_failing_conditions_are_reported_together() {
    let r = EngineReport { prp_disagreements: 2, composite_disagreements: 3, gpu_errors: 1, ..EngineReport::default() };
    let f = failures(&[1, 1], &[0, 1], &r, true);
    assert_eq!(f.len(), 5, "mismatch, prp, composite, gpu errors, no gpu tests: {f:?}");
}

// ---- the real binary ----

fn tmp(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("bigmont-canary-{}-{name}", std::process::id()))
}

struct Files(Vec<PathBuf>);
impl Drop for Files {
    fn drop(&mut self) {
        for p in &self.0 {
            let _ = std::fs::remove_file(p);
        }
    }
}

/// A 20-candidate file of 128-bit numbers plus the matching verdict file.
fn fixture(tag: &str) -> (PathBuf, PathBuf, Vec<u8>, Files) {
    let mut rng = XorShift64::new(77);
    let ns: Vec<BigUint> = (0..20).map(|_| random_odd_modulus(&mut rng, 128)).collect();
    let want: Vec<u8> = ns.iter().map(|n| u8::from(fermat_base2(n))).collect();
    let (cand, expected) = (tmp(&format!("{tag}-cand.bin")), tmp(&format!("{tag}-expected.bin")));
    formats::write_candidates(&cand, 128, &ns).unwrap();
    formats::write_verdicts(&expected, 128, &want).unwrap();
    let files = Files(vec![cand.clone(), expected.clone()]);
    (cand, expected, want, files)
}

fn canary(cand: &PathBuf, expected: &PathBuf) -> Output {
    Command::new(env!("CARGO_BIN_EXE_bigmont"))
        .args(["canary"])
        .arg(cand)
        .arg(expected)
        .args(["--backend", "cpu", "--no-cache"])
        .output()
        .expect("run bigmont")
}

#[test]
fn canary_exits_0_for_a_matching_verdict_file() {
    let (cand, expected, _, _files) = fixture("ok");
    let out = canary(&cand, &expected);
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(String::from_utf8_lossy(&out.stderr).contains("canary ok"));
}

#[test]
fn canary_exits_1_when_one_verdict_byte_is_flipped() {
    let (cand, expected, want, _files) = fixture("flip");
    let mut flipped = want.clone();
    flipped[7] ^= 1;
    formats::write_verdicts(&expected, 128, &flipped).unwrap();
    let out = canary(&cand, &expected);
    assert_eq!(out.status.code(), Some(1), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(String::from_utf8_lossy(&out.stderr).contains("canary FAILED"));
}

#[test]
fn canary_exits_1_on_a_length_mismatch() {
    let (cand, expected, want, _files) = fixture("len");
    formats::write_verdicts(&expected, 128, &want[..want.len() - 1]).unwrap();
    let out = canary(&cand, &expected);
    assert_eq!(out.status.code(), Some(1), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(String::from_utf8_lossy(&out.stderr).contains("canary FAILED"));
}
