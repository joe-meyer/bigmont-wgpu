//! The pass/fail decision of `bigmont canary`.
//!
//! Comparing final verdicts alone is not enough: the engine already corrects every GPU probable
//! prime and a sample of GPU composites with the CPU, and a bucket whose GPU run fails is redone on
//! the CPU, so the final verdicts can all be right while the GPU was wrong or never used. The canary
//! therefore also fails on any GPU disagreement or error, on any lost GPU device (even one the
//! engine recovered from), and on a GPU run that tested nothing on the GPU.

use crate::engine::EngineReport;

/// A message when the candidate and expected-verdict counts differ (checked before running the engine).
pub fn length_failure(candidates: usize, expected: usize) -> Option<String> {
    (candidates != expected)
        .then(|| format!("{candidates} candidates but {expected} expected verdicts"))
}

/// Every failed canary condition (empty means the canary passes).
///
/// `got` are the engine's verdicts, `want` the expected ones, `report` the engine's counters for the
/// run and `gpu_active` whether the engine resolved to the GPU backend (`--backend cpu`, or `auto`
/// without a usable GPU, is `false`: then having no GPU tests is expected).
pub fn failures(got: &[u8], want: &[u8], report: &EngineReport, gpu_active: bool) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(msg) = length_failure(got.len(), want.len()) {
        out.push(msg);
        return out;
    }
    let bad: Vec<String> = got
        .iter()
        .zip(want)
        .enumerate()
        .filter(|(_, (g, w))| g != w)
        .map(|(i, _)| i.to_string())
        .collect();
    if !bad.is_empty() {
        let first: Vec<&str> = bad.iter().take(10).map(String::as_str).collect();
        out.push(format!(
            "{} of {} verdicts differ; first mismatching indices: {}",
            bad.len(),
            got.len(),
            first.join(", ")
        ));
    }
    if report.prp_disagreements > 0 {
        out.push(format!(
            "the GPU called {} composite candidates probable primes (PRP disagreements)",
            report.prp_disagreements
        ));
    }
    if report.composite_disagreements > 0 {
        out.push(format!(
            "the GPU called {} probable primes composite (composite disagreements)",
            report.composite_disagreements
        ));
    }
    if report.gpu_errors > 0 {
        out.push(format!(
            "{} buckets hit a GPU error and fell back to the CPU (gpu errors)",
            report.gpu_errors
        ));
    }
    if report.device_losses > 0 {
        // The verdicts survive a recovered loss, but a canary run must show a healthy device.
        let timeouts = if report.poll_timeouts > 0 {
            format!(" ({} poll timeouts)", report.poll_timeouts)
        } else {
            String::new()
        };
        out.push(format!(
            "GPU device lost {} times, recovered {}{timeouts}",
            report.device_losses, report.device_recoveries
        ));
    }
    if gpu_active && report.gpu_tests == 0 {
        out.push(
            "the GPU backend was selected but no candidate was tested on the GPU (gpu tests 0)"
                .to_string(),
        );
    }
    out
}
