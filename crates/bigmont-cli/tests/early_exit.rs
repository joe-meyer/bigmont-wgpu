//! Pure early-exit logic: no GPU needed.

use bigmont_cli::early_exit::{EeOutcome, VerdictSource, run_waves, schedule};
use bigmont_oracle::XorShift64;
use num_bigint::BigUint;
use num_traits::ToPrimitive;
use std::collections::HashSet;

/// Candidate number `i` is `BigUint::from(i)`, so the fake can tell which global index it was asked
/// about. It answers 1 for indices in `prime`, `answer_for_prime` aside, and records every call.
struct FakeSource {
    prime: HashSet<u64>,
    answer_for_prime: u8,
    calls: Vec<Vec<u64>>,
    /// Fail (with this message) on the call with this zero-based number.
    fail_on_call: Option<(usize, String)>,
    /// Add this many (possibly negative) to the length of each reply.
    len_delta: isize,
}

impl FakeSource {
    fn new(prime: &[u64]) -> Self {
        FakeSource {
            prime: prime.iter().copied().collect(),
            answer_for_prime: 1,
            calls: Vec::new(),
            fail_on_call: None,
            len_delta: 0,
        }
    }
    fn answer(&self, i: u64) -> u8 {
        if self.prime.contains(&i) { self.answer_for_prime } else { 0 }
    }
}

impl VerdictSource for FakeSource {
    fn verdicts(&mut self, ns: &[BigUint]) -> Result<Vec<u8>, String> {
        let idx: Vec<u64> = ns.iter().map(|n| n.to_u64().unwrap()).collect();
        let call_no = self.calls.len();
        self.calls.push(idx.clone());
        if let Some((at, msg)) = &self.fail_on_call
            && *at == call_no
        {
            return Err(msg.clone());
        }
        let mut out: Vec<u8> = idx.iter().map(|&i| self.answer(i)).collect();
        if self.len_delta < 0 {
            out.truncate(out.len().saturating_sub(self.len_delta.unsigned_abs()));
        } else {
            out.extend(std::iter::repeat_n(0, self.len_delta as usize));
        }
        Ok(out)
    }
}

fn numbers(count: u64) -> Vec<BigUint> {
    (0..count).map(BigUint::from).collect()
}

fn run(src: &mut FakeSource, sizes: &[u32], chunk: i32, growth: f64, marker: u8) -> Result<EeOutcome, String> {
    let count: u64 = sizes.iter().map(|&s| u64::from(s)).sum();
    let s_max = sizes.iter().copied().max().unwrap_or(0);
    run_waves(src, &numbers(count), sizes, &schedule(chunk, growth, s_max), marker)
}

#[test]
fn schedule_matches_the_reference_semantics() {
    // growth 1.0: constant steps, last one capped at S
    assert_eq!(schedule(6, 1.0, 20), vec![0, 6, 12, 18, 20]);
    // growth 1.15 from chunk 6: steps round_ties_even(6), (6.9), (7.935), (9.12...), ...
    let b = schedule(6, 1.15, 100);
    assert_eq!(b[0], 0);
    assert_eq!(*b.last().unwrap(), 100);
    assert!(b.windows(2).all(|w| w[0] < w[1]));
    assert_eq!(&b[..5], &[0, 6, 13, 21, 30]); // 6, then +7 (6.9), +8 (7.935), +9 (9.12525)
    // chunk <= 0 gives steps of 1
    assert_eq!(schedule(0, 1.15, 3), vec![0, 1, 2, 3]);
    assert_eq!(schedule(-4, 1.0, 2), vec![0, 1, 2]);
    // nothing to test
    assert_eq!(schedule(6, 1.15, 0), vec![0]);
    // round half to even: c = 2.0 -> step 2; c = 2.5 -> step 2 (not 3); c = 3.125 -> step 3
    assert_eq!(schedule(2, 1.25, 7), vec![0, 2, 4, 7]);
}

#[test]
fn schedule_ties_round_to_even_upwards_too() {
    // chunk 2 with growth 1.75: c = 2 -> step 2; c = 3.5 -> step 4 (4 is even); c = 6.125 -> step 6.
    assert_eq!(schedule(2, 1.75, 12), vec![0, 2, 6, 12]);
}

#[test]
fn schedule_survives_extreme_inputs() {
    // A huge chunk is a single wave; growth cannot overflow into a wrong or endless schedule.
    assert_eq!(schedule(i32::MAX, 1.15, 10), vec![0, 10]);
    assert_eq!(schedule(i32::MIN, 1.0, 2), vec![0, 1, 2]);
    let b = schedule(i32::MAX, 1e300, u32::MAX);
    assert_eq!(b[0], 0);
    assert_eq!(*b.last().unwrap(), u32::MAX);
    assert!(b.windows(2).all(|w| w[0] < w[1]));
    assert_eq!(b.len(), 3); // 2^31-1, then +(2^63-1 saturated), capped at u32::MAX
    // NaN or sub-1 growth never grows the step
    assert_eq!(schedule(3, f64::NAN, 7), vec![0, 3, 6, 7]);
    assert_eq!(schedule(3, 0.5, 7), vec![0, 3, 6, 7]);
}

#[test]
fn death_and_survival_across_waves() {
    // sizes [5, 5, 3] at offsets 0, 5, 10; boundaries [0, 2, 4, 5].
    // k0: first prime at position 3 (index 3); the source would also call index 4 prime.
    // k1: no prime. k2: prime at position 0 (index 10).
    let mut src = FakeSource::new(&[3, 4, 10]);
    let out = run(&mut src, &[5, 5, 3], 2, 1.0, 3).unwrap();
    assert_eq!(
        src.calls,
        vec![
            vec![0, 1, 5, 6, 10, 11], // wave 1: every k, positions 0..2
            vec![2, 3, 7, 8],         // wave 2: k2 retired by its hit at position 0
            vec![9],                  // wave 3: k0 retired at wave 2, only k1 left
        ]
    );
    assert_eq!(
        out.verdicts,
        vec![
            0, 0, 0, 1, 3, // k0: tested 0..4, index 4 skipped despite the source saying prime
            0, 0, 0, 0, 0, // k1: survived everything
            1, 0, 3, //       k2: tested 0..2, position 2 skipped
        ]
    );
    assert_eq!(out.tested, 11);
    assert_eq!(out.waves, 3);
    assert_eq!(out.prp, 2);
}

#[test]
fn survivor_equals_a_plain_run_and_a_first_wave_hit_cuts_the_rest() {
    // sizes [6, 4]; boundaries [0, 3, 6]. k0 has primes at positions 1 and 4, k1 has none.
    let mut src = FakeSource::new(&[1, 4]);
    let out = run(&mut src, &[6, 4], 3, 1.0, 9).unwrap();
    // k0 is cut at the end of wave 1 (positions 0..3): position 4 is not tested although prime.
    assert_eq!(&out.verdicts[..6], &[0, 1, 0, 9, 9, 9]);
    // k1 never hits, so it equals what the source says for all of its positions.
    let plain: Vec<u8> = (6..10).map(|i| src.answer(i)).collect();
    assert_eq!(&out.verdicts[6..], &plain[..]);
    assert_eq!(src.calls, vec![vec![0, 1, 2, 6, 7, 8], vec![9]]);
    assert_eq!(out.tested, 7);
    assert_eq!(out.waves, 2);
}

#[test]
fn a_hit_at_the_last_position_of_a_wave_retires_the_k() {
    // sizes [4, 5]; boundaries [0, 2, 4, 5]. k0's prime is the last position of wave 1;
    // k1 (size 5, not a multiple of 2) ends at its size and never hits.
    let mut src = FakeSource::new(&[1]);
    let out = run(&mut src, &[4, 5], 2, 1.0, 3).unwrap();
    assert_eq!(src.calls, vec![vec![0, 1, 4, 5], vec![6, 7], vec![8]]);
    assert_eq!(out.verdicts, vec![0, 1, 3, 3, 0, 0, 0, 0, 0]);
    assert_eq!((out.tested, out.waves, out.prp), (7, 3, 1));

    // A prime at the very last position of a k (also the last of a wave) retires it at the end.
    let mut src = FakeSource::new(&[3]);
    let out = run(&mut src, &[4, 5], 2, 1.0, 3).unwrap();
    assert_eq!(src.calls, vec![vec![0, 1, 4, 5], vec![2, 3, 6, 7], vec![8]]);
    assert_eq!(out.verdicts, vec![0, 0, 0, 1, 0, 0, 0, 0, 0]);
    assert_eq!((out.tested, out.waves, out.prp), (9, 3, 1));
}

#[test]
fn empty_inputs_and_zero_sized_ks() {
    // no candidates, no ks
    let mut src = FakeSource::new(&[]);
    let out = run(&mut src, &[], 6, 1.15, 3).unwrap();
    assert_eq!(out, EeOutcome { verdicts: vec![], tested: 0, waves: 0, prp: 0 });
    assert!(src.calls.is_empty());
    // ks that are all empty
    let out = run(&mut src, &[0, 0], 6, 1.15, 3).unwrap();
    assert_eq!(out, EeOutcome { verdicts: vec![], tested: 0, waves: 0, prp: 0 });
    assert!(src.calls.is_empty());
    // a zero-sized k in the middle neither tests anything nor disturbs its neighbours' offsets
    let mut src = FakeSource::new(&[2]);
    let out = run(&mut src, &[2, 0, 2], 1, 1.0, 3).unwrap();
    assert_eq!(src.calls, vec![vec![0, 2], vec![1]]);
    assert_eq!(out.verdicts, vec![0, 0, 1, 3]);
    assert_eq!((out.tested, out.waves, out.prp), (3, 2, 1));
}

#[test]
fn sizes_that_do_not_sum_to_the_candidate_count_are_an_error() {
    let mut src = FakeSource::new(&[]);
    let err = run_waves(&mut src, &numbers(4), &[3, 4], &schedule(2, 1.0, 4), 3).unwrap_err();
    assert!(err.contains('7') && err.contains('4'), "{err}");
    let err = run_waves(&mut src, &numbers(9), &[3, 4], &schedule(2, 1.0, 4), 3).unwrap_err();
    assert!(err.contains('7') && err.contains('9'), "{err}");
    assert!(src.calls.is_empty(), "nothing may reach the source before the check");
}

#[test]
fn a_source_with_the_wrong_reply_length_is_an_error() {
    for delta in [-1isize, 1] {
        let mut src = FakeSource::new(&[]);
        src.len_delta = delta;
        let err = run(&mut src, &[3, 3], 2, 1.0, 3).unwrap_err();
        assert!(err.contains('4') && err.contains(&(4 + delta).to_string()), "{err}");
    }
}

#[test]
fn a_source_error_is_propagated() {
    for at in [0usize, 1] {
        let mut src = FakeSource::new(&[]);
        src.fail_on_call = Some((at, "device lost".to_string()));
        let err = run(&mut src, &[3, 3], 2, 1.0, 3).unwrap_err();
        assert_eq!(err, "device lost");
        assert_eq!(src.calls.len(), at + 1, "no calls after the failure");
    }
}

#[test]
fn the_marker_is_used_as_given_and_prp_counts_only_ones() {
    for marker in [3u8, 7, 0, 255] {
        let mut src = FakeSource::new(&[0]);
        let out = run(&mut src, &[3, 3], 1, 1.0, marker).unwrap();
        // k0 hits at wave 1, k1 never does
        assert_eq!(out.verdicts, vec![1, marker, marker, 0, 0, 0], "marker {marker}");
        assert_eq!(out.prp, 1, "marker {marker}");
    }
    // A verdict byte of 2 (the Gerbicz alarm) retires a k like a hit but is not counted as prp.
    let mut src = FakeSource::new(&[0]);
    src.answer_for_prime = 2;
    let out = run(&mut src, &[3, 3], 1, 1.0, 7).unwrap();
    assert_eq!(out.verdicts, vec![2, 7, 7, 0, 0, 0]);
    assert_eq!(out.prp, 0);
    assert_eq!(src.calls, vec![vec![0, 3], vec![4], vec![5]]);
}

/// Independent model: each k is simulated on its own, then the waves are assembled afterwards.
fn model(prime: &HashSet<u64>, sizes: &[u32], bounds: &[u32], marker: u8) -> (Vec<u8>, Vec<Vec<u64>>) {
    let count: u64 = sizes.iter().map(|&s| u64::from(s)).sum();
    let mut verdicts = vec![marker; count as usize];
    // per_wave[w][k] = global indices k contributes to wave w+1
    let mut per_wave: Vec<Vec<Vec<u64>>> = vec![vec![Vec::new(); sizes.len()]; bounds.len().saturating_sub(1)];
    let mut off = 0u64;
    for (k, &size) in sizes.iter().enumerate() {
        let mut lo = 0u32;
        for w in 1..bounds.len() {
            let hi = size.min(bounds[w]);
            let mut hit = false;
            for p in lo..hi {
                let g = off + u64::from(p);
                let v = u8::from(prime.contains(&g));
                verdicts[g as usize] = v;
                hit |= v == 1;
                per_wave[w - 1][k].push(g);
            }
            lo = hi;
            if hit || lo >= size {
                break;
            }
        }
        off += u64::from(size);
    }
    let calls: Vec<Vec<u64>> = per_wave
        .into_iter()
        .map(|ks| ks.into_iter().flatten().collect::<Vec<u64>>())
        .take_while(|c| !c.is_empty())
        .collect();
    (verdicts, calls)
}

#[test]
fn run_waves_matches_an_independent_model_on_random_layouts() {
    let mut rng = XorShift64::new(0x00EE_C0DE_5EED);
    for case in 0..200 {
        let nk = (rng.next_u64() % 9) as usize;
        let sizes: Vec<u32> = (0..nk).map(|_| (rng.next_u64() % 13) as u32).collect();
        let count: u64 = sizes.iter().map(|&s| u64::from(s)).sum();
        let density = [0u64, 5, 15, 40][(rng.next_u64() % 4) as usize];
        let prime: HashSet<u64> = (0..count).filter(|_| rng.next_u64() % 100 < density).collect();
        let chunk = (rng.next_u64() % 10) as i32 - 1;
        let growth = [1.0, 1.15, 1.5, 2.0][(rng.next_u64() % 4) as usize];
        let marker = (rng.next_u64() % 256) as u8;
        let s_max = sizes.iter().copied().max().unwrap_or(0);
        let bounds = schedule(chunk, growth, s_max);

        let mut src = FakeSource::new(&prime.iter().copied().collect::<Vec<_>>());
        let got = run_waves(&mut src, &numbers(count), &sizes, &bounds, marker).unwrap();
        let (want_v, want_calls) = model(&prime, &sizes, &bounds, marker);
        let ctx = format!("case {case}: sizes {sizes:?} chunk {chunk} growth {growth} marker {marker}");
        assert_eq!(got.verdicts, want_v, "{ctx}");
        assert_eq!(src.calls, want_calls, "{ctx}");
        assert_eq!(got.waves as usize, want_calls.len(), "{ctx}");
        assert_eq!(got.tested, want_calls.iter().map(|c| c.len() as u64).sum::<u64>(), "{ctx}");
        assert_eq!(got.prp, want_v.iter().filter(|&&b| b == 1).count() as u64, "{ctx}");
    }
}
