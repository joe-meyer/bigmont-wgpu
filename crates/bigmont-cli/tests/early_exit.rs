//! Pure early-exit logic: no GPU needed.

use bigmont_cli::early_exit::schedule;

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
