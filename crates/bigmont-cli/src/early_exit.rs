//! In-driver early exit: candidates are grouped into "k"s by a sizes file and tested in waves; a k
//! that has produced a probable prime is retired and its remaining candidates are never tested.

/// Wave boundaries `[0, b1, b2, ..., s_max]`: wave `w` covers positions `[b[w-1], b[w])` of every
/// k still alive. The first step is `chunk`; the step is multiplied by `growth` after each wave
/// when `growth > 1.0`, rounded half to even, clamped to at least 1, and the last boundary is
/// capped at `s_max` (the largest k size). `chunk <= 0` therefore gives steps of 1, and
/// `s_max == 0` gives `[0]` (nothing to test).
pub fn schedule(chunk: i32, growth: f64, s_max: u32) -> Vec<u32> {
    let s = i64::from(s_max);
    let mut bounds = vec![0u32];
    let mut at = 0i64;
    let mut c = f64::from(chunk);
    while at < s {
        // `as i64` saturates (and maps NaN to 0), so a runaway `c` cannot overflow.
        let step = (c.round_ties_even() as i64).max(1);
        at = at.saturating_add(step).min(s);
        bounds.push(u32::try_from(at).expect("at <= s_max"));
        if growth > 1.0 {
            c *= growth;
        }
    }
    bounds
}
