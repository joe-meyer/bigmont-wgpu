//! Command-line behaviour of the real `bigmont` binary: usage errors, exit codes and the tuning cache.

use bigmont_cli::formats;
use bigmont_core::gpu::Gpu;
use bigmont_oracle::{XorShift64, random_odd_modulus};
use num_bigint::BigUint;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

mod common;

fn bigmont(args: &[&str]) -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_bigmont"));
    c.args(args);
    c
}

fn run(mut c: Command) -> Output {
    c.output().expect("run bigmont")
}

fn code(out: &Output) -> Option<i32> {
    out.status.code()
}

#[test]
fn no_arguments_is_a_usage_error_with_exit_code_2() {
    let out = run(bigmont(&[]));
    assert_eq!(code(&out), Some(2));
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("usage:"),
        "usage goes to stderr"
    );
}

#[test]
fn help_exits_0_and_prints_usage() {
    for flag in ["--help", "-h"] {
        let out = run(bigmont(&[flag]));
        assert_eq!(code(&out), Some(0), "{flag}");
        assert!(
            String::from_utf8_lossy(&out.stdout).contains("usage:"),
            "{flag}"
        );
    }
}

#[test]
fn unknown_flags_and_missing_positionals_exit_2() {
    assert_eq!(code(&run(bigmont(&["--bogus"]))), Some(2));
    assert_eq!(code(&run(bigmont(&["only-one-file"]))), Some(2));
    assert_eq!(code(&run(bigmont(&["canary", "only-one-file"]))), Some(2));
}

#[test]
fn cache_and_no_cache_together_are_a_usage_error() {
    assert_eq!(
        code(&run(bigmont(&["a", "b", "--cache", "x", "--no-cache"]))),
        Some(2)
    );
}

// ---- tuning cache: default location, BIGMONT_CACHE, --no-cache ----

fn tmp_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("bigmont-cli-test-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// The engine's GPU backend needs an adapter with subgroups; see `common::usable_gpu`.
fn gpu_available() -> bool {
    common::usable_gpu()
}

/// 80 odd 1019-bit candidates (above the engine's 64-candidate GPU threshold), written to `dir`.
fn candidates(dir: &Path) -> PathBuf {
    let mut rng = XorShift64::new(5);
    let ns: Vec<BigUint> = (0..80)
        .map(|_| random_odd_modulus(&mut rng, 1019))
        .collect();
    let p = dir.join("cand.bin");
    formats::write_candidates(&p, 1024, &ns).unwrap();
    p
}

fn files_in(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    v.sort();
    v
}

#[test]
fn bigmont_cache_env_names_the_default_cache_and_no_cache_disables_it() {
    if !gpu_available() {
        return;
    }
    let dir = tmp_dir("cache");
    let cand = candidates(&dir);
    let cache_dir = dir.join("nested").join("cache-dir"); // parent does not exist yet
    let cache = cache_dir.join("tune.cache");

    let mut c = bigmont(&[
        cand.to_str().unwrap(),
        dir.join("v1.bin").to_str().unwrap(),
        "--backend",
        "gpu",
        "--no-cache",
    ]);
    c.env("BIGMONT_CACHE", &cache);
    let out = run(c);
    assert_eq!(
        code(&out),
        Some(0),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        !cache.exists() && !cache_dir.exists(),
        "--no-cache must not create the cache"
    );

    let mut c = bigmont(&[
        cand.to_str().unwrap(),
        dir.join("v2.bin").to_str().unwrap(),
        "--backend",
        "gpu",
    ]);
    c.env("BIGMONT_CACHE", &cache);
    let out = run(c);
    assert_eq!(
        code(&out),
        Some(0),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        cache.exists(),
        "the default (here BIGMONT_CACHE) cache is written"
    );
    assert_eq!(
        files_in(&cache_dir),
        ["tune.cache"],
        "an atomic write leaves no temporary file"
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn the_tune_subcommand_writes_the_cache() {
    if !gpu_available() {
        return;
    }
    let dir = tmp_dir("tune");
    let cache = dir.join("tune.cache");
    let out = run(bigmont(&[
        "tune",
        "--bits",
        "1019",
        "--cache",
        cache.to_str().unwrap(),
    ]));
    assert_eq!(
        code(&out),
        Some(0),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = std::fs::read_to_string(&cache).expect("tune writes the cache");
    // Auto may pick either scheme: the line must parse as a winner of one of them (the value is the
    // last two `|` fields, `<unrolled|unrolled32> <tpi> <wg>|<rate>`), under the Auto key
    // (`name|backend|driver|any|class`).
    let line = text
        .lines()
        .find(|l| !l.is_empty())
        .expect("one cache line");
    let mut fields = line.rsplitn(3, '|');
    let (rate, shape, key) = (
        fields.next().unwrap(),
        fields.next().unwrap(),
        fields.next().unwrap(),
    );
    assert!(
        bigmont_cli::tune::parse_value(&format!("{shape}|{rate}")).is_some(),
        "{text}"
    );
    assert!(
        key.contains("|any|") && !key.contains("only-"),
        "tune with the default --limbs uses the Auto (`any`) key: {text}"
    );
    assert_eq!(files_in(&dir), ["tune.cache"]);
    let _ = std::fs::remove_dir_all(dir);
}

// ---- --limbs auto|13|32 ----

#[test]
fn an_invalid_limbs_value_is_a_usage_error_naming_the_allowed_values() {
    for bad in ["7", "64", "", "thirty-two"] {
        let out = run(bigmont(&["a", "b", "--limbs", bad]));
        assert_eq!(code(&out), Some(2), "--limbs {bad:?}");
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(
            err.contains("auto") && err.contains("13") && err.contains("32"),
            "--limbs {bad:?}: {err}"
        );
    }
    let out = run(bigmont(&["tune", "--limbs=7"]));
    assert_eq!(code(&out), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("--limbs"));
    let out = run(bigmont(&["--help"]));
    assert!(
        String::from_utf8_lossy(&out.stdout).contains("--limbs"),
        "the usage text documents the flag"
    );
}

fn has_int64() -> Option<bool> {
    Gpu::new().ok().map(|g| g.has_int64)
}

/// Whether a forced-32 test can run here; skips loudly (an error unless BIGMONT_ALLOW_SKIP=1) when not.
fn int64_available() -> bool {
    match has_int64() {
        Some(true) => true,
        other => {
            assert!(
                std::env::var("BIGMONT_ALLOW_SKIP").is_ok_and(|v| v == "1"),
                "no int64 adapter (found: {other:?}); set BIGMONT_ALLOW_SKIP=1 to allow skips"
            );
            eprintln!("SKIP no adapter with SHADER_INT64");
            false
        }
    }
}

#[test]
fn limbs_32_runs_a_candidate_file_and_writes_the_expected_verdicts() {
    if !int64_available() {
        return;
    }
    let dir = tmp_dir("limbs32");
    let cand = candidates(&dir);
    let verdict = dir.join("v.bin");
    let cache = dir.join("tune.cache");
    let out = run(bigmont(&[
        cand.to_str().unwrap(),
        verdict.to_str().unwrap(),
        "--backend",
        "gpu",
        "--limbs",
        "32",
        "--cache",
        cache.to_str().unwrap(),
    ]));
    assert_eq!(
        code(&out),
        Some(0),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let file = formats::read_candidates(&cand).unwrap();
    let (_, got) = formats::read_verdicts(&verdict).unwrap();
    let want: Vec<u8> = file
        .numbers
        .iter()
        .map(|n| u8::from(bigmont_cpu::fermat_base2(n)))
        .collect();
    assert_eq!(got, want);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("Limb32"),
        "the engine's tune lines name the scheme: {stderr}"
    );
    assert!(
        stderr.contains("gpu 80"),
        "all 80 candidates ran on the GPU: {stderr}"
    );
    let text = std::fs::read_to_string(&cache).unwrap();
    assert!(
        text.lines()
            .all(|l| l.contains("|only-32|") && l.contains("|unrolled32 ")),
        "{text}"
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn the_tune_subcommand_forces_a_scheme_and_prints_a_scheme_column() {
    if !int64_available() {
        return;
    }
    let dir = tmp_dir("tune32");
    let cache = dir.join("tune.cache");
    let out = run(bigmont(&[
        "tune",
        "--bits",
        "1019",
        "--limbs",
        "32",
        "--cache",
        cache.to_str().unwrap(),
    ]));
    assert_eq!(
        code(&out),
        Some(0),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let header = stdout
        .lines()
        .find(|l| l.contains("squarings/s") && l.contains("tpi"))
        .expect("table header");
    assert!(header.contains("scheme"), "{header}");
    assert!(
        stdout
            .lines()
            .filter(|l| l.trim_end().ends_with("ok"))
            .all(|l| l.split_whitespace().next() == Some("32")),
        "only 32-bit rows: {stdout}"
    );
    assert!(stdout.contains("best: scheme 32"), "{stdout}");
    let text = std::fs::read_to_string(&cache).expect("tune writes the cache");
    assert!(
        text.lines()
            .all(|l| l.contains("|only-32|") && l.contains("|unrolled32 ")),
        "{text}"
    );
    let _ = std::fs::remove_dir_all(dir);
}

// ---- early exit (--ee, --bins, --chunk, --growth, --marker) and --lock ----

use bigmont_cli::early_exit::schedule;
use std::io::{BufRead, BufReader, Read};
use std::process::Stdio;
use std::time::{Duration, Instant};

const EE_BITS: u32 = 192;

/// Candidates grouped into ks by a sizes list, with the plain (no early exit) verdicts of the oracle.
struct Data {
    ns: Vec<BigUint>,
    sizes: Vec<u32>,
    plain: Vec<u8>,
}

/// Builds ks from `(size, prime positions)`: composites everywhere except at the listed positions, which
/// hold probable primes (base-2 Fermat is the oracle). Candidate widths run through 64..=130 bits.
fn build(spec: &[(u32, &[u32])], seed: u64) -> Data {
    build_with(spec, seed, |i| 64 + (i * 7) % 67)
}

/// [`build`] with the bit length of candidate `i` chosen by `bits_of`.
fn build_with(spec: &[(u32, &[u32])], seed: u64, bits_of: impl Fn(u32) -> u32) -> Data {
    let mut rng = XorShift64::new(seed);
    let (mut ns, mut sizes, mut i) = (Vec::new(), Vec::new(), 0u32);
    for &(size, primes) in spec {
        sizes.push(size);
        for p in 0..size {
            let bits = bits_of(i);
            i += 1;
            let want_prime = primes.contains(&p);
            let n = loop {
                let n = random_odd_modulus(&mut rng, bits);
                if bigmont_cpu::fermat_base2(&n) == want_prime {
                    break n;
                }
            };
            ns.push(n);
        }
    }
    let plain = ns
        .iter()
        .map(|n| u8::from(bigmont_cpu::fermat_base2(n)))
        .collect();
    Data { ns, sizes, plain }
}

/// Mixed ks: no prime, an early prime, a late prime, a prime in the last position, several primes in a k,
/// an empty k and long ks, so default early exit skips plenty and takes several waves.
fn mixed() -> Data {
    build(
        &[
            (1, &[]),
            (3, &[0]),
            (7, &[2]),
            (9, &[8]),
            (12, &[]),
            (0, &[]),
            (20, &[5]),
            (20, &[6, 7, 19]),
            (33, &[14]),
            (40, &[]),
            (41, &[40]),
            (30, &[7, 25]),
            (5, &[]),
        ],
        91,
    )
}

/// What early exit must produce, worked out per k from the plain verdicts: a k is tested up to the end of
/// the wave that holds its first prime (or to its end if it has none). Returns (verdicts, tested, waves).
fn model(plain: &[u8], sizes: &[u32], bounds: &[u32], marker: u8) -> (Vec<u8>, u64, u32) {
    let mut out = vec![marker; plain.len()];
    let (mut off, mut tested, mut max_end) = (0usize, 0u64, 0u32);
    for &size in sizes {
        let ks = &plain[off..off + size as usize];
        let end = match ks.iter().position(|&v| v == 1) {
            Some(h) => bounds
                .iter()
                .copied()
                .find(|&b| b as usize > h)
                .expect("last boundary covers every position")
                .min(size),
            None => size,
        };
        out[off..off + end as usize].copy_from_slice(&ks[..end as usize]);
        tested += u64::from(end);
        max_end = max_end.max(end);
        off += size as usize;
    }
    let waves = bounds.iter().filter(|&&b| b < max_end).count() as u32;
    (out, tested, waves)
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

fn s_max(sizes: &[u32]) -> u32 {
    sizes.iter().copied().max().unwrap_or(0)
}

/// A scratch directory holding `cand.bin` and `sizes.bin` for `d`.
fn ee_dir(tag: &str, d: &Data) -> PathBuf {
    let dir = tmp_dir(tag);
    formats::write_candidates(&dir.join("cand.bin"), EE_BITS, &d.ns).unwrap();
    formats::write_sizes(&dir.join("sizes.bin"), &d.sizes).unwrap();
    dir
}

/// `bigmont cand.bin out.bin --backend cpu --no-cache <extra>` in `dir`.
fn ee_run(dir: &Path, out_name: &str, extra: &[&str]) -> Output {
    let mut args = vec![
        s(&dir.join("cand.bin")).to_string(),
        s(&dir.join(out_name)).to_string(),
        "--backend".into(),
        "cpu".into(),
        "--no-cache".into(),
    ];
    args.extend(extra.iter().map(|a| a.to_string()));
    let refs: Vec<&str> = args.iter().map(String::as_str).collect();
    run(bigmont(&refs))
}

fn ok(out: &Output) {
    assert_eq!(
        code(out),
        Some(0),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

fn verdicts_of(path: &Path) -> Vec<u8> {
    let (bits, v) = formats::read_verdicts(path).unwrap();
    assert_eq!(
        bits, EE_BITS,
        "the verdict header carries the candidates' container width"
    );
    v
}

/// The `key=value` fields of the `g1_fermat[EE]:` line.
fn summary(out: &Output) -> Vec<(String, String)> {
    let err = String::from_utf8_lossy(&out.stderr);
    let line = err
        .lines()
        .find(|l| l.starts_with("g1_fermat[EE]: bits="))
        .unwrap_or_else(|| panic!("no summary line in: {err}"));
    line.split_whitespace()
        .filter_map(|t| t.split_once('='))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn field(sum: &[(String, String)], key: &str) -> String {
    sum.iter()
        .find(|(k, _)| k == key)
        .unwrap_or_else(|| panic!("no {key}= in {sum:?}"))
        .1
        .clone()
}

#[test]
fn the_oracle_data_has_what_the_ee_tests_need() {
    let d = mixed();
    let b = schedule(6, 1.15, s_max(&d.sizes));
    let (v, tested, waves) = model(&d.plain, &d.sizes, &b, 3);
    assert!(v.contains(&3), "some positions are skipped");
    assert!(d.plain.iter().filter(|&&x| x == 1).count() >= 8);
    assert!(
        tested < d.ns.len() as u64 && waves >= 3,
        "tested {tested} of {}, {waves} waves",
        d.ns.len()
    );
}

#[test]
fn ee_matches_a_plain_run_where_tested_and_writes_the_marker_where_skipped() {
    let d = mixed();
    let dir = ee_dir("ee-basic", &d);
    ok(&ee_run(&dir, "plain.bin", &[]));
    let plain_v = verdicts_of(&dir.join("plain.bin"));
    assert_eq!(plain_v, d.plain, "the plain run agrees with the oracle");

    let out = ee_run(&dir, "ee.bin", &["--ee", s(&dir.join("sizes.bin"))]);
    ok(&out);
    let got = verdicts_of(&dir.join("ee.bin"));
    assert_eq!(got.len(), d.ns.len());
    let (want, _, _) = model(&d.plain, &d.sizes, &schedule(6, 1.15, s_max(&d.sizes)), 3);
    assert_eq!(got, want);
    // And the properties that define it, position by position.
    let mut off = 0usize;
    for &size in &d.sizes {
        let k = off..off + size as usize;
        match got[k.clone()].iter().position(|&v| v == 3) {
            None => assert_eq!(
                got[k.clone()],
                plain_v[k.clone()],
                "a k that is never cut short equals the plain run"
            ),
            Some(first_skipped) => {
                let cut = k.start + first_skipped;
                assert_eq!(
                    got[k.start..cut],
                    plain_v[k.start..cut],
                    "the tested part equals the plain run"
                );
                assert!(
                    got[cut..k.end].iter().all(|&v| v == 3),
                    "everything after the first skip is skipped"
                );
                assert!(
                    plain_v[k.start..cut].contains(&1),
                    "only a k with a prime in its tested part is cut short"
                );
            }
        }
        off += size as usize;
    }
    assert!(!dir.join("ee.bin.tmp").exists());
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn bins_list_with_crlf_equals_the_concatenation_run_as_one_file() {
    let d = mixed();
    let dir = ee_dir("ee-bins", &d);
    // Split in the middle of a k on purpose: the files are one array.
    let cut = d.ns.len() / 3 + 1;
    formats::write_candidates(&dir.join("a.bin"), EE_BITS, &d.ns[..cut]).unwrap();
    formats::write_candidates(&dir.join("b.bin"), EE_BITS, &d.ns[cut..]).unwrap();
    let list = format!(
        "{}  \r\n\r\n{}\r\n",
        s(&dir.join("a.bin")),
        s(&dir.join("b.bin"))
    );
    std::fs::write(dir.join("list.txt"), list).unwrap();
    let sizes = s(&dir.join("sizes.bin")).to_string();

    ok(&ee_run(&dir, "one.bin", &["--ee", &sizes]));
    let out = run(bigmont(&[
        s(&dir.join("list.txt")),
        s(&dir.join("list-out.bin")),
        "--backend",
        "cpu",
        "--no-cache",
        "--bins",
        "--ee",
        &sizes,
    ]));
    ok(&out);
    assert_eq!(
        verdicts_of(&dir.join("list-out.bin")),
        verdicts_of(&dir.join("one.bin"))
    );
    assert_eq!(field(&summary(&out), "count"), d.ns.len().to_string());
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn marker_chunk_and_growth_change_the_output_as_the_semantics_require() {
    let d = mixed();
    let dir = ee_dir("ee-flags", &d);
    let sizes = s(&dir.join("sizes.bin")).to_string();
    let top = s_max(&d.sizes);

    // (flags, chunk, growth, marker)
    let cases: [(&[&str], i32, f64, u8); 5] = [
        (&["--marker", "7"], 6, 1.15, 7),
        (&["--marker", "4"], 6, 1.15, 4),
        (&["--chunk", "1", "--growth", "1.0"], 1, 1.0, 3),
        (
            &["--chunk", "2", "--growth", "1.0", "--marker", "255"],
            2,
            1.0,
            255,
        ),
        (&["--chunk", "0", "--growth", "1.5"], 0, 1.5, 3), // chunk <= 0 steps by one position at a time
    ];
    for (i, (flags, chunk, growth, marker)) in cases.iter().enumerate() {
        let mut args = vec!["--ee", sizes.as_str()];
        args.extend_from_slice(flags);
        let name = format!("o{i}.bin");
        let out = ee_run(&dir, &name, &args);
        ok(&out);
        let (want, tested, waves) =
            model(&d.plain, &d.sizes, &schedule(*chunk, *growth, top), *marker);
        assert_eq!(verdicts_of(&dir.join(&name)), want, "{flags:?}");
        let sum = summary(&out);
        assert_eq!(field(&sum, "tested"), tested.to_string(), "{flags:?}");
        assert_eq!(field(&sum, "waves"), waves.to_string(), "{flags:?}");
    }
    // One huge chunk is a single wave with nothing skipped: equal to a plain run.
    let out = ee_run(
        &dir,
        "big.bin",
        &["--ee", &sizes, "--chunk", "1000000", "--growth", "1.0"],
    );
    ok(&out);
    assert_eq!(verdicts_of(&dir.join("big.bin")), d.plain);
    assert_eq!(field(&summary(&out), "waves"), "1");
    assert_eq!(field(&summary(&out), "tested"), d.ns.len().to_string());
    // chunk 1 retires as early as possible: strictly fewer tests than the default, more waves.
    let tiny = summary(&ee_run(
        &dir,
        "tiny.bin",
        &["--ee", &sizes, "--chunk", "1", "--growth", "1.0"],
    ));
    let default = summary(&ee_run(&dir, "def.bin", &["--ee", &sizes]));
    assert!(
        field(&tiny, "tested").parse::<u64>().unwrap()
            < field(&default, "tested").parse::<u64>().unwrap()
    );
    assert!(
        field(&tiny, "waves").parse::<u32>().unwrap()
            > field(&default, "waves").parse::<u32>().unwrap()
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn the_summary_line_has_the_documented_format_and_the_independently_computed_waves() {
    let d = mixed();
    let dir = ee_dir("ee-summary", &d);
    let out = ee_run(&dir, "o.bin", &["--ee", s(&dir.join("sizes.bin"))]);
    ok(&out);
    let err = String::from_utf8_lossy(&out.stderr);
    let line = err
        .lines()
        .find(|l| l.starts_with("g1_fermat[EE]: bits="))
        .expect("summary line");
    let (want_v, tested, waves) = model(&d.plain, &d.sizes, &schedule(6, 1.15, s_max(&d.sizes)), 3);
    let count = d.ns.len();
    let prp = want_v.iter().filter(|&&v| v == 1).count();
    let head = format!(
        "g1_fermat[EE]: bits={EE_BITS} count={count} ks={} tested={tested} ({:.3}) waves={waves} | ",
        d.sizes.len(),
        tested as f64 / count as f64
    );
    assert!(
        line.starts_with(&head),
        "{line}\nwanted a line starting {head}"
    );
    assert!(
        line.contains("s gpu-phase (lock wait 0.00s) | ") && line.contains(" tests/s | "),
        "{line}"
    );
    assert!(line.ends_with(&format!("prp={prp} gec_alarm=0")), "{line}");
    let _ = std::fs::remove_dir_all(dir);
}

/// Runs `args`; asserts exit status 2, `needle` on stderr and that `out_name` (and its temporary) do not
/// exist in `dir` afterwards.
fn expect_input_error(dir: &Path, args: &[&str], out_name: &str, needle: &str) {
    let out = run(bigmont(args));
    assert_eq!(
        code(&out),
        Some(2),
        "{args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains(needle), "{args:?}: wanted {needle:?} in {err}");
    assert!(
        !dir.join(out_name).exists() && !dir.join(format!("{out_name}.tmp")).exists(),
        "{args:?}: an output file was left behind"
    );
}

#[test]
fn ee_input_errors_exit_2_with_a_message_and_leave_no_output() {
    let d = mixed();
    let dir = ee_dir("ee-errors", &d);
    let (cand, sizes, out) = (
        s(&dir.join("cand.bin")).to_string(),
        s(&dir.join("sizes.bin")).to_string(),
        s(&dir.join("o.bin")).to_string(),
    );
    // `bigmont <input> o.bin --backend cpu --no-cache <extra>` must fail with `needle`.
    let fails = |input: &str, extra: &[&str], needle: &str| {
        let mut a = vec![input, out.as_str(), "--backend", "cpu", "--no-cache"];
        a.extend_from_slice(extra);
        expect_input_error(&dir, &a, "o.bin", needle);
    };

    fails(&cand, &["--bins"], "--bins requires --ee");
    fails(&cand, &["--ee", &sizes, "--marker", "256"], "--marker");
    // 0, 1 and 2 are verdict bytes; a skipped position carrying one would read as a real verdict.
    for m in ["0", "1", "2"] {
        fails(&cand, &["--ee", &sizes, "--marker", m], "verdict bytes");
    }
    fails(&cand, &["--ee", &sizes, "--chunk", "many"], "--chunk");
    fails(&cand, &["--ee", &sizes, "--growth", "nan"], "--growth");

    // Sizes that do not add up to the candidate count.
    let mut wrong = d.sizes.clone();
    wrong[0] += 1;
    formats::write_sizes(&dir.join("wrong.bin"), &wrong).unwrap();
    fails(&cand, &["--ee", s(&dir.join("wrong.bin"))], "sizes sum");

    // Bad magic, truncated, and missing sizes files.
    let good = std::fs::read(dir.join("sizes.bin")).unwrap();
    let mut bad = good.clone();
    bad[0] ^= 0xFF;
    std::fs::write(dir.join("badmagic.bin"), &bad).unwrap();
    fails(&cand, &["--ee", s(&dir.join("badmagic.bin"))], "magic");
    std::fs::write(dir.join("short.bin"), &good[..good.len() - 2]).unwrap();
    fails(&cand, &["--ee", s(&dir.join("short.bin"))], "short.bin");
    fails(&cand, &["--ee", s(&dir.join("absent.bin"))], "absent.bin");

    // A missing candidate file, directly and through a list; mixed container widths; an empty list.
    let absent = s(&dir.join("nope.bin")).to_string();
    fails(&absent, &["--ee", &sizes], "nope.bin");
    std::fs::write(dir.join("missing.txt"), format!("{cand}\n{absent}\n")).unwrap();
    fails(
        s(&dir.join("missing.txt")),
        &["--bins", "--ee", &sizes],
        "nope.bin",
    );
    formats::write_candidates(&dir.join("wide.bin"), EE_BITS + 32, &d.ns[..1]).unwrap();
    std::fs::write(
        dir.join("mixed.txt"),
        format!("{cand}\n{}\n", s(&dir.join("wide.bin"))),
    )
    .unwrap();
    fails(
        s(&dir.join("mixed.txt")),
        &["--bins", "--ee", &sizes],
        "mixed bits",
    );
    std::fs::write(dir.join("empty.txt"), "\r\n  \r\n").unwrap();
    fails(
        s(&dir.join("empty.txt")),
        &["--bins", "--ee", &sizes],
        "empty.txt",
    );
    let _ = std::fs::remove_dir_all(dir);
}

/// `--backend gpu` with an adapter that does not exist must fail even when there is nothing to test:
/// an empty input never reaches the engine's verdict call, which used to be the only check.
#[test]
fn a_demanded_gpu_that_is_missing_fails_even_with_no_candidates() {
    let dir = tmp_dir("ee-empty-gpu");
    formats::write_candidates(&dir.join("cand.bin"), EE_BITS, &[]).unwrap();
    formats::write_sizes(&dir.join("sizes.bin"), &[]).unwrap();
    let (cand, sizes, out) = (
        s(&dir.join("cand.bin")).to_string(),
        s(&dir.join("sizes.bin")).to_string(),
        s(&dir.join("o.bin")).to_string(),
    );
    let gpu = [
        "--backend",
        "gpu",
        "--adapter",
        "no-such-adapter-zzz",
        "--no-cache",
    ];
    let mut ee = vec![cand.as_str(), out.as_str(), "--ee", sizes.as_str()];
    ee.extend_from_slice(&gpu);
    expect_input_error(&dir, &ee, "o.bin", "GPU backend requested but unavailable");
    let mut plain = vec![cand.as_str(), out.as_str()];
    plain.extend_from_slice(&gpu);
    expect_input_error(
        &dir,
        &plain,
        "o.bin",
        "GPU backend requested but unavailable",
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn an_absurd_schedule_is_refused_before_anything_is_allocated() {
    let d = build(&[(3, &[1])], 5);
    let dir = ee_dir("ee-absurd", &d);
    // One k of 4 billion candidates with a step of 1 would be 4 billion boundaries (16 GB of schedule).
    formats::write_sizes(&dir.join("huge.bin"), &[u32::MAX]).unwrap();
    let (cand, huge, out) = (
        s(&dir.join("cand.bin")).to_string(),
        s(&dir.join("huge.bin")).to_string(),
        s(&dir.join("o.bin")).to_string(),
    );
    for flags in [
        &["--chunk", "1", "--growth", "1.0"][..],
        &["--chunk", "0", "--growth", "1.0"],
        &["--chunk", "-5"],
    ] {
        let mut a = vec![
            cand.as_str(),
            out.as_str(),
            "--backend",
            "cpu",
            "--no-cache",
            "--ee",
            huge.as_str(),
        ];
        a.extend_from_slice(flags);
        let started = Instant::now();
        expect_input_error(&dir, &a, "o.bin", "waves");
        assert!(
            started.elapsed() < Duration::from_secs(20),
            "{flags:?}: refused quickly"
        );
    }
    // A growing schedule over the same k is short, so it is not refused for its length (it fails the
    // sizes-sum check instead, because the file does not hold 4 billion candidates).
    let a = [
        cand.as_str(),
        out.as_str(),
        "--backend",
        "cpu",
        "--no-cache",
        "--ee",
        huge.as_str(),
    ];
    expect_input_error(&dir, &a, "o.bin", "sizes sum");
    let a = [
        cand.as_str(),
        out.as_str(),
        "--backend",
        "cpu",
        "--no-cache",
        "--ee",
        huge.as_str(),
        "--chunk",
        "1",
    ];
    expect_input_error(&dir, &a, "o.bin", "sizes sum");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn the_ee_flags_are_rejected_by_other_subcommands() {
    for sub in ["canary", "tune"] {
        let out = run(bigmont(&[sub, "a", "b", "--ee", "sizes.bin"]));
        assert_eq!(code(&out), Some(2), "{sub}");
        assert!(
            String::from_utf8_lossy(&out.stderr).contains("--ee"),
            "{sub}"
        );
    }
}

/// Runs bigmont with `args` while this test holds an exclusive lock on `lock`, releasing it `hold` after
/// the child reports that it is waiting for it. Returns the child's output.
fn run_while_locked(args: &[&str], lock: &Path, hold: Duration) -> Output {
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(lock)
        .unwrap();
    file.lock().unwrap();
    let mut child = bigmont(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn bigmont");
    let mut stderr = BufReader::new(child.stderr.take().unwrap());
    let mut seen = String::new();
    loop {
        let mut line = String::new();
        let n = stderr
            .read_line(&mut line)
            .expect("read the child's stderr");
        assert!(
            n > 0,
            "the child exited without waiting for the lock: {seen}"
        );
        seen.push_str(&line);
        if line.contains("waiting for the lock") {
            break;
        }
    }
    std::thread::sleep(hold);
    file.unlock().unwrap();
    let mut rest = String::new();
    stderr.read_to_string(&mut rest).unwrap();
    let mut stdout = String::new();
    child
        .stdout
        .take()
        .unwrap()
        .read_to_string(&mut stdout)
        .unwrap();
    let status = child.wait().unwrap();
    Output {
        status,
        stdout: stdout.into_bytes(),
        stderr: (seen + &rest).into_bytes(),
    }
}

#[test]
fn lock_waits_for_the_holder_and_the_ee_summary_reports_the_wait() {
    let d = mixed();
    let dir = ee_dir("ee-lock", &d);
    let lock = dir.join("gpu.lock");
    assert!(!lock.exists());
    let (cand, sizes, out) = (
        s(&dir.join("cand.bin")).to_string(),
        s(&dir.join("sizes.bin")).to_string(),
        s(&dir.join("o.bin")).to_string(),
    );
    let args = [
        cand.as_str(),
        out.as_str(),
        "--backend",
        "cpu",
        "--no-cache",
        "--ee",
        sizes.as_str(),
        "--lock",
        s(&lock),
    ];
    let hold = Duration::from_millis(500);
    let o = run_while_locked(&args, &lock, hold);
    ok(&o);
    let err = String::from_utf8_lossy(&o.stderr);
    let after = err
        .split("lock wait ")
        .nth(1)
        .expect("lock wait in the summary");
    let waited: f64 = after.split('s').next().unwrap().parse().unwrap();
    assert!(
        waited >= hold.as_secs_f64() - 0.01,
        "reported a lock wait of {waited}s for a {hold:?} hold: {err}"
    ); // %.2f rounds
    let (want, _, _) = model(&d.plain, &d.sizes, &schedule(6, 1.15, s_max(&d.sizes)), 3);
    assert_eq!(verdicts_of(&dir.join("o.bin")), want);
    assert!(lock.exists());
    let again = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&lock)
        .unwrap();
    again
        .try_lock()
        .expect("the child released the lock when it exited");
    again.unlock().unwrap();

    // Uncontended: no waiting message and a near-zero wait.
    let o = ee_run(&dir, "o2.bin", &["--ee", &sizes, "--lock", s(&lock)]);
    ok(&o);
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(!err.contains("waiting for the lock"), "{err}");
    assert!(err.contains("(lock wait 0.0"), "{err}");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn lock_works_in_plain_mode_too_and_is_no_longer_ignored() {
    let d = mixed();
    let dir = ee_dir("plain-lock", &d);
    let lock = dir.join("plain.lock");
    let (cand, out) = (
        s(&dir.join("cand.bin")).to_string(),
        s(&dir.join("o.bin")).to_string(),
    );
    let args = [
        cand.as_str(),
        out.as_str(),
        "--backend",
        "cpu",
        "--no-cache",
        "--lock",
        s(&lock),
    ];

    // `run_while_locked` itself proves the child waited: it fails unless the child reports waiting.
    let o = run_while_locked(&args, &lock, Duration::from_millis(500));
    ok(&o);
    assert_eq!(verdicts_of(&dir.join("o.bin")), d.plain);
    assert!(
        !String::from_utf8_lossy(&o.stderr).contains("accepted and ignored"),
        "--lock is real now"
    );

    let fresh = dir.join("fresh.lock");
    let o = ee_run(&dir, "o2.bin", &["--lock", s(&fresh)]);
    ok(&o);
    assert!(fresh.exists(), "the lock file is created");
    assert!(!String::from_utf8_lossy(&o.stderr).contains("accepted and ignored"));
    assert_eq!(verdicts_of(&dir.join("o2.bin")), d.plain);
    // --repeat is still accepted and ignored.
    let o = ee_run(&dir, "o3.bin", &["--repeat", "2"]);
    ok(&o);
    assert!(String::from_utf8_lossy(&o.stderr).contains("--repeat is accepted and ignored"));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn ee_runs_end_to_end_on_the_gpu() {
    if !gpu_available() {
        return;
    }
    // 50 ks of 12, all 190-bit (the engine batches by exact bit length and sends 64 or more to the GPU):
    // the first wave (6 per k) is 300 candidates.
    let primes: Vec<Vec<u32>> = (0..50u32)
        .map(|k| if k % 3 == 0 { vec![] } else { vec![k % 11] })
        .collect();
    let spec: Vec<(u32, &[u32])> = primes.iter().map(|p| (12u32, p.as_slice())).collect();
    let d = build_with(&spec, 4242, |_| 190);
    let dir = ee_dir("ee-gpu", &d);
    let cache = dir.join("tune.cache");
    let sizes = s(&dir.join("sizes.bin")).to_string();
    let (cand, out_path) = (dir.join("cand.bin"), dir.join("o.bin"));
    let args = [
        s(&cand),
        s(&out_path),
        "--backend",
        "gpu",
        "--cache",
        s(&cache),
        "--ee",
        sizes.as_str(),
    ];
    let out = run(bigmont(&args));
    ok(&out);
    let (want, tested, waves) = model(&d.plain, &d.sizes, &schedule(6, 1.15, 12), 3);
    assert_eq!(verdicts_of(&dir.join("o.bin")), want);
    let sum = summary(&out);
    assert_eq!(field(&sum, "tested"), tested.to_string());
    assert_eq!(field(&sum, "waves"), waves.to_string());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("GPU:") && !err.contains("(gpu 0,"),
        "the waves ran on the GPU: {err}"
    );
    let _ = std::fs::remove_dir_all(dir);
}

/// Three random 2029-bit candidates (the Beacon harness drill input) in a 2048-bit file.
fn drill_candidates(dir: &Path) -> (PathBuf, Vec<BigUint>) {
    let mut rng = XorShift64::new(31);
    let ns: Vec<BigUint> = (0..3).map(|_| random_odd_modulus(&mut rng, 2029)).collect();
    let p = dir.join("drill.bin");
    formats::write_candidates(&p, 2048, &ns).unwrap();
    (p, ns)
}

#[test]
fn a_gec_run_writes_the_same_verdicts_as_a_plain_run() {
    if !gpu_available() {
        return;
    }
    let dir = tmp_dir("gec-clean");
    let cand = candidates(&dir);
    let cache = dir.join("tune.cache");
    let mut files = Vec::new();
    for (name, extra) in [("plain.bin", None), ("gec.bin", Some("--gec"))] {
        let verdict = dir.join(name);
        let mut args = vec![
            cand.to_str().unwrap(),
            verdict.to_str().unwrap(),
            "--backend",
            "gpu",
            "--cache",
            cache.to_str().unwrap(),
        ];
        args.extend(extra);
        let out = run(bigmont(&args));
        assert_eq!(
            code(&out),
            Some(0),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        if extra.is_some() {
            assert!(
                String::from_utf8_lossy(&out.stderr).contains("GEC alarms 0"),
                "{}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
        files.push(formats::read_verdicts(&verdict).unwrap().1);
    }
    assert_eq!(files[0], files[1]);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn the_inject_drill_gives_verdict_2_for_the_injected_candidate() {
    if !gpu_available() {
        return;
    }
    let dir = tmp_dir("gec-drill");
    let (cand, ns) = drill_candidates(&dir);
    let verdict = dir.join("v.bin");
    let cache = dir.join("tune.cache");
    let out = run(bigmont(&[
        cand.to_str().unwrap(),
        verdict.to_str().unwrap(),
        "--backend",
        "gpu",
        "--cache",
        cache.to_str().unwrap(),
        "--gec",
        "--inject",
        "1",
        "3",
    ]));
    assert_eq!(
        code(&out),
        Some(0),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let want: Vec<u8> = ns
        .iter()
        .map(|n| u8::from(bigmont_cpu::fermat_base2(n)))
        .collect();
    assert_eq!(
        formats::read_verdicts(&verdict).unwrap().1,
        vec![want[0], 2, want[2]]
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn inject_usage_errors_exit_2_before_any_work() {
    let dir = tmp_dir("gec-usage");
    let (cand, _) = drill_candidates(&dir);
    let (c, v) = (cand.to_str().unwrap(), dir.join("v.bin"));
    let v = v.to_str().unwrap();
    let cases: [(&[&str], &str); 5] = [
        (&[c, v, "--inject", "1", "3"], "--inject needs --gec"),
        (
            &[c, v, "--gec", "--inject", "5", "3"],
            "index 5 is out of range",
        ),
        (
            &[c, v, "--gec", "--inject", "0", "0", "--ee", "s.bin"],
            "plain runs only",
        ),
        (
            &[c, v, "--gec", "--inject", "0", "0", "--backend", "cpu"],
            "needs the GPU backend",
        ),
        (&[c, v, "--gec", "--inject=1"], "needs two values"),
    ];
    for (args, needle) in cases {
        let out = run(bigmont(args));
        assert_eq!(code(&out), Some(2), "{args:?}");
        // The message is the first line; the usage text after it names every flag.
        let err = String::from_utf8_lossy(&out.stderr);
        let first = err.lines().next().unwrap_or_default();
        assert!(first.contains(needle), "{args:?}: {err}");
        assert!(
            !Path::new(v).exists(),
            "{args:?} must not write a verdict file"
        );
    }
    let _ = std::fs::remove_dir_all(dir);
}
