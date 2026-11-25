use bigmont_cli::formats::*;
use bigmont_oracle::{XorShift64, random_odd_modulus};
use num_bigint::BigUint;
use std::path::PathBuf;

fn tmp(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("bigmont-files-{}-{name}", std::process::id()))
}

#[test]
fn candidate_file_round_trips_and_has_the_g2c1_layout() {
    let mut rng = XorShift64::new(1);
    let ns: Vec<BigUint> = (0..5).map(|_| random_odd_modulus(&mut rng, 2029)).collect();
    let p = tmp("cand.bin");
    write_candidates(&p, 2048, &ns).unwrap();
    let raw = std::fs::read(&p).unwrap();
    assert_eq!(&raw[0..4], &0x3143_3247u32.to_le_bytes());
    assert_eq!(&raw[4..8], &2048u32.to_le_bytes());
    assert_eq!(&raw[8..16], &5u64.to_le_bytes());
    assert_eq!(raw.len(), 16 + 5 * 64 * 4);
    let back = read_candidates(&p).unwrap();
    assert_eq!(back.bits, 2048);
    assert_eq!(back.numbers, ns);
    std::fs::remove_file(p).unwrap();
}

#[test]
fn verdict_file_layout_and_atomic_write() {
    let p = tmp("verdict.bin");
    write_verdicts(&p, 2048, &[0, 1, 0, 2]).unwrap();
    let raw = std::fs::read(&p).unwrap();
    assert_eq!(&raw[0..4], &0x3156_3247u32.to_le_bytes());
    assert_eq!(&raw[8..16], &4u64.to_le_bytes());
    assert_eq!(&raw[16..], &[0, 1, 0, 2]);
    assert!(!std::path::Path::new(&format!("{}.tmp", p.display())).exists());
    assert_eq!(read_verdicts(&p).unwrap(), (2048, vec![0, 1, 0, 2]));
    std::fs::remove_file(p).unwrap();
}

#[test]
fn malformed_files_are_errors() {
    let p = tmp("bad.bin");
    std::fs::write(&p, [0u8; 20]).unwrap();
    assert!(read_candidates(&p).is_err());
    let mut good = Vec::new();
    good.extend_from_slice(&0x3143_3247u32.to_le_bytes());
    good.extend_from_slice(&64u32.to_le_bytes());
    good.extend_from_slice(&3u64.to_le_bytes());
    good.extend_from_slice(&[0u8; 8]); // claims 3 numbers of 2 limbs, has 1
    std::fs::write(&p, &good).unwrap();
    assert!(read_candidates(&p).is_err());
    std::fs::remove_file(p).unwrap();
}

fn header(magic: u32, bits: u32, count: u64) -> Vec<u8> {
    let mut v = Vec::new();
    v.extend_from_slice(&magic.to_le_bytes());
    v.extend_from_slice(&bits.to_le_bytes());
    v.extend_from_slice(&count.to_le_bytes());
    v
}

#[test]
fn absurd_counts_are_rejected_without_allocating() {
    let p = tmp("huge.bin");
    std::fs::write(&p, header(0x3143_3247, 2048, u64::MAX)).unwrap();
    assert!(read_candidates(&p).is_err());
    std::fs::write(&p, header(0x3143_3247, u32::MAX, 1 << 40)).unwrap();
    assert!(read_candidates(&p).is_err());
    std::fs::write(&p, header(0x3156_3247, 2048, u64::MAX)).unwrap();
    assert!(read_verdicts(&p).is_err());
    std::fs::remove_file(p).unwrap();
}

#[test]
fn verdict_file_rejects_wrong_magic_and_length() {
    let p = tmp("badv.bin");
    let mut v = header(0x3143_3247, 2048, 1); // candidate magic, not verdict
    v.push(1);
    std::fs::write(&p, &v).unwrap();
    assert!(read_verdicts(&p).is_err());
    let mut v = header(0x3156_3247, 2048, 3);
    v.extend_from_slice(&[0, 1]); // one byte short
    std::fs::write(&p, &v).unwrap();
    assert!(read_verdicts(&p).is_err());
    v.extend_from_slice(&[1, 1]); // one byte long
    std::fs::write(&p, &v).unwrap();
    assert!(read_verdicts(&p).is_err());
    std::fs::remove_file(p).unwrap();
}

#[test]
fn a_number_wider_than_the_container_is_not_written() {
    let p = tmp("wide.bin");
    let wide = (BigUint::from(1u32) << 100u32) + 1u32;
    assert!(write_candidates(&p, 64, &[wide]).is_err());
    let _ = std::fs::remove_file(p);
}

fn sizes_bytes(magic: u32, nk: u32, counts: &[u32]) -> Vec<u8> {
    let mut v = Vec::new();
    v.extend_from_slice(&magic.to_le_bytes());
    v.extend_from_slice(&nk.to_le_bytes());
    for c in counts {
        v.extend_from_slice(&c.to_le_bytes());
    }
    v
}

#[test]
fn sizes_file_round_trips_and_has_the_beacon_layout() {
    let p = tmp("sizes.bin");
    write_sizes(&p, &[3, 0, 7, 1]).unwrap();
    let raw = std::fs::read(&p).unwrap();
    assert_eq!(raw, sizes_bytes(0x3153_3247, 4, &[3, 0, 7, 1]));
    assert_eq!(SIZES_MAGIC, 0x3153_3247);
    assert_eq!(read_sizes(&p).unwrap(), vec![3, 0, 7, 1]);
    write_sizes(&p, &[]).unwrap();
    assert_eq!(read_sizes(&p).unwrap(), Vec::<u32>::new());
    std::fs::remove_file(p).unwrap();
}

#[test]
fn malformed_sizes_files_are_errors() {
    let p = tmp("badsizes.bin");
    assert!(read_sizes(&p).is_err()); // missing
    std::fs::write(&p, []).unwrap();
    assert!(read_sizes(&p).is_err()); // empty
    std::fs::write(&p, &sizes_bytes(0x3153_3247, 0, &[])[..7]).unwrap();
    assert!(read_sizes(&p).is_err()); // header cut short
    std::fs::write(&p, sizes_bytes(0x3143_3247, 1, &[5])).unwrap();
    assert!(read_sizes(&p).is_err()); // wrong magic
    let mut cut = sizes_bytes(0x3153_3247, 3, &[1, 2, 3]);
    cut.pop();
    std::fs::write(&p, &cut).unwrap();
    assert!(read_sizes(&p).is_err()); // body truncated mid-entry
    std::fs::write(&p, sizes_bytes(0x3153_3247, 3, &[1, 2])).unwrap();
    assert!(read_sizes(&p).is_err()); // fewer entries than claimed
    std::fs::write(&p, sizes_bytes(0x3153_3247, u32::MAX, &[1])).unwrap();
    assert!(read_sizes(&p).is_err()); // absurd count, rejected before any allocation
    std::fs::remove_file(p).unwrap();
}

#[test]
fn sizes_file_one_entry_short_of_its_claim_is_an_error() {
    let p = tmp("shortsizes.bin");
    std::fs::write(&p, sizes_bytes(0x3153_3247, 3, &[1, 2])).unwrap();
    let err = read_sizes(&p).unwrap_err();
    assert!(
        err.contains("claims 3 entries but only has 8 bytes"),
        "{err}"
    );
    std::fs::remove_file(p).unwrap();
}

#[test]
fn sizes_file_ignores_bytes_after_the_declared_entries() {
    let p = tmp("trailing.bin");
    std::fs::write(&p, sizes_bytes(0x3153_3247, 1, &[1, 2])).unwrap(); // a whole extra entry
    assert_eq!(read_sizes(&p).unwrap(), vec![1]);
    let mut v = sizes_bytes(0x3153_3247, 2, &[4, 5]);
    v.extend_from_slice(&[9, 9, 9]); // stray bytes, not a multiple of 4
    std::fs::write(&p, &v).unwrap();
    assert_eq!(read_sizes(&p).unwrap(), vec![4, 5]);
    std::fs::write(&p, sizes_bytes(0x3153_3247, 0, &[7])).unwrap();
    assert_eq!(read_sizes(&p).unwrap(), Vec::<u32>::new());
    std::fs::remove_file(p).unwrap();
}

#[test]
fn bin_list_strips_line_endings_and_trailing_spaces_and_skips_blanks() {
    let p = tmp("list.txt");
    std::fs::write(&p, "a.bin\r\n\r\n  \nb dir/b.bin  \n\nc.bin").unwrap();
    let got = read_bin_list(&p).unwrap();
    assert_eq!(
        got,
        vec![
            PathBuf::from("a.bin"),
            PathBuf::from("b dir/b.bin"),
            PathBuf::from("c.bin")
        ]
    );
    std::fs::remove_file(p).unwrap();
}

#[test]
fn bin_list_keeps_a_leading_space() {
    let p = tmp("leadlist.txt");
    std::fs::write(&p, " lead.bin \n").unwrap();
    assert_eq!(read_bin_list(&p).unwrap(), vec![PathBuf::from(" lead.bin")]);
    std::fs::remove_file(p).unwrap();
}

#[test]
fn bin_list_without_paths_or_missing_is_an_error() {
    let p = tmp("emptylist.txt");
    std::fs::write(&p, "").unwrap();
    assert!(read_bin_list(&p).is_err());
    std::fs::write(&p, "\r\n  \n\n").unwrap();
    assert!(read_bin_list(&p).is_err());
    std::fs::remove_file(&p).unwrap();
    let err = read_bin_list(&p).unwrap_err();
    assert!(err.contains(&p.display().to_string()), "{err}");
}

#[test]
fn candidate_files_concatenate_in_listed_order() {
    let mut rng = XorShift64::new(7);
    let a: Vec<BigUint> = (0..3).map(|_| random_odd_modulus(&mut rng, 500)).collect();
    let b: Vec<BigUint> = (0..2).map(|_| random_odd_modulus(&mut rng, 500)).collect();
    let (pa, pb) = (tmp("cat-a.bin"), tmp("cat-b.bin"));
    write_candidates(&pa, 512, &a).unwrap();
    write_candidates(&pb, 512, &b).unwrap();
    let ab = read_candidate_files(&[pa.clone(), pb.clone()]).unwrap();
    assert_eq!(ab.bits, 512);
    assert_eq!(ab.numbers, [a.clone(), b.clone()].concat());
    let ba = read_candidate_files(&[pb.clone(), pa.clone()]).unwrap();
    assert_eq!(ba.numbers, [b, a].concat());
    let one = read_candidate_files(std::slice::from_ref(&pa)).unwrap();
    assert_eq!(one, read_candidates(&pa).unwrap());
    std::fs::remove_file(pa).unwrap();
    std::fs::remove_file(pb).unwrap();
}

#[test]
fn candidate_files_with_different_container_widths_are_rejected() {
    let mut rng = XorShift64::new(8);
    let a = vec![random_odd_modulus(&mut rng, 500)];
    let (pa, pb) = (tmp("mix-a.bin"), tmp("mix-b.bin"));
    write_candidates(&pa, 512, &a).unwrap();
    write_candidates(&pb, 1024, &a).unwrap();
    let err = read_candidate_files(&[pa.clone(), pb.clone()]).unwrap_err();
    assert!(err.contains("mixed bits"), "{err}");
    std::fs::remove_file(pa).unwrap();
    std::fs::remove_file(pb).unwrap();
}

#[test]
fn candidate_files_name_a_missing_file_and_reject_an_empty_list() {
    let mut rng = XorShift64::new(9);
    let pa = tmp("miss-a.bin");
    write_candidates(&pa, 512, &[random_odd_modulus(&mut rng, 500)]).unwrap();
    let missing = tmp("miss-gone.bin");
    let err = read_candidate_files(&[pa.clone(), missing.clone()]).unwrap_err();
    assert!(err.contains(&missing.display().to_string()), "{err}");
    assert!(read_candidate_files(&[]).is_err());
    std::fs::remove_file(pa).unwrap();
}
