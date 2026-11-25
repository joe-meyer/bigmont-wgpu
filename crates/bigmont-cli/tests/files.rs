use bigmont_cli::formats::*;
use bigmont_oracle::{XorShift64, random_odd_modulus};
use num_bigint::BigUint;

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
