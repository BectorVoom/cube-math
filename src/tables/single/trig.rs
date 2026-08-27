//! `sinf` / `cosf` / `sincosf` data.
//!
//! **Hand-transcribed, not generated.** Every other table in this crate comes
//! out of `rmath`'s `tools/gen_tables.py`, which reads glibc's own headers.
//! This one does not exist there, because `rmath` never ported these three:
//! on a CPU its `BitExact` path calls the platform's `sinf` lane by lane,
//! which is exact by construction and cheaper than anything it could compile.
//! A kernel has no `sinf` to call, so the data had to come from somewhere.
//!
//! Transcribed from glibc's `sysdeps/ieee754/flt-32/s_sincosf_data.c` and
//! `s_sincosf.h`. Copyright (C) 2018-2026 Free Software Foundation, Inc.,
//! SPDX-License-Identifier: `LGPL-2.1-or-later`. Values are exact bit
//! patterns; the trailing comment is upstream's own hex-float spelling.
//!
//! # One entry, not two
//!
//! Upstream's `__sincosf_table` has two entries, and the second is the first
//! with the *cosine* polynomial negated — a trick that gets the quadrant's
//! negation for free by folding it into a table index. Negation is exact, so
//! evaluating the polynomial from entry zero and negating the result is
//! bit-for-bit the same computation; the kernel does that and this table holds
//! one entry. See [`crate::single::trig`].
//!
//! # `hpi_inv` is the pre-scaled one
//!
//! Upstream keeps two spellings of `2/pi` behind `TOINT_INTRINSICS`: the plain
//! value where the target has round-and-convert instructions (aarch64), and
//! one pre-scaled by `2^24` where it does not, so that the quadrant lands in
//! bits 24..31 of a truncating conversion. glibc on x86-64 compiles the second,
//! and the second is therefore what "bit-exact" means here.

/// `2/pi`, pre-scaled by `2^24`. Upstream: `0x1.45F306DC9C883p+23`.
pub const HPI_INV: f64 = f64::from_bits(0x41645f306dc9c883);

/// `pi/2`. Upstream: `0x1.921FB54442D18p0`.
pub const HPI: f64 = f64::from_bits(0x3ff921fb54442d18);

/// The cosine polynomial's constant term. Upstream: `0x1p0`.
pub const C0: f64 = f64::from_bits(0x3ff0000000000000);

/// Upstream: `-0x1.ffffffd0c621cp-2`.
pub const C1: f64 = f64::from_bits(0xbfdffffffd0c621c);

/// Upstream: `0x1.55553e1068f19p-5`.
pub const C2: f64 = f64::from_bits(0x3fa55553e1068f19);

/// Upstream: `-0x1.6c087e89a359dp-10`.
pub const C3: f64 = f64::from_bits(0xbf56c087e89a359d);

/// Upstream: `0x1.99343027bf8c3p-16`.
pub const C4: f64 = f64::from_bits(0x3ef99343027bf8c3);

/// The sine polynomial's cubic coefficient. Upstream: `-0x1.555545995a603p-3`.
pub const S1: f64 = f64::from_bits(0xbfc555545995a603);

/// Upstream: `0x1.1107605230bc4p-7`.
pub const S2: f64 = f64::from_bits(0x3f81107605230bc4);

/// Upstream: `-0x1.994eb3774cf24p-13`.
pub const S3: f64 = f64::from_bits(0xbf2994eb3774cf24);

/// `2 pi * 2^-64`, the scale `reduce_large` puts its fixed-point residue back
/// on. Upstream: `0x1.921FB54442D18p-62`.
pub const PI63: f64 = f64::from_bits(0x3c1921fb54442d18);

/// `pi/4`, the small-argument cutoff. Upstream: `0x1.921FB6p-1f`.
pub const PIO4: f32 = f32::from_bits(0x3f490fdb);

/// `4/pi` to 192 bits, as 24 overlapping 32-bit windows.
///
/// Upstream stores 8 new bits per entry rather than 32, which makes the table
/// four times larger and every access aligned — the reduction reads `arr[0]`,
/// `arr[4]` and `arr[8]` from a base chosen by the argument's exponent, and
/// the overlap is what lets that base move in single-byte steps.
pub const INV_PIO4: [u32; 24] = [
    0xa2, 0xa2f9, 0xa2f983, 0xa2f9836e, //
    0xf9836e4e, 0x836e4e44, 0x6e4e4415, 0x4e441529, //
    0x441529fc, 0x1529fc27, 0x29fc2757, 0xfc2757d1, //
    0x2757d1f5, 0x57d1f534, 0xd1f534dd, 0xf534ddc0, //
    0x34ddc0db, 0xddc0db62, 0xc0db6295, 0xdb629599, //
    0x6295993c, 0x95993c43, 0x993c4390, 0x3c439041,
];
