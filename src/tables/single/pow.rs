//! `powf`'s logarithm data.
//!
//! **Hand-transcribed, not generated** — see [`super::trig`] for why this
//! crate has two such files and only two. `rmath` delegates `powf` to the
//! platform, so its generator never emitted this table.
//!
//! Transcribed from glibc's `sysdeps/ieee754/flt-32/e_powf_log2_data.c`.
//! Copyright (C) 2017-2026 Free Software Foundation, Inc.,
//! SPDX-License-Identifier: `LGPL-2.1-or-later`. Values are exact bit
//! patterns; the trailing comment is upstream's own hex-float spelling.
//!
//! Every value upstream writes is multiplied by `POWF_SCALE`, which is `1`
//! wherever the target lacks round-and-convert intrinsics — x86-64 included —
//! so the products are the values themselves. See [`super::trig`]'s note on
//! the same fork.

/// `[1/c, log2 c]` for 16 subintervals of `[0x1.66p-1, 0x1.66p0]`.
pub static TAB: [u64; 32] = [
    0x3ff661ec79f8f3be, 0xbfdefec65b963019, // 0x1.661ec79f8f3bep+0, -0x1.efec65b963019p-2
    0x3ff571ed4aaf883d, 0xbfdb0b6832d4fca4, // 0x1.571ed4aaf883dp+0, -0x1.b0b6832d4fca4p-2
    0x3ff49539f0f010b0, 0xbfd7418b0a1fb77b, // 0x1.49539f0f010bp+0,  -0x1.7418b0a1fb77bp-2
    0x3ff3c995b0b80385, 0xbfd39de91a6dcf7b, // 0x1.3c995b0b80385p+0, -0x1.39de91a6dcf7bp-2
    0x3ff30d190c8864a5, 0xbfd01d9bf3f2b631, // 0x1.30d190c8864a5p+0, -0x1.01d9bf3f2b631p-2
    0x3ff25e227b0b8ea0, 0xbfc97c1d1b3b7af0, // 0x1.25e227b0b8eap+0,  -0x1.97c1d1b3b7afp-3
    0x3ff1bb4a4a1a343f, 0xbfc2f9e393af3c9f, // 0x1.1bb4a4a1a343fp+0, -0x1.2f9e393af3c9fp-3
    0x3ff12358f08ae5ba, 0xbfb960cbbf788d5c, // 0x1.12358f08ae5bap+0, -0x1.960cbbf788d5cp-4
    0x3ff0953f419900a7, 0xbfaa6f9db6475fce, // 0x1.0953f419900a7p+0, -0x1.a6f9db6475fcep-5
    0x3ff0000000000000, 0x0000000000000000, // 0x1p+0,               0x0p+0
    0x3fee608cfd9a47ac, 0x3fb338ca9f24f53d, // 0x1.e608cfd9a47acp-1, 0x1.338ca9f24f53dp-4
    0x3feca4b31f026aa0, 0x3fc476a9543891ba, // 0x1.ca4b31f026aap-1,  0x1.476a9543891bap-3
    0x3feb2036576afce6, 0x3fce840b4ac4e4d2, // 0x1.b2036576afce6p-1, 0x1.e840b4ac4e4d2p-3
    0x3fe9c2d163a1aa2d, 0x3fd40645f0c6651c, // 0x1.9c2d163a1aa2dp-1, 0x1.40645f0c6651cp-2
    0x3fe886e6037841ed, 0x3fd88e9c2c1b9ff8, // 0x1.886e6037841edp-1, 0x1.88e9c2c1b9ff8p-2
    0x3fe767dcf5534862, 0x3fdce0a44eb17bcc, // 0x1.767dcf5534862p-1, 0x1.ce0a44eb17bccp-2
];

/// The degree-5 `log1p(r)/ln2` polynomial's leading coefficient.
/// Upstream: `0x1.27616c9496e0bp-2`.
pub const A0: f64 = f64::from_bits(0x3fd27616c9496e0b);
/// Upstream: `-0x1.71969a075c67ap-2`.
pub const A1: f64 = f64::from_bits(0xbfd71969a075c67a);
/// Upstream: `0x1.ec70a6ca7baddp-2`.
pub const A2: f64 = f64::from_bits(0x3fdec70a6ca7badd);
/// Upstream: `-0x1.7154748bef6c8p-1`.
pub const A3: f64 = f64::from_bits(0xbfe7154748bef6c8);
/// Upstream: `0x1.71547652ab82bp0`.
pub const A4: f64 = f64::from_bits(0x3ff71547652ab82b);

/// The table's centring offset, `bits(0x1.66p-1)`.
pub const OFF: u32 = 0x3f33_0000;

/// Above this `y log2(x)`, `x^y` overflows outright.
/// Upstream: `0x1.fffffffd1d571p+6`.
pub const OFLOW_HI: f64 = f64::from_bits(0x405fffffffd1d571);
/// Above this it overflows only under a rounding mode that rounds away from
/// zero; in round-to-nearest the answer is `FLT_MAX`.
/// Upstream: `0x1.fffffffa3aae2p+6`.
pub const OFLOW_LO: f64 = f64::from_bits(0x405fffffffa3aae2);
/// The top 16 magnitude bits of `126.0`, which is how `powf` tests whether
/// `|y log2 x|` has left the main path's range.
pub const GUARD_TOP: u64 = 0x80bf;
