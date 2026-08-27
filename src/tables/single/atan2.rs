//! `atan2f` data.
//!
//! **Hand-transcribed, not generated** — the third and last such file; see
//! [`super::trig`] for the reason, which is the same one. `rmath` delegates
//! `atan2f` to the platform, so its generator never emitted this.
//!
//! Transcribed from glibc's `sysdeps/ieee754/flt-32/e_atan2f.c` (CORE-MATH's
//! `cr_atan2f`). Copyright (C) 2025-2026 Free Software Foundation, Inc.,
//! SPDX-License-Identifier: `LGPL-2.1-or-later`. Values are exact bit
//! patterns; the trailing comment is upstream's own hex-float spelling.

/// The main path's rational numerator, in `z^2`.
pub const CN: [u64; 7] = [
    0x3ff0000000000000, // 0x1p+0
    0x40040e0698f94c35, // 0x1.40e0698f94c35p+1
    0x400248c5da347f0d, // 0x1.248c5da347f0dp+1
    0x3fed873386572976, // 0x1.d873386572976p-1
    0x3fc46fa40b20f1d0, // 0x1.46fa40b20f1dp-3
    0x3f833f5e041eed0f, // 0x1.33f5e041eed0fp-7
    0x3f1546bbf28667c5, // 0x1.546bbf28667c5p-14
];

/// The main path's rational denominator, in `z^2`.
pub const CD: [u64; 7] = [
    0x3ff0000000000000, // 0x1p+0
    0x4006b8b143a3f6da, // 0x1.6b8b143a3f6dap+1
    0x4008421201d18ed5, // 0x1.8421201d18ed5p+1
    0x3ff8221d086914eb, // 0x1.8221d086914ebp+0
    0x3fd670657e3a07ba, // 0x1.670657e3a07bap-2
    0x3fa0f4951fd1e72d, // 0x1.0f4951fd1e72dp-5
    0x3f4b3874b8798286, // 0x1.b3874b8798286p-11
];

/// The accurate path's degree-31 `atan(z)/z` series, in `z^2`, as
/// double-double `[hi, lo]` pairs.
pub const C: [u64; 64] = [
    0x3ff0000000000000, 0xba88c1dac5492248, // 0x1p+0, -0x1.8c1dac5492248p-87
    0xbfd5555555555555, 0xbc755553bf3a2abe, // -0x1.5555555555555p-2, -0x1.55553bf3a2abep-56
    0x3fc999999999999a, 0xbc699deed1ec9071, // 0x1.999999999999ap-3, -0x1.99deed1ec9071p-57
    0xbfc2492492492492, 0xbc5fd99c8d18269a, // -0x1.2492492492492p-3, -0x1.fd99c8d18269ap-58
    0x3fbc71c71c71c717, 0xbc2651eee4c4d9d0, // 0x1.c71c71c71c717p-4, -0x1.651eee4c4d9dp-61
    0xbfb745d1745d1649, 0xbc5632683d6c44a6, // -0x1.745d1745d1649p-4, -0x1.632683d6c44a6p-58
    0x3fb3b13b13b11c63, 0x3c5bf69c1f8af41d, // 0x1.3b13b13b11c63p-4, 0x1.bf69c1f8af41dp-58
    0xbfb11111110e6338, 0x3c23c3e431e8bb68, // -0x1.11111110e6338p-4, 0x1.3c3e431e8bb68p-61
    0x3fae1e1e1dc45c4a, 0xbc4be2db05c77bbf, // 0x1.e1e1e1dc45c4ap-5, -0x1.be2db05c77bbfp-59
    0xbfaaf286b8164b4f, 0x3c2a4673491f0942, // -0x1.af286b8164b4fp-5, 0x1.a4673491f0942p-61
    0x3fa86185e9ad4846, 0x3c4e12e32d79fcee, // 0x1.86185e9ad4846p-5, 0x1.e12e32d79fceep-59
    0xbfa642c6d5161fae, 0x3c43ce76c1ca03f0, // -0x1.642c6d5161faep-5, 0x1.3ce76c1ca03fp-59
    0x3fa47ad6f277e5bf, 0xbc3abd8d85bdb714, // 0x1.47ad6f277e5bfp-5, -0x1.abd8d85bdb714p-60
    0xbfa2f64a2ee8896d, 0x3c2ef87d4b615323, // -0x1.2f64a2ee8896dp-5, 0x1.ef87d4b615323p-61
    0x3fa1a6a2b31741b5, 0x3c1a5d9d973547ee, // 0x1.1a6a2b31741b5p-5, 0x1.a5d9d973547eep-62
    0xbfa07fbdad65e0a6, 0xbc265ac07f5d35f4, // -0x1.07fbdad65e0a6p-5, -0x1.65ac07f5d35f4p-61
    0x3f9ee9932a9a5f8b, 0x3c2f8b9623f6f55a, // 0x1.ee9932a9a5f8bp-6, 0x1.f8b9623f6f55ap-61
    0xbf9ce8b5b9584dc6, 0x3c2fe5af96e8ea2d, // -0x1.ce8b5b9584dc6p-6, 0x1.fe5af96e8ea2dp-61
    0x3f9ac9cb288087b7, 0xbc3450cdfceaf5ca, // 0x1.ac9cb288087b7p-6, -0x1.450cdfceaf5cap-60
    0xbf984b025351f3e6, 0x3c2579561b0d73da, // -0x1.84b025351f3e6p-6, 0x1.579561b0d73dap-61
    0x3f952f5b8ecdd52b, 0x3c3036bd2c6fba47, // 0x1.52f5b8ecdd52bp-6, 0x1.036bd2c6fba47p-60
    0xbf9163a8c44909dc, 0x3c318f735ffb9f16, // -0x1.163a8c44909dcp-6, 0x1.18f735ffb9f16p-60
    0x3f8a400dce3eea6f, 0xbc2c90569c0c1b5c, // 0x1.a400dce3eea6fp-7, -0x1.c90569c0c1b5cp-61
    0xbf81caa78ae6db3a, 0xbc24c60f8161ea09, // -0x1.1caa78ae6db3ap-7, -0x1.4c60f8161ea09p-61
    0x3f752672453c0731, 0x3c1834efb598c338, // 0x1.52672453c0731p-8, 0x1.834efb598c338p-62
    0xbf65850c5be137cf, 0xbc0445fc150ca7f5, // -0x1.5850c5be137cfp-9, -0x1.445fc150ca7f5p-63
    0x3f523eb98d22e1ca, 0xbbf388fbaf1d7830, // 0x1.23eb98d22e1cap-10, -0x1.388fbaf1d783p-64
    0xbf38f4e974a40741, 0x3bd271198a97da34, // -0x1.8f4e974a40741p-12, 0x1.271198a97da34p-66
    0x3f1a5cf2e9cf76e5, 0xbbb887eb4a63b665, // 0x1.a5cf2e9cf76e5p-14, -0x1.887eb4a63b665p-68
    0xbef420c270719e32, 0x3b8efd595b27888b, // -0x1.420c270719e32p-16, 0x1.efd595b27888bp-71
    0x3ec3ba2d69b51677, 0xbb64fb06829cdfc7, // 0x1.3ba2d69b51677p-19, -0x1.4fb06829cdfc7p-73
    0xbe829b7e6f676385, 0xbb2a783b6de718fb, // -0x1.29b7e6f676385p-23, -0x1.a783b6de718fbp-77
];

/// `pi`. Upstream: `0x1.921fb54442d18p+1`.
pub const PI: f64 = f64::from_bits(0x400921fb54442d18);

/// `pi/2`. Upstream: `0x1.921fb54442d18p+0`.
pub const PI2: f64 = f64::from_bits(0x3ff921fb54442d18);

/// `pi/2`'s low half. Upstream: `0x1.1a62633145c07p-54`.
pub const PI2L: f64 = f64::from_bits(0x3c91a62633145c07);

/// `3 pi/4`, for `atan2(+-inf, -inf)`. Upstream: `0x1.2d97c7f3321d2p+1`.
pub const TQPI: f64 = f64::from_bits(0x4002d97c7f3321d2);

/// `pi/4`, for `atan2(+-inf, +inf)`. Upstream: `0x1.921fb54442d18p-1`.
pub const QPI: f64 = f64::from_bits(0x3fe921fb54442d18);

/// `-1/3`, the tiny path's only coefficient. Upstream: `-0x1.5555555555555p-2`.
pub const MTHIRD: f64 = f64::from_bits(0xbfd5555555555555);

