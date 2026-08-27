//! `erf(x)`, correctly rounded.
//!
//! # Why this one is not like the others
//!
//! Everywhere else in this crate, bit-exactness is a claim about a platform:
//! reproduce glibc's operation schedule and you match glibc, and a different
//! `libm` would need a different port. `erf` is the exception. glibc's is
//! CORE-MATH's, and CORE-MATH's is **correctly rounded** — it returns the
//! representable value nearest the true `erf(x)` for every input. Correct
//! rounding is a property of the *answer*, not of the route to it, so any
//! correctly-rounded implementation agrees with any other on every input. The
//! port below is therefore bit-exact to glibc *and* to every other correctly
//! rounded `erf`, on every platform, forever.
//!
//! # The two-step scheme
//!
//! 1. A **fast path** evaluates `erf` as a double-double `h + l` with a proven
//!    relative error bound `err`, about `2^-69`. It is straight-line
//!    arithmetic over one table row.
//! 2. A **rounding test** asks whether `h + l` is far enough from a rounding
//!    boundary that the bound settles which way it goes: round `h + l - err`
//!    and `h + l + err` and see whether they agree. They almost always do —
//!    the bound is some `2^15` times narrower than an ulp — so the test fails
//!    on roughly one input in thirty thousand.
//! 3. Those few take an **accurate path**: a degree-18 double-double
//!    polynomial, plus a short table of arguments that even it cannot resolve.
//!
//! On a CPU vector unit step 3 is the awkward one: one lane failing the test
//! drags the whole register onto the slow path, so `rmath` runs it scalar.
//! Here it is just a branch. The threads that fail diverge for as long as it
//! takes and the rest of the warp waits, and at one input in thirty thousand
//! most warps never take it at all.
//!
//! Both policy axes are accepted and have no effect. There is no cheaper `erf`
//! worth having that is still correctly rounded, and correct rounding is the
//! whole point of this one.

use cubecl::prelude::*;

use crate::bits::{is_nan64, opaque64};
use crate::config::MathConfig;
use crate::double::dd::{a_mul, fast_two_sum, two_sum};
use crate::double::exact::copysign;
use crate::fma::{FmaKind, fma64};
use crate::tables::consts::{
    erf_c0_tab, erf_c2_tab, erf_c_tab, erf_exc_tab, erf_exc_tiny_tab, erf_p_tab,
};

/// `CH + CL` is `2/sqrt(pi)` to double-double precision.
pub const CH: f64 = f64::from_bits(0x3ff20dd750429b6d);
/// The low half of `2/sqrt(pi)`. See [`CH`].
pub const CL: f64 = f64::from_bits(0x3c71ae3a914fed80);
/// The largest `x` with `erf(x) != 1` after rounding.
pub const ERF_MAX: u64 = 0x4017afb48dc96626;
/// Below this magnitude `erf(x)` is `2/sqrt(pi) x` and nothing else: `2^-61`.
pub const ERF_TINY: f64 = f64::from_bits(0x3c20000000000000);
/// The relative error the fast path holds to for `z < 1/16`.
pub const ERR_SMALL: f64 = f64::from_bits(0x3ba7800000000000);
/// The relative error the fast path holds to for `z >= 1/16`.
pub const ERR_TABLE: f64 = f64::from_bits(0x3ba1100000000000);
/// Entries in the near-zero hard-case table.
pub const EXC_TINY_ROWS: u32 = 171;
/// Entries in the general hard-case table.
pub const EXC_ROWS: u32 = 5;
/// `2^106`: the scaling that keeps the near-zero correction term out of the
/// subnormal range, where it would be flushed.
const P106: f64 = f64::from_bits(0x4690000000000000);
/// `2^-106`. See [`P106`].
const M106: f64 = f64::from_bits(0x3950000000000000);
/// Half an ulp of one, which steps `1` down to its neighbour below.
const HALF_ULP1: f64 = f64::from_bits(0x3c90000000000000);

// ---------------------------------------------------------------------------
// The double-double Horner steps
// ---------------------------------------------------------------------------

/// One step of an even double-double Horner fold: `(h, l) <- c + z (h + l)`.
#[cube]
pub fn dd_step(h: f64, l: f64, z: f64, c: f64, #[comptime] fk: FmaKind) -> (f64, f64) {
    let (th, tl0) = a_mul(h, z, fk);
    let tl = fma64(l, z, tl0, fk);
    let (nh, nl) = two_sum(c, th);
    (nh, nl + tl)
}

/// [`dd_step()`] with a double-double coefficient `chi + clo`.
#[cube]
pub fn dd_step2(h: f64, l: f64, z: f64, chi: f64, clo: f64, #[comptime] fk: FmaKind) -> (f64, f64) {
    let (th, tl0) = a_mul(h, z, fk);
    let tl = fma64(l, z, tl0, fk);
    let (nh, nl) = two_sum(chi, th);
    (nh, nl + (clo + tl))
}

/// One step of the *odd* fold the near-zero polynomial uses: two multiplies by
/// `z` per coefficient, because only odd powers appear.
#[cube]
pub fn dd_step_odd(h: f64, l: f64, z: f64, c: f64, #[comptime] fk: FmaKind) -> (f64, f64) {
    let (th, tl0) = a_mul(h, z, fk);
    let tl = fma64(l, z, tl0, fk);
    let (h2, nl) = a_mul(th, z, fk);
    let l2 = fma64(tl, z, nl, fk);
    let (h3, tl2) = fast_two_sum(c, h2);
    (h3, l2 + tl2)
}

/// [`dd_step_odd()`] with a double-double coefficient `chi + clo`.
#[cube]
pub fn dd_step_odd2(
    h: f64,
    l: f64,
    z: f64,
    chi: f64,
    clo: f64,
    #[comptime] fk: FmaKind,
) -> (f64, f64) {
    let (th, tl0) = a_mul(h, z, fk);
    let tl = fma64(l, z, tl0, fk);
    let (h2, nl) = a_mul(th, z, fk);
    let l2 = fma64(tl, z, nl, fk);
    let (h3, tl2) = fast_two_sum(chi, h2);
    (h3, l2 + (clo + tl2))
}

// ---------------------------------------------------------------------------
// The fast path
// ---------------------------------------------------------------------------

/// `erf(z)` as `(h, l, err)` for `0 <= z <= 0x1.7afb48dc96626p+2`.
///
/// `err` is the *relative* error bound: `|(h + l)/erf(z) - 1| < err`.
#[cube]
pub fn erf_fast(z0: f64, #[comptime] fk: FmaKind) -> (f64, f64, f64) {
    let mut rh = 0.0 + z0;
    let mut rl = 0.0 * z0;
    if z0 < 0.0625 {
        // Evaluated at zero rather than at a midpoint: `z - 1/32` would not be
        // exact for a tiny `z`, and a midpoint fit loses all its relative
        // accuracy there anyway.
        let c0 = erf_c0_tab();
        let (z2h, z2l) = a_mul(z0, z0, fk);
        let z4 = z2h * z2h;
        let c9 = fma64(f64::reinterpret(c0[7]), z2h, f64::reinterpret(c0[6]), fk);
        let c5a = fma64(f64::reinterpret(c0[5]), z2h, f64::reinterpret(c0[4]), fk);
        let c5 = fma64(c9, z4, c5a, fk);

        let (th, tl) = a_mul(z2h, c5, fk);
        let (h0, l0) = fast_two_sum(f64::reinterpret(c0[2]), th);
        let l1 = l0 + (tl + f64::reinterpret(c0[3]));

        let (th2, tl2) = a_mul(z2h, h0, fk);
        let tl3 = tl2 + fma64(z2h, l1, f64::reinterpret(c0[1]), fk);
        let (h2, l2) = fast_two_sum(f64::reinterpret(c0[0]), th2);
        let l3 = l2 + fma64(z2l, h0, tl3, fk);

        let (h3, tl4) = a_mul(h2, z0, fk);
        rh = h3;
        rl = fma64(l3, z0, tl4, fk);
    } else {
        let i = u32::cast_from(16.0 * z0);
        let v = f64::cast_from(i);
        // Exact: `0.03125` is an integer multiple of `ulp(z)` in `z`'s binade,
        // or both are multiples of the smaller binade's, and `0.0625 v` is
        // exact for the same reason.
        let z = (z0 - 0.03125) - 0.0625 * v;
        let tab = erf_c_tab();
        let c = usize::cast_from((i - 1u32) * 13u32);

        let z2 = z * z;
        let z4 = z2 * z2;
        let c9 = fma64(f64::reinterpret(tab[c + 12]), z, f64::reinterpret(tab[c + 11]), fk);
        let c7a = fma64(f64::reinterpret(tab[c + 10]), z, f64::reinterpret(tab[c + 9]), fk);
        let c5 = fma64(f64::reinterpret(tab[c + 8]), z, f64::reinterpret(tab[c + 7]), fk);
        let (c3h0, c3l0) =
            fast_two_sum(f64::reinterpret(tab[c + 5]), z * f64::reinterpret(tab[c + 6]));
        let c7 = fma64(c9, z2, c7a, fk);

        let (c3h1, tl5) = fast_two_sum(c3h0, c5 * z2);
        let c3l1 = c3l0 + tl5;
        let (c3h, tl6) = fast_two_sum(c3h1, c7 * z4);
        let c3l = c3l1 + tl6;

        let (th, tl) = a_mul(z, c3h, fk);
        let (c2h, c2l0) = fast_two_sum(f64::reinterpret(tab[c + 4]), th);
        let c2l = c2l0 + fma64(z, c3l, tl, fk);

        let (th2, tl2) = a_mul(z, c2h, fk);
        let (h6, l6) = fast_two_sum(f64::reinterpret(tab[c + 2]), th2);
        let l7 = l6 + (tl2 + fma64(z, c2l, f64::reinterpret(tab[c + 3]), fk));

        let (th3, tl3) = a_mul(z, h6, fk);
        let tl4 = fma64(z, l7, tl3, fk);
        let (h7, l8) = fast_two_sum(f64::reinterpret(tab[c]), th3);
        rh = h7;
        rl = l8 + tl4 + f64::reinterpret(tab[c + 1]);
    }
    (rh, rl, select(z0 < 0.0625, ERR_SMALL, ERR_TABLE))
}

// ---------------------------------------------------------------------------
// The accurate path
// ---------------------------------------------------------------------------

/// The accurate path for `|z| < 1/8`, as a degree-21 odd double-double
/// polynomial.
///
/// `exceptions` selects whether the hard-case table is consulted: `erf` needs
/// it, `erfc` does not, because the two round at different places.
///
/// The table is consulted *after* the polynomial rather than instead of it.
/// A branch that skipped the polynomial would save nothing measurable — this
/// is the slow path of the slow path — and letting a hit overwrite the result
/// keeps the whole function one straight line with no early exit, which is
/// what a kernel wants.
#[cube]
pub fn erf_accurate_tiny(
    z: f64,
    #[comptime] exceptions: bool,
    #[comptime] fk: FmaKind,
) -> (f64, f64) {
    let p = erf_p_tab();
    let z2 = z * z;

    // Degree 21, odd: the top five terms fold in `z^2` alone, as plain `f64`.
    let mut h = f64::reinterpret(p[14]);
    h = fma64(h, z2, f64::reinterpret(p[13]), fk);
    h = fma64(h, z2, f64::reinterpret(p[12]), fk);
    h = fma64(h, z2, f64::reinterpret(p[11]), fk);
    h = fma64(h, z2, f64::reinterpret(p[10]), fk);
    // `l` has to start from a runtime value, and every coefficient is finite,
    // so `h - h` is `+0.0`.
    let l = h - h;

    let (h9, l9) = dd_step_odd(h, l, z, f64::reinterpret(p[9]), fk);
    let (h8, l8) = dd_step_odd(h9, l9, z, f64::reinterpret(p[8]), fk);
    let (h6, l6) = dd_step_odd2(h8, l8, z, f64::reinterpret(p[6]), f64::reinterpret(p[7]), fk);
    let (h4, l4) = dd_step_odd2(h6, l6, z, f64::reinterpret(p[4]), f64::reinterpret(p[5]), fk);
    let (h2, l2) = dd_step_odd2(h4, l4, z, f64::reinterpret(p[2]), f64::reinterpret(p[3]), fk);
    let (h0, l0) = dd_step_odd2(h2, l2, z, f64::reinterpret(p[0]), f64::reinterpret(p[1]), fk);

    let (hf, tl) = a_mul(h0, z, fk);
    let mut rh = hf;
    let mut rl = fma64(l0, z, tl, fk);

    if comptime!(exceptions) {
        let exc = erf_exc_tiny_tab();
        for e in 0..EXC_TINY_ROWS {
            let row = usize::cast_from(e * 3u32);
            if f64::reinterpret(exc[row]) == z {
                rh = f64::reinterpret(exc[row + 1]);
                rl = f64::reinterpret(exc[row + 2]);
            }
        }
    }
    (rh, rl)
}

/// `erf(z)` as a double-double, for the inputs the rounding test could not
/// settle.
#[cube]
pub fn erf_accurate(z0: f64, #[comptime] fk: FmaKind) -> (f64, f64) {
    let mut rh = 0.0 + z0;
    let mut rl = 0.0 * z0;

    if z0 < 0.125 {
        let (h, l) = erf_accurate_tiny(z0, true, fk);
        rh = h;
        rl = l;
    } else {
        let i = u32::cast_from(8.0 * z0);
        let v = f64::cast_from(i);
        let z = (z0 - 0.0625) - 0.125 * v;
        let tab = erf_c2_tab();
        let p = usize::cast_from((i - 1u32) * 27u32);

        // Degree 18, plain `f64` from the top down to slot 19.
        let mut h = f64::reinterpret(tab[p + 26]);
        h = fma64(h, z, f64::reinterpret(tab[p + 25]), fk);
        h = fma64(h, z, f64::reinterpret(tab[p + 24]), fk);
        h = fma64(h, z, f64::reinterpret(tab[p + 23]), fk);
        h = fma64(h, z, f64::reinterpret(tab[p + 22]), fk);
        h = fma64(h, z, f64::reinterpret(tab[p + 21]), fk);
        h = fma64(h, z, f64::reinterpret(tab[p + 20]), fk);
        h = fma64(h, z, f64::reinterpret(tab[p + 19]), fk);
        let l = h - h;

        // Three steps carrying only a high coefficient...
        let (h18, l18) = dd_step(h, l, z, f64::reinterpret(tab[p + 18]), fk);
        let (h17, l17) = dd_step(h18, l18, z, f64::reinterpret(tab[p + 17]), fk);
        let (h16, l16) = dd_step(h17, l17, z, f64::reinterpret(tab[p + 16]), fk);
        // ...then eight carrying a double-double one. `two_sum` rather than
        // `fast_two_sum` throughout: for band 3 the degree-7 coefficient is
        // smaller than the degree-8 one, so `|a| >= |b|` does not hold.
        let (h14, l14) =
            dd_step2(h16, l16, z, f64::reinterpret(tab[p + 14]), f64::reinterpret(tab[p + 15]), fk);
        let (h12, l12) =
            dd_step2(h14, l14, z, f64::reinterpret(tab[p + 12]), f64::reinterpret(tab[p + 13]), fk);
        let (h10, l10) =
            dd_step2(h12, l12, z, f64::reinterpret(tab[p + 10]), f64::reinterpret(tab[p + 11]), fk);
        let (h8, l8) =
            dd_step2(h10, l10, z, f64::reinterpret(tab[p + 8]), f64::reinterpret(tab[p + 9]), fk);
        let (h6, l6) =
            dd_step2(h8, l8, z, f64::reinterpret(tab[p + 6]), f64::reinterpret(tab[p + 7]), fk);
        let (h4, l4) =
            dd_step2(h6, l6, z, f64::reinterpret(tab[p + 4]), f64::reinterpret(tab[p + 5]), fk);
        let (h2, l2) =
            dd_step2(h4, l4, z, f64::reinterpret(tab[p + 2]), f64::reinterpret(tab[p + 3]), fk);
        let (h0, l0) =
            dd_step2(h2, l2, z, f64::reinterpret(tab[p]), f64::reinterpret(tab[p + 1]), fk);
        rh = h0;
        rl = l0;
    }

    // The five arguments the polynomial itself cannot resolve, overwriting
    // whichever branch ran. See [`erf_accurate_tiny()`] on why this is a
    // trailing overwrite rather than an early exit.
    let exc = erf_exc_tab();
    for e in 0..EXC_ROWS {
        let row = usize::cast_from(e * 3u32);
        if f64::reinterpret(exc[row]) == z0 {
            rh = f64::reinterpret(exc[row + 1]);
            rl = f64::reinterpret(exc[row + 2]);
        }
    }
    (rh, rl)
}

/// `erf(x)`.
#[cube]
pub fn erf(x0: f64, #[comptime] cfg: MathConfig) -> f64 {
    let x = opaque64(x0);
    let fk = comptime!(cfg.fma());
    let z = f64::abs(x);
    let ux = u64::reinterpret(z);

    let mut out = x;
    if ux > ERF_MAX {
        let os = copysign(1.0, x);
        if is_nan64(x) {
            out = x + x; // propagate the payload
        } else if ux == 0x7ff0_0000_0000_0000u64 {
            out = os;
        } else {
            // Just short of one: the correctly-rounded value is the neighbour
            // below `1`, reached by subtracting half an ulp of it.
            out = os - HALF_ULP1 * os;
        }
    } else if z < ERF_TINY {
        // Written so that `x = -0` comes back as `-0` rather than `+0`.
        out = x;
        if x != 0.0 {
            // `erf(x) = 2/sqrt(pi) x + O(x^3)`, and the cubic term is `2^-123`
            // of the linear one here. The scaling keeps the double-double
            // correction out of the subnormal range.
            let y = CH * x;
            let sx = x * P106;
            let (h, l0) = a_mul(CH, sx, fk);
            let l1 = fma64(CL, sx, l0, fk);
            // `h - y 2^106` is exact: the two are within an ulp of each other.
            let l2 = l1 + (h - y * P106);
            out = fma64(l2, M106, y, fk);
        }
    } else {
        let (h, l, err) = erf_fast(z, fk);
        // Re-apply the sign by bit surgery: `erf` is odd, so the whole
        // double-double negates.
        let sign = u64::reinterpret(x) & 0x8000_0000_0000_0000u64;
        let uf = f64::reinterpret(u64::reinterpret(h) ^ sign);
        let vf = f64::reinterpret(u64::reinterpret(l) ^ sign);

        let left = uf + fma64(err, -uf, vf, fk);
        let right = uf + fma64(err, uf, vf, fk);
        out = left;
        if left != right {
            let (ah, al) = erf_accurate(z, fk);
            out = ah + al;
            if x < 0.0 {
                out = (-ah) + (-al);
            }
        }
    }
    out
}
