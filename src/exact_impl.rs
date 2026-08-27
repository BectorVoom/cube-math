//! The functions IEEE-754 pins down exactly, as one template.
//!
//! Unlike the transcendentals — whose algorithms genuinely differ between
//! precisions — these are the same code with different field widths, so they
//! are written once here and instantiated twice, by [`crate::double::exact`]
//! and [`crate::single::exact`].
//!
//! A `#[cube]` function cannot be generic over precision the way `rmath`'s
//! are: the bit reinterpretations need a concrete integer type of matching
//! width. The macro gives the same single-source guarantee by another route.

/// Generate the IEEE-exact family for one precision.
macro_rules! exact_for {
    (
        $f:ident, $u:ident,
        mant: $mant:expr,
        mant_i: $mant_i:expr,
        sign: $sign:expr,
        exp_mask: $exp_mask:expr,
        exp_field: $exp_field:expr,
        one: $one:expr,
        bias: $bias:expr,
        max_biased_exp: $max_be:expr,
        half_minus_ulp: $hmu:expr,
        two_pow_mant: $tpm:expr,
        min_positive: $minpos:expr,
        inf: $inf:path,
        opaque: $opaque:path,
    ) => {
    use cubecl::prelude::*;

    /// Zero, as a named constant.
    const ZERO: $f = 0.0;
    /// `INT_MIN`, which is what glibc returns for `ilogb(0)` and `ilogb(NaN)`.
    const ILOGB0: $f = -2147483648.0;
    /// False, as a named constant.
    const FALSE: bool = false;
    /// An integer zero, at the width of this precision's bit pattern.
    const UZERO: $u = 0;
    /// A signed zero, for the quotient counters.
    const IZERO: i32 = 0;
    /// The smallest positive value of this precision.
    const MIN_SUBNORMAL: $f = $f::from_bits($one);


    /// True when `x` is NaN.
    ///
    /// Read off the bits, not written as `x != x`. The IEEE definition is the
    /// clearer spelling, but it depends on `!=` being an *unordered* compare,
    /// and not every backend lowers it that way — CubeCL's CPU runtime emits
    /// an ordered compare, which answers `false` for a NaN and quietly turns
    /// every NaN test in the crate into a no-op. A magnitude above the
    /// infinity pattern is a NaN on every conforming device, with no
    /// comparison involved at all.
    #[cube]
    pub fn is_nan(x: $f) -> bool {
        ($u::reinterpret(x) & !$sign) > $exp_field
    }

    /// True unless `x` is an infinity or a NaN.
    #[cube]
    pub fn is_finite(x: $f) -> bool {
        ($u::reinterpret(x) >> $mant) & $exp_mask != $exp_mask
    }

    /// The magnitude of `x` with the sign of `y`.
    ///
    /// Exact for every input including zeros, infinities and NaN, which is
    /// what the C `copysign` contract requires: `abs` clears a sign bit and
    /// negation flips one, and neither touches anything else.
    ///
    /// Not the more obvious `reinterpret((bits(x) & !SIGN) | (bits(y) & SIGN))`
    /// — that reinterprets `x`, and `x` is often a constant here
    /// (`copysign(0.0, x)` appears in `modf`, `fmod` and `remquo`). The C++
    /// backends compile a reinterpretation to `reinterpret_cast<T const&>`,
    /// which needs an lvalue, and a constant is not one. Only `y` is
    /// reinterpreted below, and `y` is always a runtime value.
    #[cube]
    pub fn copysign(x: $f, y: $f) -> $f {
        let y = $opaque(y);
        let a = $f::abs(x);
        select($u::reinterpret(y) & $sign != UZERO, -a, a)
    }

    /// `2^k`, built straight into the exponent field.
    ///
    /// Exact, and needs no table. `k` must be in `-(bias - 1) ..= bias`.
    #[cube]
    pub fn two_pow(k: i32) -> $f {
        $f::reinterpret($u::cast_from(k + $bias) << $mant)
    }

    /// The stored exponent field of `|x|`. Zero for a subnormal.
    #[cube]
    pub fn biased_exp(x: $f) -> u32 {
        u32::cast_from(($u::reinterpret(x) & !$sign) >> $mant)
    }

    /// Largest integer not greater than `x`.
    #[cube]
    pub fn floor(x: $f, #[comptime] _cfg: crate::config::MathConfig) -> $f {
        let x = $opaque(x);
        $f::floor(x)
    }

    /// Smallest integer not less than `x`.
    #[cube]
    pub fn ceil(x: $f, #[comptime] _cfg: crate::config::MathConfig) -> $f {
        let x = $opaque(x);
        $f::ceil(x)
    }

    /// `x` truncated towards zero.
    #[cube]
    pub fn trunc(x: $f, #[comptime] _cfg: crate::config::MathConfig) -> $f {
        let x = $opaque(x);
        $f::trunc(x)
    }

    /// `sqrt(x)`, correctly rounded.
    ///
    /// Both policy axes are no-ops: IEEE-754 *requires* square root to be
    /// correctly rounded, and every device implements it as one instruction.
    #[cube]
    pub fn sqrt(x: $f, #[comptime] _cfg: crate::config::MathConfig) -> $f {
        let x = $opaque(x);
        $f::sqrt(x)
    }

    /// `|x|`.
    #[cube]
    pub fn abs(x: $f, #[comptime] _cfg: crate::config::MathConfig) -> $f {
        let x = $opaque(x);
        $f::abs(x)
    }

    /// `x` rounded to the nearest integer, ties away from zero.
    ///
    /// Ties *away*, matching C's `round` — not the ties-to-even that hardware
    /// gives you, and not whatever a particular backend's `round` intrinsic
    /// happens to do. Built explicitly as `trunc(x + copysign(0.5 - 1ulp, x))`:
    /// adding the largest value below a half shifts a half-integer to the next
    /// integer away from zero and leaves everything else alone, in a single
    /// rounding, and it passes signed zeros, subnormals and NaN payloads
    /// through untouched.
    #[cube]
    pub fn round(x: $f, #[comptime] _cfg: crate::config::MathConfig) -> $f {
        let x = $opaque(x);
        $f::trunc(x + copysign($hmu, x))
    }

    /// `x` rounded to the nearest integer, ties to even.
    ///
    /// C's `rint` rounds in the *current* mode; there is no way to leave
    /// round-to-nearest-even from here, so that is the mode this reproduces.
    /// Spelled with the add-and-subtract trick rather than a `round`
    /// intrinsic, because the intrinsics disagree about ties across backends
    /// and this does not: the addition itself rounds, in the one mode there is.
    #[cube]
    pub fn rint(x: $f, #[comptime] _cfg: crate::config::MathConfig) -> $f {
        let x = $opaque(x);
        let a = $f::abs(x);
        // Past `2^mant` every value is already an integer, and the trick would
        // overflow the significand. That branch also catches infinities and
        // NaN, which are their own answer.
        let r = (a + $tpm) - $tpm;
        copysign(select(a >= $tpm, a, r), x)
    }

    /// The positive difference: `x - y` if `x > y`, and `+0` otherwise.
    ///
    /// Testing the *arguments* for NaN rather than the difference is what
    /// keeps `fdim(inf, inf)` at `+0` — the difference is NaN there, but
    /// neither argument is.
    #[cube]
    pub fn fdim(x: $f, y: $f, #[comptime] _cfg: crate::config::MathConfig) -> $f {
        let x = $opaque(x);
        let y = $opaque(y);
        let quiet = select(is_nan(x) || is_nan(y), x + y, 0.0);
        select(x > y, x - y, quiet)
    }

    /// The larger of `x` and `y`, ignoring NaN.
    ///
    /// C's `fmax`, not IEEE-754-2019's `maximum`: a NaN argument is *ignored*
    /// rather than propagated, so `fmax(NaN, 1.0)` is `1.0`. Equal arguments
    /// return the first, which pins `fmax(+0, -0)` to `+0` and `fmax(-0, +0)`
    /// to `-0` — C leaves that unspecified, and this is what glibc does.
    #[cube]
    pub fn fmax(x: $f, y: $f, #[comptime] _cfg: crate::config::MathConfig) -> $f {
        let x = $opaque(x);
        let y = $opaque(y);
        select(x >= y || (!is_nan(x) && is_nan(y)), x, y)
    }

    /// The smaller of `x` and `y`, ignoring NaN. See [`fmax()`] for the NaN and
    /// signed-zero conventions.
    #[cube]
    pub fn fmin(x: $f, y: $f, #[comptime] _cfg: crate::config::MathConfig) -> $f {
        let x = $opaque(x);
        let y = $opaque(y);
        select(x <= y || (!is_nan(x) && is_nan(y)), x, y)
    }

    /// `x * 2^n`, with `n` carried in a float lane.
    ///
    /// The exponent travels as a float rather than an integer because the
    /// launch surface has one element type per buffer, and threading a
    /// parallel integer buffer through the whole crate to serve four functions
    /// would cost every other one clarity. `n` is used as if truncated towards
    /// zero, which is what the C `int` argument does.
    ///
    /// Three steps rather than one: a shift larger than the exponent range has
    /// to land on the right infinity or zero instead of wrapping the exponent
    /// field, and scaling up before down keeps a result that ends up subnormal
    /// from being rounded twice.
    #[cube]
    pub fn ldexp(x: $f, n: $f, #[comptime] _cfg: crate::config::MathConfig) -> $f {
        let x = $opaque(x);
        let n = $opaque(n);
        let mut out = x;
        // Zeros, infinities and NaN are fixed points: no exponent to shift.
        if x != 0.0 && !is_nan(x) && $f::abs(x) != $inf() {
            let limit = $f::cast_from($bias + $mant_i + 2i32);
            // A NaN exponent has no meaning — C passes an `int` and cannot
            // produce one — but this signature can, so it is pinned down
            // rather than left to the device: a NaN shift is no shift, which
            // is what `rmath` does and what falls out of C's saturating
            // float-to-int conversion. Left to `cast_from` it would be
            // whatever the hardware does with an out-of-range convert.
            let nn = select(is_nan(n), ZERO, $f::trunc(n));
            let nf = clamp(nn, -limit * 4.0, limit * 4.0);
            let mut k = i32::cast_from(nf);

            let up = $bias;
            let down = -($bias - 1);
            let mut y = x;
            if k > up {
                y = y * two_pow(up);
                k -= up;
                if k > up {
                    y = y * two_pow(up);
                    k -= up;
                    if k > up {
                        k = up;
                    }
                }
            } else if k < down {
                // Down in two hops of `down + mant + 1`: the extra significand
                // width keeps the intermediate normal, so the single rounding
                // happens on the last multiply.
                let step = two_pow(down) * ($tpm + $tpm);
                let back = -down - ($mant_i + 1i32);
                y = y * step;
                k += back;
                if k < down {
                    y = y * step;
                    k += back;
                    if k < down {
                        k = down;
                    }
                }
            }
            out = y * two_pow(k);
        }
        out
    }

    /// `x * 2^n`, under its other name.
    ///
    /// C defines `scalbn` and `ldexp` to compute the same thing wherever
    /// `FLT_RADIX` is 2, and glibc makes the second a literal alias of the
    /// first. So this is [`ldexp()`], not a second algorithm.
    #[cube]
    pub fn scalbn(x: $f, n: $f, #[comptime] cfg: crate::config::MathConfig) -> $f {
        ldexp(x, n, cfg)
    }

    /// The binary exponent of `x`, as an integer in a float lane.
    ///
    /// `floor(log2(|x|))` for a finite nonzero `x`, read off the exponent
    /// field rather than computed, so it is exact. Subnormals report their
    /// *true* exponent, not the stored one.
    ///
    /// C specifies the three degenerate cases as macros; glibc on this target
    /// makes `ilogb(0)` and `ilogb(NaN)` both `INT_MIN`, and `ilogb(inf)` is
    /// `INT_MAX`. Since the answer travels in a float lane, `INT_MIN` is exact
    /// in both precisions (it is `-2^31`) but `INT_MAX` is not representable
    /// in `f32` and arrives as `2147483648`.
    #[cube]
    pub fn ilogb(x: $f, #[comptime] _cfg: crate::config::MathConfig) -> $f {
        let x = $opaque(x);
        let a = $f::abs(x);
        let mut out = ILOGB0.runtime();
        if a == $inf() {
            out = 2147483647.0;
        } else if a != 0.0 && !is_nan(x) {
            if a < $minpos {
                // Subnormal: normalise by an exact power of two and correct.
                let n = a * $tpm;
                out = $f::cast_from(i32::cast_from($u::reinterpret(n) >> $mant) - $bias - $mant_i);
            } else {
                out = $f::cast_from(i32::cast_from($u::reinterpret(a) >> $mant) - $bias);
            }
        }
        out
    }

    /// Split `x` into a significand in `[0.5, 1)` and a power of two, so that
    /// `x == frac * 2^exp`.
    ///
    /// The exponent comes back in a float lane, for the same reason
    /// [`ldexp()`]'s argument arrives in one, and it is always an exact small
    /// integer.
    #[cube]
    pub fn frexp(x: $f, #[comptime] _cfg: crate::config::MathConfig) -> ($f, $f) {
        let x = $opaque(x);
        let mut frac = x;
        let mut e = ZERO.runtime();
        // Zero, infinity and NaN have no significand to normalise; C leaves
        // the exponent unspecified for them and every libm returns 0.
        if x != 0.0 && !is_nan(x) && $f::abs(x) != $inf() {
            let sub = $f::abs(x) < $minpos;
            // Subnormal: scale into the normal range first, then correct.
            // Exact, because multiplying by a power of two only shifts.
            let bits = select(sub, $u::reinterpret(x * $tpm), $u::reinterpret(x));
            let adj = select(sub, -$mant_i, 0i32);
            let raw = i32::cast_from((bits >> $mant) & $exp_mask);
            e = $f::cast_from(adj + raw - ($bias - 1i32));
            frac = $f::reinterpret(
                (bits & !$exp_field) | ($u::cast_from($bias - 1i32) << $mant),
            );
        }
        (frac, e)
    }

    /// Split `x` into its fractional and integral parts, both with `x`'s sign.
    ///
    /// Branch-free apart from the infinity. The two subtleties are both about
    /// signs: the fractional part takes `x`'s sign explicitly, so `modf(-0.0)`
    /// is `(-0.0, -0.0)` rather than the `+0.0` a bare subtraction gives; and
    /// an infinite `x` is selected out, because `inf - inf` is NaN where C
    /// requires `(±0, ±inf)`.
    #[cube]
    pub fn modf(x: $f, #[comptime] _cfg: crate::config::MathConfig) -> ($f, $f) {
        let x = $opaque(x);
        let int = $f::trunc(x);
        let frac = copysign(x - int, x);
        let inf = $f::abs(x) == $inf();
        (select(inf, copysign(0.0, x), frac), int)
    }

    /// The next representable value after `x` in the direction of `y`.
    #[cube]
    pub fn nextafter(x: $f, y: $f, #[comptime] _cfg: crate::config::MathConfig) -> $f {
        let x = $opaque(x);
        let y = $opaque(y);
        let mut out = x + y; // NaN in, NaN out — with a payload, and quietened
        if !is_nan(x) && !is_nan(y) {
            if x == y {
                // Including `nextafter(+0, -0) == -0`: the direction wins,
                // which is what C requires and what returning `x` gets wrong.
                out = y;
            } else if x == 0.0 {
                // Step off zero into the smallest subnormal, signed towards `y`.
                out = copysign(MIN_SUBNORMAL, y);
            } else {
                // Away from zero when `y` lies further out in the same
                // direction as `x`'s sign, towards zero otherwise. Magnitude
                // and bit pattern are monotone together, so either way it is a
                // single integer step.
                let bits = $u::reinterpret(x);
                out = select(
                    (x < y) == (x > 0.0),
                    $f::reinterpret(bits + $one),
                    $f::reinterpret(bits - $one),
                );
            }
        }
        out
    }

    /// `x` reduced modulo `y`, with the sign of `x`.
    #[cube]
    pub fn fmod(x: $f, y: $f, #[comptime] _cfg: crate::config::MathConfig) -> $f {
        let x = $opaque(x);
        let y = $opaque(y);
        let (r, _odd) = reduce(x, y);
        r
    }

    /// The IEEE-754 remainder: `x - y * n` with `n` the nearest integer to
    /// `x / y`, ties to even.
    ///
    /// Differs from [`fmod()`] only in how the quotient is rounded — nearest
    /// rather than towards zero — so the result can take either sign and
    /// satisfies `|r| <= |y| / 2`.
    #[cube]
    pub fn remainder(x: $f, y: $f, #[comptime] _cfg: crate::config::MathConfig) -> $f {
        let x = $opaque(x);
        let y = $opaque(y);
        let (r, odd) = reduce(x, y);
        let mut out = r;
        if !is_nan(r) && $f::abs(y) != $inf() {
            let ay = $f::abs(y);
            let ar = $f::abs(r);
            // `ar + ar` rather than `ay * 0.5`: halving `ay` is inexact when
            // `ay` is subnormal, and the doubling only overflows when `ar` is
            // above half of MAX, where `2 * ar > ay` holds anyway.
            let two = ar + ar;
            if two > ay || (two == ay && odd) {
                out = copysign(ar - ay, x);
            }
        }
        out
    }

    /// `(fmod(x, y), the quotient is odd)`.
    ///
    /// The shared core of [`fmod()`] and [`remainder()`], and it is exact for a
    /// reason worth stating: the divisor `d` only ever takes values
    /// `|y| * 2^j`, so doubling and halving it are exact; and a subtraction
    /// only happens when `d <= r < 2d`, where Sterbenz's lemma makes `r - d`
    /// exact too. No rounding occurs anywhere, which is why this needs no
    /// reference to agree with.
    ///
    /// The shift-and-subtract loop runs one iteration per binary digit of the
    /// quotient, which is up to the width of the exponent range. That is the
    /// same work `rmath` does and the same work glibc does; on a GPU it is
    /// also the one place in this module where threads in a warp can diverge
    /// badly, since the trip count depends on the ratio of the arguments.
    #[cube]
    pub fn reduce(x: $f, y: $f) -> ($f, bool) {
        let ax = $f::abs(x);
        let ay = $f::abs(y);
        let mut out = x;
        let mut odd = FALSE.runtime();
        if is_nan(x) || is_nan(y) || ax == $inf() || ay == 0.0 {
            // The invalid cases all produce a NaN. Spelled as a division so
            // that the device's own invalid-operation NaN comes back.
            out = (x - x) / (y - y);
        } else if ay == $inf() || ax < ay {
            out = x;
        } else if ax == ay {
            out = copysign(0.0, x);
            odd = true;
        } else {
            // Largest `|y| * 2^j` that does not exceed `ax`. `d + d` rather
            // than `d * 2` so the overflow case saturates to infinity and
            // stops the walk instead of wrapping.
            let mut d = ay;
            while d + d <= ax {
                d = d + d;
            }
            // Walk back down, one binary digit of the quotient per step. The
            // guard is `>= ay` rather than a flag because halving past `ay`
            // ends the walk on its own, including when `ay` is the smallest
            // subnormal and the halving rounds to zero.
            let mut r = ax;
            while d >= ay {
                if r >= d {
                    r = r - d;
                    odd = d == ay;
                }
                d = d * 0.5;
            }
            out = copysign(r, x);
        }
        (out, odd)
    }

    /// [`copysign()`] in the two-argument kernel shape.
    #[cube]
    pub fn copysign_fn(x: $f, y: $f, #[comptime] _cfg: crate::config::MathConfig) -> $f {
        copysign(x, y)
    }

    /// `x` reduced modulo `y`, together with the low bits of the quotient.
    ///
    /// Returns `(remainder, quotient)`, where the remainder is exactly
    /// [`remainder()`]'s and the quotient carries the sign of `x / y` and the low
    /// three bits of `|x / y|` rounded to nearest. C passes the quotient
    /// through an `int *`; here it comes back in a float lane, like
    /// [`frexp()`]'s exponent, and is always a small exact integer in `-7..=7`.
    ///
    /// A transcription of glibc's `s_remquo.c`: reduce modulo `8y` once with
    /// [`fmod()`], then subtract off `4y`, `2y` and `y` in turn, counting as it
    /// goes. Every step is exact — Sterbenz again — so this agrees with the
    /// platform by construction rather than by measurement.
    #[cube]
    pub fn remquo(x: $f, y: $f, #[comptime] cfg: crate::config::MathConfig) -> ($f, $f) {
        let x = $opaque(x);
        let y = $opaque(y);
        let sx = $u::reinterpret(x) & $sign;
        let negq = (sx ^ ($u::reinterpret(y) & $sign)) != UZERO;

        let ax = $f::abs(x);
        let ay = $f::abs(y);
        let mut rem = x;
        let mut quo = ZERO.runtime();

        if ay == 0.0 || is_nan(x) || is_nan(y) || ax == $inf() {
            // Invalid. Spelled as a division so the device's own NaN comes back.
            rem = (x * y) / (x * y);
        } else if ax == ay {
            rem = copysign(0.0, x);
            quo = select(negq, -1.0, 1.0);
        } else {
            // Reduce to `|x| < 8|y|`, but only where `8y` cannot overflow.
            let ey = biased_exp(y);
            let mut r = select(ey <= $max_be - 3u32, fmod(x, y * 8.0, cfg), x);
            r = $f::abs(r);
            let mut q = IZERO.runtime();
            if ey <= $max_be - 2u32 && r >= ay * 4.0 {
                r = r - ay * 4.0;
                q += 4;
            }
            if ey < $max_be && r >= ay + ay {
                r = r - (ay + ay);
                q += 2;
            }
            if ey == 0u32 {
                // `y` is subnormal, where halving it would be inexact; double
                // the remainder instead, which cannot overflow because
                // `r < 2|y|`.
                if r + r > ay {
                    r = r - ay;
                    q += 1;
                    if r + r >= ay {
                        r = r - ay;
                        q += 1;
                    }
                }
            } else {
                let half = ay * 0.5;
                if r > half {
                    r = r - ay;
                    q += 1;
                    if r >= half {
                        r = r - ay;
                        q += 1;
                    }
                }
            }
            rem = select(sx != UZERO, -r, r);
            // Negate as an integer, not as a float: `q` can be zero, and
            // negating a float zero gives `-0.0` where C's `int` quotient
            // gives `0`.
            quo = $f::cast_from(select(negq, -q, q));
        }
        (rem, quo)
    }
    };
}

pub(crate) use exact_for;
