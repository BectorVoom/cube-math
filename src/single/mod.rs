//! Single-precision kernels.
//!
//! Three shapes, and the distinction is the crate's most important piece of
//! small print:
//!
//! * [`exact`] — what IEEE-754 pins down. No approximation anywhere, so no
//!   claim about a particular `libm` is involved and these run on a backend
//!   that cannot do `f64` at all.
//! * **Schedule ports** — [`exp`]'s three, [`logx`]'s two, [`trig`]'s three,
//!   [`pow`], [`atan2`] and [`bessel`]'s six. The platform does *not* compute
//!   these correctly rounded, so matching it means replaying its operation
//!   schedule. Every one of them evaluates in `f64` over a small table and
//!   rounds once — that is the algorithm rather than an implementation
//!   detail, which is why they need a device with usable double precision.
//! * [`wide`] — everything else, computed in `f64` and rounded once. That
//!   module explains why it reaches the platform's answer, and the measured
//!   rate at which it does not.

pub mod atan2;
pub mod bessel;
pub mod exact;
pub mod exp;
pub mod logx;
pub mod pow;
pub mod trig;
pub mod wide;
