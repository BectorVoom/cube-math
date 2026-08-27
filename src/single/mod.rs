//! Single-precision kernels.
//!
//! So far the IEEE-exact family only; the single-precision transcendentals are
//! not ported yet. See the README for why they are their own job rather than
//! the double-precision kernels with narrower lanes — the platform's `float`
//! routines are separate algorithms, and matching them means porting those.

pub mod exact;
