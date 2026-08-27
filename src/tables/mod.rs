//! Generated data tables, one subtree per precision.
//!
//! Produced by `tools/gen_tables.py` from ARM optimized-routines' C sources,
//! rather than transcribed. The generator evaluates the same `#if` conditions
//! the C build would, so the constants are provably the ones glibc compiles
//! in — which is what lets [`crate::Accuracy::BitExact`] mean *bit*-exact.
//!
//! Regenerate with `python3 tools/gen_tables.py` in `rmath`.
//!
//! [`consts`] is the part the kernels actually touch: it turns these arrays
//! into constant arrays compiled into the kernel, so that no table has to be
//! uploaded or bound.

pub mod consts;
pub mod double;
pub mod single;
