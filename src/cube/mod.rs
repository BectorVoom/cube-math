//! The kernels, written once and expanded per backend.
//!
//! # One thread per element
//!
//! `rmath` writes each function against a `Simd` vector and then repairs the
//! rare lanes with a scalar reference, because on a CPU the lanes of a
//! register cannot take different branches. A GPU is SIMT: the lanes *are*
//! threads, and a branch is just a branch. So the kernels here are written in
//! the shape of `rmath`'s [`reference`](https://docs.rs/rmath) module — the
//! scalar algorithm, branches and all — and the parallelism comes from the
//! launch geometry rather than from the type.
//!
//! That is not a compromise. It is the same operation schedule, so it is
//! bit-exact for the same reason; it handles infinities, NaN and subnormals on
//! the main path instead of in a fix-up pass; and it removes the whole
//! `patch_lanes` apparatus, which is where a vector libm's complexity lives.
//!
//! # What bit-exactness rests on here
//!
//! Every IEEE-754 operation rounds identically on every conforming device, so
//! replaying the reference schedule on a GPU gives the same bits as replaying
//! it on a CPU — *provided* the backend does not rewrite the arithmetic. Three
//! things could, and the crate pins all three down:
//!
//! * **Contraction.** `a * b + c` must stay two roundings. CubeCL emits
//!   separate `Mul` and `Add` instructions and only sets the SPIR-V
//!   `AllowContract` fast-math flag when asked, so the default is safe;
//!   [`crate::probe`] checks it on the live device rather than assuming.
//! * **Fusion.** `fma(a, b, c)` must be *one* rounding. This is the one that
//!   actually varies — see [`fma`].
//! * **Subnormals.** Flushing them to zero would change `exp` near the bottom
//!   of its range. [`crate::probe`] checks this too.

// A `#[cube]` local that several branches assign has to be *initialised* from
// a runtime value first: the macro turns a literal initialiser into a comptime
// constant, and assigning a runtime value to it afterwards is a type error. So
// the kernels open with `let mut out = <some real expression>` and then
// overwrite it on every path, which the compiler reads as a dead store. It is
// not one; it is the only spelling the macro accepts.
#![allow(unused_assignments)]

pub mod bits;
pub mod double;
pub mod exact;
pub mod fma;
pub mod single;
