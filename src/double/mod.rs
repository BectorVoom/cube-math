//! Double-precision kernels.
//!
//! These are `#[cube]` functions, which is the point: call them from inside
//! your own kernel rather than launching a pass of your own over memory.
//!
//! ```ignore
//! use cube_math::{double as m, MathConfig};
//!
//! #[cube]
//! fn my_kernel(x: &Array<f64>, out: &mut Array<f64>, #[comptime] cfg: MathConfig) {
//!     out[ABSOLUTE_POS] = m::exp::exp(x[ABSOLUTE_POS], cfg);
//! }
//! ```
//!
//! Nothing has to be bound or uploaded to make that work: the tables are
//! constant arrays compiled into the kernel. See [`crate::tables::consts`].

pub mod cbrt;
pub mod exact;
pub mod exp;
pub mod exp10;
pub mod exp2;
pub mod expm1;
pub mod hypot;
pub mod ln;
pub mod log1p;
pub mod logx;
pub mod pow;
