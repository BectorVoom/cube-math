//! The two axes a function is configured on — the same pair `rmath` uses,
//! carried across to CubeCL as *comptime* values.
//!
//! In `rmath` the policy is a type parameter and LLVM folds the branch away.
//! Here it is a `#[comptime]` argument: the kernel is expanded once per
//! configuration, so the branch is gone before any GPU code is generated and
//! the shader contains only the path you asked for. Two different policies
//! produce two different kernels with two different [`cubecl::prelude::KernelId`]s,
//! which is what keeps them out of each other's compilation cache.

/// How accurate the result must be.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Accuracy {
    /// Bit-for-bit identical to the platform `libm` — the reference
    /// algorithm's exact operation schedule, replayed one thread per element.
    #[default]
    BitExact,
    /// A cheaper approximation, accurate to the few ulp each kernel documents.
    Fast,
}

/// Which inputs the caller promises to supply.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Domain {
    /// Anything: infinities, NaN, subnormals, out-of-range values.
    #[default]
    FullRange,
    /// The caller guarantees every input is finite and inside the main path.
    ///
    /// Skips the range test. Faster, and **unsound to use loosely** — an
    /// out-of-range input silently produces a wrong number. What "in range"
    /// means is documented per function.
    Finite,
}

/// The pair, as one comptime value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Policy {
    /// How accurate the result must be.
    pub accuracy: Accuracy,
    /// Which inputs the caller promises to supply.
    pub domain: Domain,
}

impl Policy {
    /// `BitExact` + `FullRange` — the default, safe on any input.
    pub const EXACT: Self = Self { accuracy: Accuracy::BitExact, domain: Domain::FullRange };
    /// `Fast` + `FullRange` — approximate, still safe on any input.
    pub const FAST: Self = Self { accuracy: Accuracy::Fast, domain: Domain::FullRange };
    /// `Fast` + `Finite` — approximate, and the caller vouches for the inputs.
    pub const FAST_FINITE: Self = Self { accuracy: Accuracy::Fast, domain: Domain::Finite };
    /// `BitExact` + `Finite`.
    pub const EXACT_FINITE: Self = Self { accuracy: Accuracy::BitExact, domain: Domain::Finite };

    /// True when the reference schedule is required.
    pub const fn bit_exact(self) -> bool {
        matches!(self.accuracy, Accuracy::BitExact)
    }
    /// True when out-of-range inputs must be handled.
    pub const fn checked(self) -> bool {
        matches!(self.domain, Domain::FullRange)
    }
    /// A short stable tag, used to name the generated kernel.
    pub const fn tag(self) -> &'static str {
        match (self.accuracy, self.domain) {
            (Accuracy::BitExact, Domain::FullRange) => "exact",
            (Accuracy::BitExact, Domain::Finite) => "exact_finite",
            (Accuracy::Fast, Domain::FullRange) => "fast",
            (Accuracy::Fast, Domain::Finite) => "fast_finite",
        }
    }
}
