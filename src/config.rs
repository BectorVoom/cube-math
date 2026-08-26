//! Everything a kernel is specialised on, in one comptime value.
//!
//! [`Policy`] is the caller's choice; [`FmaKind`] is the device's. Both are
//! resolved before expansion, so a kernel contains one path and no tests on
//! configuration — and two configurations are two kernels with two
//! [`cubecl::prelude::KernelId`]s, which is what keeps them from sharing a
//! compilation cache entry.

use crate::cube::fma::FmaKind;
use crate::policy::Policy;

/// The full compile-time configuration of a kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Config {
    /// Accuracy and domain — what the caller asked for.
    pub policy: Policy,
    /// Which multiply-add to build with — what the device can do.
    pub fma: FmaKind,
}

impl Config {
    /// A configuration with the default policy and a given multiply-add.
    pub const fn new(policy: Policy, fma: FmaKind) -> Self {
        Self { policy, fma }
    }
    /// True when the reference schedule is required.
    pub const fn bit_exact(self) -> bool {
        self.policy.bit_exact()
    }
    /// True when out-of-range inputs must be handled.
    pub const fn checked(self) -> bool {
        self.policy.checked()
    }
    /// Which multiply-add the kernel body should use.
    ///
    /// [`crate::Accuracy::Fast`] makes no exactness claim, so it always takes
    /// the intrinsic even where that is not a fused operation — emulating a
    /// fused multiply-add to feed an approximation would be paying for
    /// precision the caller has already said they do not want.
    pub const fn fma(self) -> FmaKind {
        if self.policy.bit_exact() { self.fma } else { FmaKind::Hardware }
    }
}
