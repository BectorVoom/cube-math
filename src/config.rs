//! Everything a kernel is specialised on, in one comptime value.
//!
//! [`Policy`] is the caller's choice; [`FmaKind`] is the device's. Both are
//! resolved before expansion, so a kernel contains one path and no tests on
//! configuration — and two configurations are two kernels with two
//! [`cubecl::prelude::KernelId`]s, which is what keeps them from sharing a
//! compilation cache entry.

use crate::fma::FmaKind;
use crate::policy::Policy;

/// The full compile-time configuration of a kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct MathConfig {
    /// Accuracy and domain — what the caller asked for.
    pub policy: Policy,
    /// Which multiply-add to build with — what the device can do.
    pub fma: FmaKind,
}

impl MathConfig {
    /// Bit-exact, safe on any input, assuming the device has a fused
    /// multiply-add.
    ///
    /// True on CUDA, HIP and the CPU runtime; **not** true on the SPIR-V
    /// backend. Prefer [`MathConfig::exact_for`], which takes the answer from
    /// the device instead of assuming it.
    pub const EXACT: Self = Self::new(Policy::EXACT, FmaKind::Hardware);

    /// The cheap approximation, still safe on any input.
    ///
    /// The multiply-add choice is moot here: [`crate::Accuracy::Fast`] makes
    /// no exactness claim, so it always takes the intrinsic.
    pub const FAST: Self = Self::new(Policy::FAST, FmaKind::Hardware);

    /// A configuration with a given policy and multiply-add.
    pub const fn new(policy: Policy, fma: FmaKind) -> Self {
        Self { policy, fma }
    }

    /// Bit-exact, built for what this device's multiply-add actually does.
    ///
    /// ```no_run
    /// # use cube_math::prelude::*;
    /// # use cubecl::prelude::*;
    /// # fn f<R: Runtime>(client: &ComputeClient<R>) {
    /// let fid = fidelity(client);
    /// let cfg = MathConfig::exact_for(fid.f64);
    /// # }
    /// ```
    pub const fn exact_for(precision: crate::probe::Precision) -> Self {
        Self::new(Policy::EXACT, precision.fma_kind())
    }

    /// The cheap approximation, built for this device.
    pub const fn fast_for(precision: crate::probe::Precision) -> Self {
        Self::new(Policy::FAST, precision.fma_kind())
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
