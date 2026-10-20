# 0003: Ship a private Rust core in the VoteKit distribution

Status: implemented; platform CI verification pending.

## Context

Array ownership removes repeated DataFrame reconstruction but Python ballot loops still dominate
scoring, cleaning, and ordinary elections. A native extension also changes installation portability
and must be built and checked as part of the distribution.

## Decision

Use Maturin for one mixed distribution, preserving `src/votekit` and exposing the private module
`votekit._votekit_core`. PyO3 and rust-numpy provide bindings and typed arrays. Rust copies inputs
while attached, owns no Python references in numeric algorithm state, and detaches numeric execution.
The later PyO3 `ElectionCore` boundary owner retains labels and public states with GC support; those
references remain outside detached numerical work.
Public wrappers select native support before execution. Native errors surface directly; a missing
extension is an installation error. A scoped private Python override supports differential tests.

Keep Cargo.lock in source distributions, test with Rust 1.88.0, and publish the dependency inventory
in `rust/DEPENDENCIES.md`. Build CPython 3.11, 3.12, and 3.13 wheels for Linux glibc x86_64/aarch64,
macOS x86_64/arm64, and Windows x86_64. Every release cell installs its wheel and runs the native
smoke script. Source distributions build outside the checkout. Existing publishing authorization
and trusted-publishing checks remain in force.

## Consequences

Supported wheel users need no compiler; source builds and development require Rust and a C toolchain.
The local verification covers Linux x86_64/Python 3.13. Other matrix cells require hosted CI before
release acceptance. No release is published by this implementation. Numba remains for the supported
pairwise kernel and existing FastSTV implementation. Exact and uncommon numeric cases stay in Python.
