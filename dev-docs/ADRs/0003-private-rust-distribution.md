# 0003: A private Rust extension in the VoteKit distribution

## Context

With authoritative profile storage in Rust, the extension is a required component of VoteKit. The
distribution must deliver the Python interface and its implementation together, including the
libraries needed for exact arithmetic.

## Decision

Use Maturin to build a single mixed distribution containing `src/votekit` and the private
extension `votekit._votekit_core`, with PyO3 bindings and rust-numpy array adaptation. The
distribution preserves public imports, package metadata, extras, typing information, and bundled
generator data.

A missing or unloadable extension is an installation error once Rust storage is activated.
Compatible Python execution remains available for unsupported operations within the mixed package;
there is no separate extension-free distribution. Private reference modes may support differential
testing.

Source distributions include Rust sources and `Cargo.lock`. Release builds record dependency
versions and satisfy the distribution obligations of GMP and its bindings. Supported wheels
require no compiler on the user's machine; source builds require the documented Rust and C
toolchains.

Target VoteKit's supported CPython versions on Linux glibc x86_64/aarch64, macOS x86_64/arm64, and
Windows x86_64, with the final release matrix and minimum toolchain confirmed during
implementation. Support for each target requires installed-wheel tests outside the checkout that
exercise construction, computation, and replay. Source builds receive equivalent verification
under the existing release authorization and trusted-publishing controls.

## Consequences

The mixed build replaces Hatchling and makes Rust packaging a prerequisite for activation.
GMP-backed arithmetic must work from installed artifacts, not only from a development environment
with system libraries available.

Numba remains a dependency while pairwise computation and FastSTV require it. The
[distribution plan](../design-blueprints/rust-core-migration/stage-5-rust-execution.md) defines
build, installation, and release verification.
