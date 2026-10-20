# Stage 5: native boundaries, dispatch, and distribution

## Decision and status

Implemented: one mixed Maturin distribution exposes the private `votekit._votekit_core` extension.
Supported scoring, ranked cleaning, validation, ordinary IRV/STV, and generator operations use
Rust behind Python interfaces. Native scored cleaning exists but remains opt-in for measurement;
NumPy is the default because the complete native operation did not show a consistent benefit.

Local verification covers Linux x86_64 with CPython 3.13. The configured platform wheel matrix
still requires hosted CI before release acceptance. A working local build is not evidence that
all distribution targets pass.

## Native ownership boundary

Use PyO3 and rust-numpy for typed extraction. Numeric kernels receive checked Rust values, not
ballots, DataFrames, or arbitrary Python callbacks. A PyO3 boundary object may own Python labels
and states, as `ElectionCore` does, provided that ownership participates in Python GC and remains
outside detached numeric work. “No Python references” applies to numeric algorithm state, not
every struct in the extension.

At entry, copy private NumPy inputs into Rust-owned buffers before detached execution. Retain
those buffers for a complete election or prepared sampler lifetime. This costs bandwidth and
transient memory but establishes ownership without trusting a read-only flag or caller lifetime.
Independent elections own independent mutable workspaces; there is no shared workspace pool or
process-wide sampler cache.

Validate dimensions, dtype, strides/byte order, code membership, candidate bounds, offset and
metadata alignment, and allocation arithmetic. A wrapper can select a compatible Python path
for a valid but unsupported representation. A malformed direct private binding call must return
a deterministic error instead of indexing unchecked data or panicking.

General ranked kernels receive ranking-set codes and numeric membership information. Singleton
elections receive dense candidate indices. Neither layout changes public code dictionaries.
Weights become `f64` only under the operation's coercion and eligibility contract. Metadata stays
with the Python profile adapter; native cleaning returns row classifications and transformed
numeric data from which the adapter restores index labels and provenance.

## Execution and policy

Standalone operations cross the boundary for a complete calculation. Validation scanners return
enough context to reconstruct an offending ballot only at an error boundary. Numeric validation
must preserve constructor/schema checks and the public order of failures; bypassing those checks
because the ranking matrix looks valid is incorrect.

The [native election runner](stage-4-election-state.md) owns ordinary progression synchronously.
It uses attached Python set operations for label-dependent order and call-scoped Python helpers
for existing tiebreak policies. Numeric scans, transfers, and replay remain Rust work. There is
no public backend argument and no second suspend/resume protocol. Native history uses recorded
choices rather than invoking those helpers again.

Exact arithmetic, custom transfers, random surplus transfers, unported variants, custom election
or profile subclasses, and unsupported values retain Python execution. A Python election can
still use an independently eligible native scorer or cleaner. Direct storage access must honor
the separate dynamic-access contract of unknown profile subclasses.

Resolve eligibility before execution or RNG consumption. Do not catch a native error and rerun
in Python: validation errors, allocation failures, invalid callback responses, and implementation
defects must surface once. Missing or unloadable native code is an installation error. The private
`force_python` route selects reference numerical execution for differential tests; it does not
make the package extension-free because shared election ownership still uses `ElectionCore`.

## Activation and evidence

Compare native operations with both the pinned baseline and the retained Python/NumPy predecessor.
Include setup, extraction, native copies, Python policy calls, and public result construction in
end-to-end measurements. A faster isolated kernel does not justify a slower public default.
Do not invent size cutoffs without workload evidence that both paths satisfy the same contract.

Exercise real caller types, especially integer-weight output from generators, before claiming a
workflow benefits from Rust. Measure tie-heavy inputs and separate execution, numerical replay,
and public materialization. Preserve summation order and per-entry-point rounding; using Rust
does not authorize reassociation or an incremental reduction with different floating-point results.

The `stv-cursors` Cargo feature is a default-off experiment. It adds per-row positions and summary
caching, with the additional memory included in admission estimates. Its comparisons preserve
ordinary STV policies. It is not a public runtime backend setting or a completed default switch.

## Distribution contract

Maturin builds one extension per wheel, preserving `src/votekit`, package metadata, extras,
`py.typed`, and generator JSON data. Source distributions include Rust source and `Cargo.lock`.
The documented toolchain is Rust 1.88.0; dependencies and licenses are recorded in
`rust/DEPENDENCIES.md`. Source builds/development require Rust and the native toolchain, while
supported wheel installations require no compiler.

The release configuration covers CPython 3.11, 3.12, and 3.13 across Linux glibc x86_64/aarch64,
macOS x86_64/arm64, and Windows x86_64. Each cell must install and exercise its wheel outside the
checkout. Source distributions must also build and run outside the checkout. Other targets depend
on source-build support; PyPy, free-threaded Python, and a separate pure-Python distribution are
not promised by this architecture.

Retain existing publishing authorization and trusted-publishing controls. Updating this blueprint
or building artifacts does not authorize publishing a release. See
[ADR 0003](../../ADRs/0003-private-rust-distribution.md) for the distribution decision.

## Verification required for release

Native unit tests exercise valid/invalid construction, bounds, empty dimensions, exhaustion,
transitions, transfer arithmetic, replay, and checked allocation sizes. Numeric algorithms should
be testable without Python initialization. Binding tests additionally cover Python ownership,
callback errors/reentrancy, GC, and serialization.

Python parity tests compare full public structure and error behavior with native execution forced
on and off, including exact/fallback cases, subclass dispatch, mutable snapshots, row and candidate
order, and RNG state. Vary Python hash seeds where set order affects transitions. Test independent
elections for accidental shared mutation without parallelizing one election's reductions.

Installed-artifact smoke tests must load the extension, exercise native scoring and ordinary
IRV/STV in floating and exact modes, verify custom fallbacks, and materialize history. Check package
data and existing
optional-import smoke tests as their dependencies permit. Importing the Python package alone is
not a native smoke test. Retain raw timings and fresh-process memory measurements with build and
dependency metadata; do not describe incomplete platform checks as a completed release gate.
