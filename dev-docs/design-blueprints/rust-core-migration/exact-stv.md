# Exact STV in Rust

## Exact arithmetic contract

The Rust implementations of `STV(..., exact=True)` and `IRV(..., exact=True)` keep weights,
tallies, transfer factors, and quotas rational throughout the count. The implementation reduces
repeated work and conversion while preserving exact decisions and public Fraction results.
[ADR 0005](../../ADRs/0005-profile-numeric-contracts.md) defines representation and conversion
policy; [ADR 0002](../../ADRs/0002-retained-stv-transitions.md) defines count and replay
separation.

Preserve arbitrary-size integers and exact public Fraction values within the profile's `2**50`
total-weight cap. That cap bounds value, not denominator size or arithmetic complexity. Reject
nonintegral floats at exact conversion rather than guessing the rational the caller intended.
Supported custom behavior uses Python hooks within the Rust runner. Unsupported custom algorithms
retain compatible Python execution selected before counting and its existing reconstruction
limits. Rust errors never cause a floating or Python retry.

## Rust and Python responsibilities

Rust owns exact values, arithmetic, tallying, transfers, and recorded exact transitions. Python
validates public inputs, supplies supported custom hooks, and exposes results as `Fraction`
values. The binding converts arbitrary-size integers without narrowing; intermediate Rust
arithmetic does not round-trip through Python. Stages 2, 4, and 5 integrate these responsibilities
into storage, elections, and the distribution respectively.

## Authoritative input and execution

The exact matrix owns GMP rational weights and immutable ranking-set IDs. A count shares that
storage directly, avoiding a second Rust profile cached beside Python-owned arrays. Elections own
changing exact weights or factors, activity, row order, cursors, and transitions. General profiles
remain capable of ties and empty positions; singleton eligibility belongs to the count adapter.

Floating and exact implementations can share election mechanics while keeping their numeric
operations and reduction rules explicit. Exact addition permits regrouping without rounding, but
aggregation must still preserve row identity, cast order, voter metadata, and the information
needed for public replay.

Keep quota and score comparisons exact within the Rust runner, sorting GMP rationals without
repeatedly converting them to Python Fractions. Convert scores at public boundaries and when
custom Python hooks need them, keeping intermediate tallies in Rust. If a numerical tiebreak
requires a profile, wrap the validated exact matrix rather than reconstructing ballots or a
DataFrame.

Record actual exact transfer factors, actions, and supported callback effects. Retained replay
derives row values from original input and transitions without new policy decisions, randomness,
or callback invocation. A custom transformation may require retained values rather than a scalar
factor. Preserve simultaneous-transfer survival rules, zero/empty rows where required, cleaning
provenance, and terminal profile behavior.

## GMP implementation

Use GMP with a pinned binding and build configuration. Integration verifies supported wheel and
source builds, dependency obligations, integer conversion, and allocation handling.

The following optimizations can reduce allocation and arithmetic without changing the exact model:

- Intern repeated exact inputs before allocating large rational objects; preserve validation for
  distinct inputs and supported custom numeric behavior.
- Share transfer multipliers and retain immutable base weights where that reduces repeated work.
- Group compatible denominator contributions during tallying while retaining source-row identity.
- Use bounded dense factor/candidate lookups with checked sparse fallbacks.
- Advance affected ballots through their remaining suffix rather than repeatedly scanning
  prefixes.
- Update exact totals incrementally where the invariant and full transition semantics are proved.
- Avoid repeated Python/Rust integer conversion and redundant Fraction reduction on export.

Rational-to-float conversion, when an explicit public operation requests it, must follow the
Python contract including rounding and overflow. A backend's default conversion may differ; check
nearest-even ties, subnormals, and overflow instead of relying on library defaults.

## Allocation, persistence, and failure behavior

Validate numerator/denominator payloads before constructing GMP rationals: denominators must be
positive and weights nonnegative. Check integer sizes and byte counts before expensive extraction
or allocation where possible. A malformed private call must not reach a zero-denominator panic.

Account for input limbs, exact base values, shared factors, summaries, transition logs, lookup
storage, and replay scratch. Use checked arithmetic and fallible large-buffer reservation.
Conservative available-memory admission is not a guarantee against allocation failure inside a
big-integer dependency or elsewhere in the process.

Validate denominator-growth estimates against the factor representation and allowed transitions
before using products of input denominators or transfer bounds for admission. Such bounds can
reject work whose reduced fractions would fit, so document their conservatism and derive per-row
allowances from the production layout.

Persist canonical exact values and transitions rather than allocation identities or grouping
caches. Repeated factors contribute once per occurrence; two one-third transfers can produce a
one-ninth multiplier. Restore into a temporary checked owner and publish only after validation.
Keep source labels and Python parent relationships aligned with replayed numeric rows.

## Verification

Compare complete public results with the current Python exact reference: round scores, candidate
and dictionary order where contractual, tiebreaks, final RNG state, every replayed profile,
cleaned provenance, exceptions, and custom fallback behavior. Cover sequential/simultaneous
transfers, near ties below float precision, arbitrary-size values, partial rankings, both
supported quotas, malformed private payloads, memory refusal, and independent counts sharing one
matrix.
