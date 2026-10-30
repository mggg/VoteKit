# 0004: Matrix-owned ballot interpretation

## Context

Scoring, validation, and ballot materialization all interpret ranking IDs, score columns, and
aligned ballot data. If each consumer implements that interpretation, a storage change can alter
semantics in several places independently. The matrix interface should own both the representation
and the operations that give it meaning.

## Decision

The common `ProfileMatrix` protocol specifies construction, row count, metadata, grouping, and
export, with ranked and scored operations defined on their concrete types. Because this protocol
describes the Python interface, it may include construction adapters without making pandas part of
the Rust trait.

Ranked cells index a dense, unique candidate-set table. ID -2 denotes the empty set, -1 denotes
end-of-ranking padding, and ordinary IDs begin at zero; table position is ID plus two. IDs are
local to a matrix, so concatenation reconciles their meanings. Only end sentinels may follow an
end sentinel. Ties, explicit empty positions, and repeated candidates remain valid profile data
even when an election rejects them.

`has_rankings` includes explicit empty positions; `has_ranked_candidates` requires an actual
candidate. These queries, together with `has_ties` and `get_ranking`, define the common
interpretation of ranking rows. Integer storage width is an implementation choice subject to
checked bounds.

Score storage in Rust uses float64, accepting other numeric dtypes only when conversion preserves
their values. Scores remain nonnegative, with canonicalized zeros, NaN for missing entries, and
support for positive infinity. Infinity is retained for compatibility with the existing constructor
and its tests; rejecting it would be a separate behavior change. Conversion preserves these meanings
without retaining NaN payload bits or original pandas dtypes. Weights follow the separate contract
in [ADR 0005](0005-profile-numeric-contracts.md).

Although stored column order may differ from declared candidate order, `get_scores`, score bounds,
and score totals interpret rows within the declared universe and order. Storage may also retain
undeclared candidates on zero-weight rows. In both cases, internal indexing preserves the identity
of string and non-boolean integer labels.

Grouping, concatenation, weight replacement, and built-in storage transformations belong to the
matrix classes. Shared ballot data owns weight validation, voter unions, and row alignment.
Grouping is explicit and preserves first-appearance order. Profiles provide `get_ballot` and
`iter_ballots`, alongside the cached tuple for callers requesting all ballots.

## Consequences

Consumers obtain structural facts from matrix operations and retain responsibility for domain
policy and error reporting. Rust replacement can therefore proceed operation by operation without
requiring each consumer to decode exported arrays.

Float64 storage replaces the direct Python constructor's retention of other floating dtypes.
Exports preserve ballot meaning, not historical dtype or numeric ID assignments. The
[operations plan](../design-blueprints/rust-core-migration/stage-3-profile-operations.md)
specifies consumer migration and compatibility checks.
