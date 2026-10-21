# Engineering guidelines

Rust operations must preserve the information needed to compute, inspect, and explain a result.
These guidelines address constraints shared across the migration: representation, ownership,
numerical behavior, extension hooks, and provenance. The
[ADRs](../../README.md#accepted-matrix-migration-decisions) record the decisions; the individual
stages specify implementation and verification.

## Representation and index spaces

Distinguish ranking-set IDs, candidate indices, physical row positions, and public labels through
their names and checked conversions. These distinctions prevent a valid numeric result from being
attached to the wrong candidate or source row, and keep internal integer width from restricting
public labels.

The representation follows [ADR 0004](../../ADRs/0004-matrix-representation-and-operations.md):
ranking IDs are matrix-local, with -2 for empty sets and -1 for trailing end sentinels. Reconcile
candidate-set tables before combining matrices. An explicit empty position remains distinct from
padding, and profile validity does not imply eligibility for an election that rejects ties or
repeats.

Normalize candidate names at the public boundary and reject collisions. Intern labels without
coercing integer identities to strings or conflating declared order with set-ID or score-column
order. Candidates appearing only on zero-weight rows may fall outside the declared universe; Rust
membership must represent them without treating them as election choices.

Float64 score conversion preserves values, missingness, explicit zeros, and positive infinity.
Reject lossy narrowing. Original pandas dtypes and NaN payload bits are not retained. The separate
weight contract remains stricter.

## Ownership and Python adapters

Construction validates and owns external input before publishing the numeric payload with its
aligned metadata. Internal transformations may then rely on invariants established by those
owners, but changed values and alignment still require validation. Trust follows the established
invariants, not merely the use of a private method.

NumPy read-only flags do not establish allocation lifetime or exclude writable aliases. Public
exports copy; Rust transformations may share immutable allocations. Any future shared-view API
must prove lifetime and immutability before its buffers are used during detached execution.

During DataFrame adaptation, normalize each distinct candidate set once and transfer the completed
buffers directly to the Rust owner. Validate candidate types before interning since equality can
otherwise let an invalid object inherit a valid identity. Keep Python labels and callbacks outside
detached loops, and snapshot mutable callback inputs before relying on their contents.

## Operations and materialization

Expose matrix operations that return what consumers need: structural masks, score bounds, totals,
decoded rows, or transformed matrices. This avoids turning formerly inexpensive `.rankings` or
`.weights` accesses into repeated full-buffer copies. Batch numerical queries, reserving array
exports for consumers that need the arrays themselves.

Apply row selection before decoding. Construct one ballot for an error, stream requested ballots
with `iter_ballots`, and materialize the full tuple only when requested. Computation and
serialization read authoritative values rather than public snapshots. Export and provenance
contracts are defined in [ADR 0006](../../ADRs/0006-profile-exports-and-provenance.md).

Before replacing a consumer, audit supported subclass overrides and callback behavior. Dispatch
must preserve those extension points, using a compatible Python implementation where necessary.
Edits to detached exports do not become authoritative through that compatibility path.

## Numeric policy

[ADR 0005](../../ADRs/0005-profile-numeric-contracts.md) specifies weight representation, the
`2**50` total cap, operation-specific reductions, and comparison tolerance. The cap bounds
rational value rather than numerator or denominator size. It also leaves small fractional
contributions vulnerable to loss under naive floating accumulation.

Use one scalar Neumaier implementation for Rust total-weight and mentions reductions, preserving
each operation's contribution order and multiplicity. This deliberately replaces `math.fsum` in
those paths; it does not change grouped-weight, score-row, or positional-scoring reductions. SIMD
and parallel summation are outside the current implementation.

Check Neumaier against exact sums of the binary input values, separating accumulation error from
the final rounding to float64. Include empty inputs, mixed magnitudes, input-order variations, and
values near total-weight caps and election thresholds. Compare with the Python reference to
identify changed decisions without treating bitwise agreement with `math.fsum` as a requirement.
The comparison tolerance remains a decision rule rather than an accuracy guarantee for summation.

Preserve scalar types, traversal order, and rounding boundaries in differential fixtures. For
example, `round(np.float64(1.5e-10), 10)` yields `2e-10` while `round(1.5e-10, 10)` yields
`1e-10`. Score-row `np.nansum` needs fixtures across widths because a sequential sum need not
match its reduction. Reassociation, parallel reductions, and `fastmath` require validation against
election decisions, including halfway cases and thresholds; ordinary numerical agreement is
insufficient.

NaN represents missing score data but is invalid as a weight. Entirely unscored rows remain
distinct from explicit-zero rows even when their totals agree. Exact STV keeps quotas, transfers,
and tallies rational throughout computation. Public rational-to-float conversion must reproduce
Python's rounding and overflow behavior.

## Randomness and dispatch

The Rust runner invokes custom Python behavior through the event interface defined in
[stage 4](stage-4-election-state.md). Release incompatible borrows before a callback, validate its
result, and retain its effects for replay. Callbacks may query completed rounds but cannot
recursively advance the count or inspect a partial transition.

Preserve election order where it affects computation. Python set iteration can influence live row
order and thus floating reductions. Retained replay consumes neither randomness nor callbacks;
legacy custom algorithms retain their documented reconstruction limitations.

Generator randomness follows [stage 6](stage-6-native-generators.md): Python seeds Rust's ChaCha8
RNG. Define stream boundaries and batching behavior as the sampler is integrated. Verify
combined/by-bloc and grouped/ungrouped relationships. This contract does not alter election
tiebreak streams.

Choose execution eligibility before mutation or RNG consumption. Valid inputs outside a Rust
path's domain use a compatible implementation where available; invalid inputs fail at their
documented boundary. Once execution begins, errors propagate without retry or sampler
substitution, which could repeat side effects or change the distribution.

## Validation and failures

Matrix construction establishes structural validity. Election adapters impose their narrower
preconditions without cleaning inputs or rejecting otherwise valid profiles globally. Ties,
repeats, empty positions, and zero weights therefore require explicit eligibility handling.

Before allocation or indexing, check dimensions, dtype and byte order, relevant strides,
membership, alignment, candidate bounds, offset monotonicity, and arithmetic overflow. Concrete
Rust owners should establish relationships among their arrays. Public and deserialized inputs must
establish these invariants anew.

Return typed failures that bindings translate into the appropriate Python exceptions. Accepted
input must not reach a deliberate panic or unchecked index. Fallible reservation reduces
allocation risk but cannot guarantee recovery from process-wide exhaustion or allocations within
every dependency. Keep an unsupported operation, detected before dispatch, distinct from an
execution failure.

## Metadata, replay, and persistence

Rust row transformations carry the correspondence needed to recover public identities: source
positions for selection, many-to-one membership for grouping, and category masks with parent
relationships for cleaning. Because labels may repeat and cleaning categories may overlap, a fresh
RangeIndex cannot reconstruct these relationships.

Grouping is explicit and preserves first appearance, the specified weight reduction, and voter
unions. Other transformations retain row multiplicity unless their contract requires grouping.

Retained transitions preserve row order, filtering, declarations, weights, and terminal behavior.
Immutable summaries and replay refer to the same rounds. Stage 4 covers callback records, history
publication, and compatibility with legacy reconstruction.

Persist authoritative values and provenance in a versioned format, excluding live handles and
derived caches. Capture baseline fixtures before changing serialization, then verify restoration
into a checked owner. Pickle compatibility does not make pickle safe for untrusted input.

## Memory admission and distribution

Estimate allocation sizes with checked arithmetic, including numeric buffers, Python metadata,
exports, result construction, and replay scratch while counting shared allocations only once. For
generators, the estimate includes all resident tables and sample output; independent elections
each require their own workspace capacity.

Available-memory checks guide refusal rather than reserve capacity. Preserve resource errors
without retrying exact work in floating arithmetic. Validate rational-growth estimates and
thresholds against the production representation instead of importing fixed allowances from an
experiment.

[Stage 5](stage-5-rust-execution.md) covers GMP distribution, supported targets, and toolchain
requirements. Installed-wheel tests and source builds run outside the checkout.

## References

[NumPy writeability documentation](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.setflags.html)
describes the limits of read-only flags.
[PyO3's parallelism guidance](https://pyo3.rs/main/parallelism) explains detached execution.
Consult the versions selected for implementation when establishing the binding's lifetime and
borrowing rules.
