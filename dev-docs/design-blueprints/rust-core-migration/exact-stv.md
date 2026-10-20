# Native exact STV

## Contract and scope

Move supported `STV(..., exact=True)` and `IRV(..., exact=True)` execution to Rust without
converting rational weights, tallies, transfer factors, or quotas through floating point.
Generators still return profiles; election methods accept those profiles independently.
Keep custom election subclasses, custom/random transfers, and unsupported profile layouts on the
existing Python exact implementation. Admission requires strictly positive input weights and a
positive quota. Custom `Fraction` subclasses keep Python arithmetic so their hooks are honored.
Exact dispatch requires `native_enabled()`; `force_python()` always selects the
original eager exact reference, never the float-only Python retained runner.

Use an internal `ExactWeight` with inline `Ratio<u128>` values and `BigRational` overflow fallback.
Reuse the rational library's checked arithmetic, including intermediate products and subtraction;
promote before overflow can affect a value. Reduced results may return to inline storage. Numeric
ordering must compare values across representations, never enum discriminants. Enable PyO3's existing
arbitrary-precision conversions for output. Validate integer numerator/denominator pairs before constructing
ratios so malformed private calls cannot trigger a zero-denominator panic. Arbitrary-size inputs
must not silently narrow or fall back to floats.

## Ownership and execution

Reuse the existing retained workspace and STV runner with two concrete numeric instantiations.
A small internal numeric trait owns zero/one, checked domain validation, score reporting, surplus
factors, and accumulation/multiplication. Float reporting keeps the existing ten-place rounding;
exact reporting preserves the rational value. Avoid a second copy of the election algorithm.

Separate row-weight storage from the round algorithm. Floating weights stay per row with ordered
reductions. Exact weights retain immutable base values and shared cumulative transfer factors.
Rows with the same factor and next candidate contribute their base-weight sum times that factor;
they remain distinct rows in the workspace and replay. Each elected candidate splits only its
affected rows, even when multiple winners shared a factor before a simultaneous transfer.

Use reference-counted immutable factors to bound persistent allocations by the original row count.
A transfer-local identity cache pins both old and new factors until its bucket is complete, so
allocation addresses cannot be reused as another group's identity. Cast ordering still comes from
live-row traversal; hash-table iteration may reorder exact additions only. When every row shares
one factor, sum base values directly and apply that factor once per candidate.

```rust
struct Workspace<N = f64> { /* rankings, weights, live rows, transitions */ }
struct RoundState<N = f64> { /* ranked groups, scores, tiebreak records */ }
enum NativeHistory {
    Empty,
    Floating(CompletedRun<f64>),
    Exact(CompletedRun<ExactWeight>),
}
// Kept behind ExactWeight so callers cannot construct invalid inline ratios.
enum Representation {
    Small(Ratio<u128>),
    Big(BigRational),
}
struct SharedExactWeights {
    base: Vec<ExactWeight>,
    factors: Vec<Arc<ExactWeight>>,
}
struct NativeRankedProfile {
    rankings: Arc<[i64]>,
    width: usize,
    weights: Vec<u64>,
    candidates: usize,
    total: u128,
}
struct CandidatePiles<N> {
    first: Vec<Option<usize>>,
    rows: Vec<Vec<usize>>,
    totals: Vec<N>,
}
```

`ElectionCore` keeps the numeric history variant and the existing optional public Python list.
Exact summaries become Python `Fraction` objects lazily. Once materialized, edits to that real
list continue to govern summary queries. Replay uses the original exact input and recorded exact
transitions without rerunning tiebreaks. Preserve candidate ordering, row multiplicity and order,
empty rankings after simultaneous transfers, elimination provenance, and terminal empty profiles.
Serialization must restore exact weights and transitions without narrowing, while accepting existing
version-one floating/Python history payloads. Exact version-two payloads encode weights and factors
as validated integer pairs, with size admission before extraction. Replay budgets distinct input
denominators times every recorded factor denominator. Factors must lie in [0, 1] and contribute
once per occurrence: repeated 1/3 transfers can produce 1/9.
Restore into a temporary owner and replace the core only after successful validation and replay.
Replay materializes each row's base value times its shared factor. Serialization continues to store
original weights and transitions, never allocation identities or grouping caches.

Keep Python label/set ordering and the existing Python RNG policies. Exact loser ties rank initial
exact tallies and randomize unresolved groups. Exact winner ties use random/lexicographic policy
or exact current first-place/Borda scores. Python callbacks may resolve policies at tie events;
the retained numerical loop and transfer arithmetic execute in Rust. Reuse the exact Python tie
helpers, including tied-frozenset iteration and seeded RNG consumption. For numeric winner policies,
construct a rational `RankProfile` from ballots at the tie event; the numeric-array constructor
does not accept object-dtype Fraction weights. Callbacks are call-scoped and must not create
retained election-reference cycles.

Cache an immutable `NativeRankedProfile` snapshot on supported built-in integer-weight profiles.
This is a generic candidate-index profile owner, independent of any generator or election method.
Rust validates its singleton rankings and positive `u64` weights, accumulates the total in checked
`u128`, and calculates the exact quota with integer division. Election workspaces share its ranking
buffer but own their changing state. Discard the cache during profile serialization and restoration.
Repeated elections reuse validation and indexing; memory admission still runs for every election.

Construct Python `Fraction` weights for the initial profile only when replay requests round zero.
Other exact inputs retain the Python validation/conversion boundary: share immutable ranking storage
where supported and retain frame conversion for unusual layouts or custom numeric types. Preserve
the converted profile's candidate order and metadata. Nonintegral floats remain invalid.

Exact workspaces cache first-choice row piles and totals. After a transition, route rows from removed
candidates and recompute only the affected candidates' totals using shared factors. Keep the existing
live-row sequence authoritative for cast ordering, simultaneous-transfer survival, and replay. Rebuild
pile row IDs in that sequence; the optimization removes repeated arithmetic, not every full-row scan.
Floating workspaces retain ordered reductions. Transition validation checks cast membership without
performing rational tallies whose totals would be discarded.

## Allocation safety

Recheck native allocation admission at election entry. Include rational input limb sizes, stored
weights, round summaries, transfer factors and replay scratch in an overflow-checked estimate.
Admit Python integer payload sizes before extracting big integers; use a list of strict integer
numerator/denominator pairs, not unchecked PyO3 rational extraction. Admit input copies first, then
calculate the execution bound from extracted pairs. Reject negative numerators and nonpositive
denominators before constructing a ratio.

For execution admission, all live denominators divide a common denominator D. Bound its initial
bit length by the product of distinct input denominators (ignore ones), and bound total weight T
by the sum of input numerators. A transfer multiplies the common denominator by at most T*D, so
iterate `b_next = 2*b + t + 2` at most `n_seats` times. Numerators are bounded by T times the new
common denominator. Charge several simultaneous rational copies per row, candidate-square summary
and factor storage, arithmetic scratch, and numeric matrix buffers using checked arithmetic.
The shared-factor backend budgets sixteen rational slots and 512 bytes of bookkeeping per original
row, in addition to ranking/cursor buffers. This covers original and replay base/input arrays,
per-row factor allocations, overlapping old/new factors while splitting, grouped base sums and
hash-table slack. Retired rows can retain factors, but retained factor count remains bounded by
original rows rather than the number of rounds. Bounds also cover cumulative factors before
cancellation with base weights.
The bookkeeping allowance also covers cached first choices, candidate-pile capacities and temporary
routing changes. Admit the immutable native profile copy separately before allocating it; integer
execution derives the same rational-growth bound directly from integer bit lengths.
Overflow or a bound above the memory budget produces `MemoryError`, never a float retry. This can
reject a workload whose reduced fractions would fit. Large native buffers use fallible reservation
where supported. Big-integer arithmetic can allocate internally;
available-memory admission is conservative rather than an OOM guarantee. The bound covers the full
run and deterministic replay, including promotion to arbitrary precision; it introduces no silent
precision limit. Inline rationals are a performance choice, not a reduced memory-admission promise.

## Verification and timing

Compare the complete public result to the Python exact reference: round scores and dict order,
winners, eliminated/remaining groups, tiebreak records and RNG state, every replayed profile,
mutable history, pickling/deepcopy, and custom-hook fallback. Cover large numerators/denominators,
near ties below float precision, sequential and simultaneous transfers, exact Borda/first-place
policies, partial rankings, both quotas, malformed native inputs and memory refusals.

Benchmark ordinary public `STV` on the same profiles with Python exact, Rust exact, and Rust float.
Use prepared exact BT only to generate ordinary profiles, record table setup separately, and time
both STV alone and generation plus STV. Include 1k/10k/100k voters, nine candidates, three seats,
plus a rational-weight case. Report warmups, repetitions, stored rows, conversions, and raw samples.

Implementation results, review findings, and timings are recorded in the
[exact STV report](../../../reports/native-exact-stv-2026-09-18.md).
Storage reuse and checked inline arithmetic are measured in the
[exact STV speed report](../../../reports/exact-stv-speed-2026-09-18.md).
Shared transfer factors are measured in the
[shared-factor report](../../../reports/exact-shared-factors-2026-09-18.md).
Integer profile snapshots and candidate piles are measured in the
[native speed report](../../../reports/native-speed-next-2026-09-18.md).
