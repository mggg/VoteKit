# Preference profiles, elections, and generators: Rust core migration

VoteKit keeps its public Python model while moving supported computation and long-lived execution
state into a private Rust core. Profiles own immutable numeric storage; elections own mutable
workspaces and recorded transitions; prepared generators own reusable sampling tables. Python
provides labels, public materialization, exact-input validation, and extension points. Supported
exact STV executes with arbitrary-size Rust rationals; other exact operations retain Python paths.

Status: the core architecture described here is implemented locally. Distribution configuration is
implemented, with the full platform CI matrix still required for release acceptance. The optional
STV cursor backend remains an experiment, disabled by default. Checked validation and prepared
profile sampling have been accelerated; further traversal experiments require measured benefit.

## Architectural map

| Document                                  | Responsibility                                                           |
| ----------------------------------------- | ------------------------------------------------------------------------ |
| [Engineering notes](engineering-notes.md) | Shared representation, ownership, numeric, RNG, and error contracts      |
| [Stage 1](stage-1-numeric-access.md)      | Behavioral baseline and performance evidence                             |
| [Stage 2](stage-2-array-storage.md)       | Authoritative profile storage and public snapshots                       |
| [Stage 3](stage-3-profile-operations.md)  | Scoring, cleaning, validation, and existing kernel integration           |
| [Stage 4](stage-4-election-state.md)      | Rust election ownership, Python execution hooks, and lazy history        |
| [Stage 5](stage-5-rust-execution.md)      | Native boundaries, dispatch, packaging, and release verification         |
| [Stage 6](stage-6-native-generators.md)   | Native generators, prepared Bradley–Terry sampling, and memory admission |

The stage numbers preserve the migration's dependency order. They are not claims that every
operation in a stage should use Rust. Activation depends on compatible behavior and measured
end-to-end benefit. In particular, scored cleaning keeps its NumPy default even though a native
implementation exists.

## Ownership and execution

| Owner                          | Authoritative state                                                                             | Python-facing behavior                                                            |
| ------------------------------ | ----------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| Built-in profile               | Ranked code arrays or separate scored columns, weights, schema, provenance                      | Detached lazy DataFrames, ballots, and code maps                                  |
| Rust `ElectionCore`            | Candidate labels, compact native rounds or Python states, optional workspace and transition log | Common election queries and a lazy, mutable `ElectionState` list                  |
| Numeric election workspace     | Owned candidate-index rankings, weights, active candidates, live row order                      | Checked execution and replay; profiles built only on request or at a policy event |
| Prepared Bradley–Terry sampler | Owned bloc parameters and exact partition tables                                                | Repeated profile samples, combined or by bloc                             |

Generators return profiles; election methods consume profiles independently of their producer.
Do not add generator-specific election methods or hand sampled buffers directly to an election
runner. The profile is the shared domain boundary, even when both operations execute in Rust.

There is one authoritative input representation per profile. Native work may copy its numeric
buffers at entry to establish independent ownership, then retain them across rounds. This avoids
repeated profile conversion without relying on NumPy's read-only flag as a memory-safety guarantee.
Scored storage preserves each column's dtype; a single homogeneous matrix can silently lose large
integer values or nullable semantics.

An ordinary ranked cell remains a local code for a candidate set, not a candidate index. General
scoring and cleaning preserve ties and explicit empty sets. The stricter singleton-election
adapter constructs a checked candidate-index matrix. No universal matrix representation replaces
these distinct meanings.

For supported ordinary IRV/STV, Rust controls progression and records numeric rounds. Its PyO3
boundary uses Python label ordering and the existing election RNG where compatibility requires
those behaviors. Other `Election` families can execute their Python `_run_election()` and
`_run_step()` hooks while using the same core for history and queries. The separate `NumpySTVBase`
hierarchy remains supported; this migration does not require merging every election family.

## Recommended core types

Keep the reported state of a round separate from the mutable ballot state needed to execute the
next transition. I recommend the following two Rust representations. The sketches follow the
implemented types, with optional acceleration fields omitted; their ownership and invariants are
more important than the exact field layout.

### Round summaries: `RoundState`

```rust
type Groups = Vec<Vec<usize>>;

struct RoundState<N = f64> {
    remaining: Groups,
    elected: Groups,
    eliminated: Groups,
    scores: Vec<(usize, N)>,
    tiebreaks: Vec<(Vec<usize>, Groups)>,
}
```

Every index refers to the election's fixed candidate-label table. `Groups` preserves ranking-group
order and ties, while the ordered score pairs preserve Python dictionary insertion order. A
recorded tiebreak contains both the tied candidates and their resolved grouping. These are public
round summaries, including the entry point's score rounding, rather than a cache of unrounded
tallies. Quota comparisons and transfer factors still use rounded scores wherever the public
election contract requires them.

Let the runner produce these records and let `ElectionCore` translate them into Python
`ElectionState` objects on demand. Instantiate these types with `f64` for ordinary STV and
`ExactWeight` for exact STV, using checked inline rationals with arbitrary-precision fallback;
exact scores become Python `Fraction` values without rounding.
Python-defined elections retain their existing Python state objects. Once a public history
list exists, summary queries observe its edits; the numeric transition log still describes the
original execution. [Stage 4](stage-4-election-state.md#state-ownership) specifies that boundary.

### Checked execution state: `Workspace`

```rust
struct Workspace<N = f64> {
    rankings: Vec<i64>,
    width: usize,
    initial_weights: Vec<N>,
    weights: Vec<N>,
    live: Vec<usize>,
    active: Vec<bool>,
    transitions: Vec<Transition<N>>,
}
```

Keep these fields private and establish their relationship in `Workspace::new`: `width` is
positive, the flat matrix has exactly `width * weights.len()` cells, and each row contains distinct
in-range candidate indices followed by optional `-1` padding. Weights are finite and nonnegative;
ordinary public STV applies its stricter eligibility rules before constructing this owner.

`live` contains physical source-row indices in reduction order, and `active` is indexed by candidate.
Those index spaces must remain distinct even though both use `usize`. Checked transitions update
weights, activity, live order, and the log together. Retain original rankings and initial weights
for replay instead of copying a ballot matrix into every `RoundState`. The
[`Transition` enum](stage-4-election-state.md#numeric-transitions-and-profile-replay) distinguishes
elimination, fractional transfer, and terminal completion.

This owner contains no Python labels or callbacks. The election adapter constructs it from a
profile before entering the native runner. Keep the Python references and lazy public list in the
[`ElectionCore` boundary owner](stage-4-election-state.md#state-ownership). No shared lock or generic
backend interface is needed for an election-local workspace.

## Compatibility decisions

The pinned behavioral baseline is commit `cdb97ef4e950b6e7701a59f3b82593b0c436f68f` (version 3.5.0).
Record dependency versions with comparisons because pandas, NumPy, and Numba affect observable
behavior. Candidate labels remain supported strings and integers, including mixtures.

Preserve per-entry-point arithmetic, candidate and row order, provenance, validation timing, and
RNG behavior unless a scoped change is documented. `STV` and `FastSTV` have distinct policies;
FastSTV is an implementation reference, not ordinary STV's correctness oracle.

The implemented deliberate changes are:

- Built-in profile materializations are detached snapshots. Editing them does not alter subsequent
  computation. Unknown profile subclasses retain their dynamic-access compatibility path.
- Retained history reconstructs actual transitions without fresh tiebreaks, including the terminal
  empty-profile correction described in [ADR 0002](../../ADRs/0002-retained-stv-transitions.md).
- Public election history remains a real mutable list. Rust storage is lazy until that list is
  requested, and subsequent queries observe its edits. The execution log remains independent.
- Bradley–Terry generation uses a native random stream with fixed-version determinism and
  documented zero-support model corrections. Prepared/default exact sampling groups rows natively
  in first-occurrence order; MCMC retains public profile grouping. Seeded profiles need not match
  older versions; see [stage 6](stage-6-native-generators.md).
- Native STV reports a Hare over-election as `ValueError` instead of the baseline's eventual
  `IndexError`; valid outcomes are unchanged. See [stage 4](stage-4-election-state.md).
- `RankProfile.from_rankings_array` and `PreparedBradleyTerry` provide explicit numeric construction
  and reusable sampling interfaces without changing existing labeled constructors.

Custom election/profile subclasses, arbitrary transfers and callbacks, random
surplus transfers, and unsupported numeric representations retain compatible Python execution.
Supported `STV(exact=True)` and exact IRV share the native runner and preserve arbitrary-size
fractions; see [the exact-STV design](exact-stv.md). An error after native execution starts is
propagated rather than triggering a second run.

## Lessons that constrain the architecture

Removing repeated profile reconstruction produced larger election gains than translating isolated
loops. Ownership must therefore span a complete operation or election. Conversely, detached
DataFrames are a compatibility mechanism, not a cheap numeric access path.

Eligibility is part of performance: generated profiles commonly carry integer weights. The native
engine accepts positive integer weights whose total is exactly representable in `f64`, while
preserving the original profile's dtype. Benchmarking only manually converted float profiles
would conceal a dispatch failure in the ordinary generation-to-election workflow.

Native execution is not automatically faster. Include conversion and result construction in
measurements, preserve NumPy paths when they win, and compare against both the original baseline
and the immediately preceding architecture. Lazy `ElectionState` construction improves ownership
and avoids unwanted objects, but measurements found its conversion cost small; it is not the main
remaining throughput opportunity.

Prepared sampling amortizes exact-model setup without a process-wide cache. Memory admission must
cover all bloc tables at preparation and new output allocations at sampling. Elections separately
budget their workspace and history; resident sampler/profile memory already reduces availability. Available-memory
checks reduce accidental exhaustion but cannot reserve memory or guarantee that a process will
never run out.

## Evidence and remaining work

The implementation reports retain measurements and detailed review findings:

- [Initial migration and design decisions](../../../reports/rust-migration.md)
- [Baseline, retained Python, and native timings](../../../reports/rust-migration-speed-2026-09-16.md)
- [Numeric validation and generator kernels](../../../reports/rust-validation-generators-2026-09-16.md)
- [Native Bradley–Terry model, randomness, and memory](../../../reports/rust-bradley-terry-2026-09-16.md)
- [Integer-weight dispatch in the BT/STV workflow](../../../reports/rust-bt-stv-2026-09-17.md)
- [Native election ownership and prepared sampling](../../../reports/native-election-core-2026-09-17.md)
- [Native exact STV and measured timings](../../../reports/native-exact-stv-2026-09-18.md)
- [Optional cursor backend experiment](../../../reports/faststv-cursor-benchmark-2026-09-17.md)
- [Validation, fused sampling, and traversal experiments](../../../reports/stv-speed-improvements-2026-09-17.md)

The cursor experiment preserves ordinary STV reductions and policies while borrowing traversal
ideas from FastSTV. It does not establish that incremental tallies, different summation orders, or
FastSTV transfer rules are compatible. Its full-pipeline gain is smaller when generation dominates.

The experimental fused generator-to-election API was removed to keep profiles as the common
boundary; see the [separation report](../../../reports/generator-election-separation-2026-09-17.md). Broader validation reuse and
parallel independent simulations remain investigations. Affected-ballot traversal, selective tally
recomputation, and scratch-buffer reuse were tested together and separately where useful; they
did not justify retention on the measured workloads. Their report preserves the experiments. This blueprint does not choose those algorithms or make them release prerequisites.
Each requires its own compatibility, ownership, memory, and end-to-end evidence.

Release acceptance still requires the supported wheel matrix and source-distribution checks in
[stage 5](stage-5-rust-execution.md). CVR parsing, plotting, animation, unrelated graph algorithms,
and arbitrary user code remain outside this migration. Pandas remains part of the public interface;
Numba remains while supported production consumers need it.
