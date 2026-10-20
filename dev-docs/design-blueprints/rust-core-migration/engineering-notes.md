# Engineering guidelines

These notes define the contracts shared across migration stages: the meaning of integer codes, the
ownership of profile data, the treatment of numbers and randomness, and the distinction between a
public input error and an unsupported native path.

Preserving winners is insufficient if the migration changes the information needed to explain them.
A profile carries raw ranking positions, weights, candidate declarations, and provenance, and an
election adds a sequence of decisions. Compact storage must preserve those public values without
requiring their reconstruction during every computational step.

## Representation and index spaces

Use a two-dimensional, C-contiguous NumPy `ndarray` of signed 64-bit integers for the canonical
ranking codes. Rows correspond to stored ballots and columns to ranking positions. Preserve the
baseline width rules, including explicit `max_ranking_length`, even when trailing positions contain
only padding. The baseline padding code is `-1`, identifying `frozenset({"~"})`. An explicit empty
set has a different code and must remain distinguishable from padding.

The code table preserves the baseline mapping for each constructor path. Codes are local to a
profile and identify sets, not candidates. Code zero must not be assumed to identify the first
candidate: the baseline constructor starts ordinary codes at one. When adding profiles, decode
or reconcile their tables before concatenating codes. Equal code numbers from unrelated profiles
have no implied relationship.

Distinguish these values in names and types:

- ranking-set codes, which identify one cell's meaning;
- dense candidate indices, which index tallies and candidate activity;
- physical ballot row positions, which index arrays; and
- public ballot index labels, which identify provenance and need not equal row positions.

The canonical Python code table can contain frozensets. Native extraction converts the needed
membership data into candidate indices and offsets, with checked code lookup. Do not allocate a
table from an unchecked maximum code, or use negative codes as NumPy or Rust indices. A private
native remapping does not change the profile's public code dictionaries.

Scored profiles own separate ordered score columns, preserving NumPy or pandas extension dtypes.
Do not stack mixed signed/unsigned columns into a homogeneous matrix before checking that the
conversion is lossless: large integers can silently become imprecise floats. A supported numerical
operation may construct a dense matrix after its existing coercion rules permit that conversion.
Missing values retain the baseline missing-value semantics; they are not automatically replaced
with zero in storage. A calculation may treat absence as zero only after performing the
same validity checks as its Python reference. Preserve the baseline boundary distinction:
DataFrame construction can retain explicit zero scores, while `ScoreBallot` construction drops
them. Some baseline signatures and docstrings mention `include_zero_score`, but the concrete
`ScoreBallot` constructor does not implement that option; this migration does not introduce it.

## Profile ownership and public snapshots

Profile constructors take ownership of new buffers or copy caller-owned data. An input DataFrame
or array must not remain a writable alias into a frozen profile. Private trusted constructors may
share storage only when its owner is immutable and its lifetime covers every derived profile.
An election always owns the buffers it mutates.

Mark internal NumPy arrays read-only to catch accidental writes. That flag is a programming aid,
not a sufficient foreign-memory safety argument: owners can sometimes restore writeability and
existing writable aliases can survive. Public numeric exports, if later added, must return
copies or use an ownership scheme that prevents mutation of the backing store. The first Rust
implementation copies inputs into Rust-owned buffers before detached execution.

The `.df`, `.ballots`, `candidate_id_map`, and `id_candidate_map` interfaces retain their public
shapes and values, but become detached representations. Keep authoritative code tables private.
Internal consumers of built-in profiles must not read a mutable public cache. DataFrame copies must
copy mutable cell payloads independently because pandas copying an object column does not make its
sets independent. Frozen ballot tuples likewise do not justify sharing a mutable score dictionary.

For built-in profiles, this removes the ability to influence computations by mutating a derived
object. It does not promise that the same cached DataFrame repairs itself after a caller edits
it. A caller who wants a changed profile constructs one from the edited data. Copying, equality,
persistence, and election execution use authoritative state even after a public cache is edited.

Restrict direct-storage execution to known built-in profile types, including the two cleaned
variants. An unknown subclass can override `.df`, `.ballots`, or other observable accessors. Its
compatibility route must honor the same dynamic accesses as the baseline. This rule applies to
standalone operations and validation as well as election dispatch. Selecting a Python election is
insufficient if its scoring helper then bypasses the profile override. Unknown profile subclasses do
not receive the new built-in snapshot guarantee or native acceleration.

Candidate ordering and `candidates_cast` remain operation-specific baseline behavior. Ranked
ballot construction uses positive-weight first encounter, while ranked DataFrame construction
sorts the inferred candidates. Scored DataFrame inference also requires a positive score, unlike
the ballot path's retained-score rule. Preserve these distinctions and declared candidates.
Do not infer activity from a nonzero tally: an active candidate can currently have zero votes.
Internal candidate indices may remain fixed while public candidate tuples change between rounds.

## Numeric policy

Ranked ballot weights already support both floats and `Fraction`. Preserve exact values and mixed
weight sequences in a separate weight vector, using an object array when necessary. That path is
Python-owned storage; supported exact STV copies it into native arbitrary-size rationals. Numeric
storage is not authorization to approximate a rational
weight, replace it with a fixed-width numerator and denominator, or reinterpret a float as the
rational value the caller might have intended.

The `STV(exact=True)` conversion contract remains in
`elections/election_types/ranking/stv/exact_fraction_ops.py`: nonnegative integral real values
and nonnegative fractions are accepted, while nonintegral floats are rejected. Ordinary STV's
explicit conversion of fractional input weights to floats also remains where the election
currently requests it. Profile construction itself does not perform that conversion. Native exact
STV uses the same retained algorithm with checked inline rationals and `BigRational` overflow
fallback, preserving Python `Fraction`
materialization, exact quotas and ties. See [exact STV](exact-stv.md) for allocation and replay bounds.

Not every public numeric operation currently uses the same reduction. The general positional scorer
rounds each weighted allocation to ten decimal places before adding it. The no-ties scorer uses
bucket sums and rounds the resulting totals. FastSTV tallies and the existing Numba pairwise kernel
have their own arithmetic. Record these differences in stage 1 and preserve them per entry point
because consolidating them under one rounding rule would change public results.

For ordinary floats, preserve the operation order needed to reproduce public results and election
decisions. Do not enable reassociation, fused arithmetic, parallel reductions, or `fastmath` in a
new kernel by default. In particular, Rust's decimal-rounding implementation must be tested against
Python's behavior, including halfway cases. An unrelated rounding primitive may produce different
results. Existing Numba behavior remains the reference for its own entry point.

Exact values require exact comparisons in parity tests. Public float scores and weights also
require baseline equality unless a documented operation-specific tolerance is justified before
activation. Any allowed tolerance must still produce identical threshold decisions, ties, and
round structure. A tolerance chosen after discovering different winners is not a remedy.

## Randomness and dispatch

Ordinary elections keep their existing `random.Random` instance and draw order. The native runner
controls progression synchronously, using attached Python set operations for label ordering and
call-scoped Python callbacks for tiebreaks. It does not introduce a second Rust election RNG or a
suspend/resume protocol. Preserve the decision context: winner or loser, initial or current
scores, candidate sorting, backup rule, prints, and exceptions.

Bradley–Terry generation has a different contract: deterministic output within a fixed package
version for a given seed or initial caller-RNG state. Its native stream is seeded from Python,
with a defined per-bloc seed draw. Cross-version identity with the old Python sampler is not
promised. See [stage 6](stage-6-native-generators.md) for grouping and model changes; do not apply
that relaxed cross-version contract to ordinary election tiebreaks.

Python set differences and frozenset iteration determine some candidate tuples and ballot-group
orders even without random ties. Dense candidate indices alone cannot reproduce these
label-dependent orders. Apply the baseline set operations to the same label inputs, retain the
resulting live-row order, and record it for replay. That order affects later floating-point
reductions, not just presentation. Normal progression exchanges candidate-scale information;
only policy events that require a profile materialize one. Random surplus transfers stay on the
complete Python path until their sampling and row-order contracts are separately migrated.

Resolve support using the complete configuration and profile before entering native execution.
A custom transfer, overridden election method, unsupported numeric value, or
unported policy selects its compatible Python path without consuming RNG state. A public input
error is still reported at the corresponding baseline validation boundary. Native errors after
execution begins are translated and propagated; they do not trigger an implicit second run.

Native retained history access must not consume fresh random draws. For retained paths it uses recorded
choices and weights. For fallback paths preserve existing callback routing and replay behavior;
do not describe them as having the stronger retained-history guarantee until they implement it.

## Validation and failures

Separate structural invariants from election preconditions. A raw profile can retain tied,
repeated, or empty ranking positions that a particular election rejects. Constructing the new
storage must not silently clean them or impose the stricter singleton-ranking engine contract.

The baseline ballot and DataFrame constructors do not perform identical numeric validation. Record
accepted values and error timing by input path. Values outside a native kernel's finite, nonnegative
numeric domain use the Python operation where it accepts them rather than acquiring a new public
constructor error. Invalid native payloads supplied directly to the private binding must still
produce deterministic errors.

Check matrix shape, weight and metadata alignment, group membership, candidate-index bounds, offset
monotonicity, and multiplication overflow before allocation or indexing in Rust. Convert indices to
platform sizes explicitly. Native constructors establish these relationships before execution; later
algorithms should not coordinate unchecked parallel vectors by convention. Public profile generation
and election entry validate across separate owners, connected by the common profile representation.
Neither owner trusts arbitrary private-call arrays or depends on a specific producer/consumer.

Expected failures return typed results, translated by the PyO3 adapter into the established Python
exceptions. Allocation failure should produce an ordinary error where allocation is explicitly
fallible; this does not promise recovery from process-wide exhaustion in every dependency. No
accepted input may reach a deliberate panic, unchecked index, or poisoned shared workspace.

## Metadata and transition history

Preserve input row multiplicity. Grouping is an explicit public operation that sums weights,
unions voter sets, and resets indices according to its baseline behavior. An implementation may
not group rows merely to improve an election benchmark, even when the winning set is unchanged.
Prepared/default exact Bradley–Terry sampling uses native grouping in first-occurrence order.
That scoped choice does not redefine grouping for MCMC samples or arbitrary profiles.

`CleanedRankProfile` and `CleanedScoreProfile` expose parent profiles and altered-index sets. Their
index labels belong in authoritative metadata. The baseline reindexes the public DataFrame after
initialization, so copying its internal frame index without examining that override loses
provenance. Store labels independently of physical row positions from stage 2 onward.

Election transitions must retain enough information to reconstruct the actual row order,
weights, filtering, candidate lists, returned profile type, and parent relationships of their
public class. A shared initial matrix does not imply that every round presents rows in initial
order. Store row selections or permutations where the baseline transfer groups ballots.

The execution log and mutable public election history have different roles. Once requested, the
public state list controls summary queries and can be edited or replaced. Recorded transitions
continue to describe the original run and drive profile replay. See [stage 4](stage-4-election-state.md)
for ownership, aliasing, garbage collection, and serialization.

## Memory admission

Estimate native allocations where their layouts are known. Checked arithmetic and fallible large
allocations protect boundary invariants; conservative available-memory checks reduce accidental
exhaustion. Prepared generators account for all tables at preparation and new output at sampling;
elections separately account for their workspace and history. Recheck availability at each boundary
without counting already resident tables or profiles twice. These checks cannot reserve
memory or guarantee recovery from process-wide OOM. See [stage 6](stage-6-native-generators.md).

## References and implementation discipline

The [NumPy array documentation](https://numpy.org/doc/stable/reference/arrays.html) describes the
homogeneous storage model used here. Its
[writeability rules](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.setflags.html)
explain why a read-only flag alone does not establish immutable foreign memory.

[PyO3's parallelism guidance](https://pyo3.rs/main/parallelism) describes detached execution. Keep
Python labels, metadata, and callbacks outside detached regions; the numeric Rust algorithms
should also be testable without initializing Python.

Type excerpts describe important ownership boundaries, not requirements to create empty layers.
Reuse existing helpers where their contract fits. Introduce a type when it enforces an invariant
or gives an operation clear ownership, and create a module when its first implementation
needs a home. Representation work should reduce the state maintainers must keep synchronized.
