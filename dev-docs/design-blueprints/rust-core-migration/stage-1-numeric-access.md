# Stage 1: baseline contracts and shared numeric access

## Goal

Establish the behavioral reference and a shared route from profiles to numeric data before changing
storage ownership. At the pinned baseline, FastSTV, pairwise comparison, and scoring performed
related conversions independently. This stage makes their requirements explicit and prepares
consumers that can use the existing integer encoding directly.

Deliver a tested private interface and a reproducible comparison procedure. Activate consumers that
bypass mutable public caches with the relevant stage-2 storage checkpoint so cache-mutation behavior
changes consistently across operations on that profile family.

Status: numeric access and comparison harnesses are implemented. The procedure below defines
release evidence; exploratory benchmark runs do not by themselves satisfy every release gate.

## Constraints inherited from the baseline

Use the commit and dependency manifest specified in [the master document](master.md). Test each
constructor and public election family against that reference. Profile equality alone is not a
sufficient oracle: it groups ballots and does not establish row order, index labels, metadata,
cached values, or exact scalar types.

Constructors preserve raw rankings that some elections reject. Ranking-set codes depend on the
profile's construction order, and the code table includes a distinct padding value. Candidate
inference also differs between ballot and DataFrame construction. A shared accessor must expose
those values without silently sorting, grouping, cleaning, or changing weight types.

No public API is introduced in this stage. In particular, callers do not select a matrix backend
or supply dense candidate indices. Existing labeled constructors retain their compatibility
contract. Stage 2 later adds an explicit checked numeric constructor for generators and other
numeric callers.

## Relationship to existing acceleration

At the baseline, `NumpySTVBase._convert_profile_to_numpy_arrays` copies `pf.df`, maps singleton sets to
candidate indices, narrows them to `int8`, and adds an exhaustion sentinel column. Pairwise
comparison has a different mapping, including separate skip and stop values, before a Numba
kernel. Neither conversion should become the universal profile representation.

The new access layer returns the profile's ranking-set codes and table. Consumer adapters apply
their own checked interpretation. FastSTV's sentinel column and candidate-index matrix belong
to its workspace; pairwise comparison retains its current treatment of undefined sets. This
permits code reuse without replacing either consumer's contract with the other's assumptions.

## Design decisions and tradeoffs

The private interface should expose ranked data, scored data, weights, and row metadata through
small concrete values. Put the extraction code in a private module such as
`pref_profile/_storage.py` when it is ready to implement. Avoid a backend registry or a public
abstract storage protocol.

Initial extraction may copy numeric columns out of `_df`. That copy is visible in benchmarks and
disappears or becomes a view when stage 2 changes the owner. Do not cache extracted arrays against
today's mutable dictionaries and frames because mutation would make that cache stale.

Validate an adapter's numeric prerequisites without materializing every ballot merely to check
its type. If a baseline error includes a ballot representation, construct the offending ballot
at the error boundary. Preserve validation order when more than one input is invalid.

## Proposed implementation

### Behavioral fixtures

Create focused fixtures under the existing test directories for the contracts below. Expected values
must be inspected against the baseline implementation. Save baseline-produced pickle fixtures before
changing class internals.

| Area                | Required distinctions                                                                |
| ------------------- | ------------------------------------------------------------------------------------ |
| Ranked construction | Ties, repeats, empty sets, padding, declared width, empty profiles                   |
| Codes and labels    | Local codes, mixed string/integer labels, declared and cast candidates               |
| Score construction  | Missing versus zero, negative values, empty dictionaries, column dtypes              |
| Weights             | Fractions, mixed types, zero weights, large integers, non-finite values              |
| Public structure    | Row order, index labels, voter sets, grouping, copy, addition, profile type          |
| Cleaning            | Parent chains, overlapping index categories, filtering flags, callback order         |
| Elections           | Every round, threshold, ties, exhausted weight, zero-vote candidates, transfer order |
| Persistence         | Cached and uncached profiles, cleaned parents, CSV, baseline pickle state            |

For values accepted by a profile but rejected later, record both boundaries. Do not convert
accidental acceptance into a native invariant, and do not reject it globally merely because the
first native engine needs a stricter domain.

Include custom profile subtypes whose `.df` or `.ballots` override changes the observed data.
Check standalone scorers, cleaners, and election validation against those subtypes. Capture
accepted extra DataFrame columns and nonstandard ranking-column layouts as separate schema cases.

### Consumer preparation

Implement checked singleton conversion using the ranking-set table, with the candidate order
requested by the consumer. Keep a separate pairwise conversion for skip/stop semantics. Prepare
numeric first-preference and score-total consumers behind private test entry points, comparing
their outputs and errors with the baseline functions. Stage 3 owns the general operation ports.

Inventory internal reads of `.df`, `.ballots`, `_df`, and both code dictionaries. Classify each
as numeric computation, intentional public materialization, or compatibility code. Include
cleaned profiles, exact STV, profile utilities, CVR loaders, generators, matrices, and plots.
Stage 2 uses this inventory to prevent built-in readers from consulting a mutated public cache,
while preserving the dynamic-access route for custom profile subtypes.

### Benchmark procedure

Add `benchmarks/benchmark_profiles.py` using the standard library and existing NumPy/psutil
dependencies. It must support the baseline and the working implementation through their public APIs.
The benchmark owns deterministic fixtures rather than relying on a generator whose implementation
may change during the migration.

Record the baseline commit, Python and dependency versions, platform, CPU, workload parameters,
seed, warmups, repetitions, and whether public caches have been populated. Run comparisons in
isolated environments from the same fixture files. Run at least two warmups and seven measured
repetitions, alternating target order to reduce machine drift. Preserve raw measurements along with
medians and spread so comparisons reflect variability across runs.

Measure these modes separately:

1. construction from ballots, DataFrames, and representative local CVR files;
2. first and repeated numeric access, `.df`, and `.ballots` materialization;
3. standalone scoring, cleaning, grouping, and pairwise construction;
4. a complete election, including validation and conversion, without forcing lazy public states;
5. several elections on one profile and one election on each of several fresh profiles; and
6. first, repeated, reverse-order, and all-round `get_profile()` requests after execution; and
7. first and repeated public `ElectionState` list materialization, measured separately.

Use small profiles that expose setup overhead, larger profiles with many repeated rows, and
profiles dominated by distinct rows. A starting synthetic grid is 100, 10,000, and 100,000 rows;
5, 20, and 150 candidates; and short and full rankings. Select representative combinations rather
than making every Cartesian-product case a mandatory benchmark. Include tied rankings for
scorers, exact weights for fallback measurements, and tie-heavy elections for native dispatch.
Inputs that a baseline class cannot execute are correctness or capacity cases, not speed ratios.

Measure peak resident process memory in fresh child processes, including array copies, public
caches, native allocations, and retained history. Report the idle import footprint separately.
`tracemalloc` may help diagnose Python allocations but is not the native memory gate. Warm Numba
compilation separately and also report first-use latency, accounting for JIT costs consistently
across targets.

Before implementation, freeze a representative subset as the release workloads. As an initial gate,
reject a repeatable median slowdown above 10% when the absolute increase also exceeds 1 ms, and a
peak-memory increase above 10% when it also exceeds 10 MiB. Investigate measurements whose spread
makes that judgment uncertain. These are proposed regression budgets, not measured properties of
this repository. Any adjustment must precede activation and record the affected workload and the
reason. In addition to meeting the regression budget, an intended hot path must show a repeatable
end-to-end improvement outside measurement noise.

Compare every later stage with both the fixed baseline and the preceding active checkpoint.
That exposes a Rust regression which might otherwise be hidden by earlier array improvements.

## Implementation checkpoints

1. Capture structural fixtures and baseline artifacts, including pre-migration pickles.
2. Add the benchmark program, freeze release workloads, and record initial measurements.
3. Implement private numeric extraction and differential-tested consumer adapters. Keep public
   dispatch unchanged until the associated stage-2 ownership contract is ready.

Each checkpoint can be reviewed independently. This stage creates no Rust build requirement.

## Verification

Run the focused fixture and adapter tests against the pinned baseline. Check exact types and
metadata explicitly rather than relying on `PreferenceProfile.__eq__`. Verify that extraction
does not mutate the profile or change code meanings when the candidate table contains ties.

Benchmark instrumentation must identify conversion and materialization separately. Check that
the complete-election mode includes initialization and that its history mode runs after timing
execution, so requesting every profile does not accidentally become part of the election-only
measurement. Verify the comparison program against an unchanged baseline copy before trusting
ratios from an implementation change.

## Exit criteria

The private numeric interface can represent every baseline profile fixture without loss, and
prepared consumers match their reference behavior. Baseline artifacts, workloads, and regression
budgets are reproducible. The remaining public-cache readers have an explicit migration route,
and stage 2 can change ownership without first having to discover what those consumers expect.

## Interpretation after native activation

“Retained/numeric Python” measures the new storage and retained execution architecture with
numerical work forced through Python/NumPy. “Native” measures the complete public operation with
supported Rust execution, including setup, conversion, and result construction. Neither column
is an isolated kernel timing. Current reference-mode elections still use the shared Rust history
owner, so this label does not imply an extension-free package.

Always report dispatch eligibility and input dtype. Integer-weight generator output must appear
in generation-to-STV comparisons alongside float fixtures. Measure prepared table setup, repeated
sampling, and election execution separately, then measure the full workflow. Distinguish sampled
voters from stored rows after grouping. Benchmark both the ordinary path and any opt-in experiment;
record build features so cursor results cannot be mistaken for the default implementation.

Fresh-process peak memory matters: a before/after RSS snapshot can miss temporary copies and table
construction. Isolated kernel gains, warm-cache numbers, and faster state conversion do not establish
whole-pipeline gains. Keep raw exploratory measurements and their actual repetition counts in
reports rather than presenting them as completed release-gate measurements.
