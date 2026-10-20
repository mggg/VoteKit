# Stage 3: numeric profile operations

## Goal

Move shared scoring and built-in cleaning onto the authoritative arrays. This gives several
election families a common computational path and lets native kernels share those contracts
without changing their Python interfaces. Existing FastSTV and pairwise kernels
consume the same profile data rather than performing their own labeled round trips.

Status: numeric scoring, built-in cleaning, validation, and kernel adapters are implemented.
Ranked/scored native dispatch is operation-specific. Scored cleaning remains NumPy by default:
its native implementation did not consistently improve the complete operation after copy and
result-construction costs. The private forced-native route remains available for comparison.

## Constraints inherited from the baseline

An integer code identifies a ranking set, so an operation must either interpret that set or
establish the narrower singleton precondition. The matrix column is a ranking position, not
necessarily a positional-score index. `score_dict_from_score_vector` advances its score-vector
position by the size of each set: a three-candidate tie consumes three positions. It rejects
explicit empty sets and skips padding according to its existing rules.

Public cleaning includes more than the resulting matrix. The baseline functions classify
original row labels, retain parents, distinguish semantic equivalence from a textual change,
and expose flags for dropping empty or zero-weight rows and retaining candidate declarations.
Those rules must survive even when no DataFrame is constructed during the operation.

Rounding, score coercion, and candidate membership follow the per-entry-point contract in the
[engineering notes](engineering-notes.md). In particular, directly multiplying raw score-array
values is not always equivalent to materializing `ScoreBallot` first: that constructor coerces
values to floats and drops zeros. Numeric scoring must reproduce the calculation's existing
boundary behavior while storage continues to preserve the original data.

## Relationship to existing acceleration

Keep the NumPy operations that already perform useful compiled work. Replace the repeated
frozenset-to-code conversion around them. Keep the Numba pairwise tally kernel during this stage;
its arithmetic and skip/stop semantics remain the reference for pairwise comparison.

The general and no-ties scoring functions can share extraction, code membership, and output
materialization without sharing a reduction that changes rounding. FastSTV and ordinary STV
likewise need separate tally policies where the baseline differs. Avoid creating a generic
scoring engine whose configuration obscures these few concrete differences.

## Design decisions and tradeoffs

### First-preference and positional scoring

Use code-to-membership and code-to-size lookups prepared from the private table. For the no-ties
first-preference function, tally the first ranking column directly when the profile satisfies
that function's current assumptions. The baseline routes through a full positional score vector;
the optimization must still reproduce its validation, ignored-code behavior, output keys, and
final rounding. Profiles outside the proven equivalent subset retain the reference operation.

For general positional scoring, traverse rows and ranking sets in baseline order. Calculate the
score-vector slice using cumulative set size, apply `high`, `low`, or `average`, and perform the
existing rounding at the same point. Numeric membership tables remove Python set traversal from the
Rust path. The numeric Python implementation remains an executable reference. Do not construct a
three-dimensional ballot-by-rank-by-candidate tensor.

Score-ballot totals use ordered score columns with the operation's existing coercion and
missing-value rules. Distinguish all-missing rows, explicit zeros that materialize as an empty dictionary, and
nonzero scores on zero-weight rows. Validate before interpreting missing entries as arithmetic
zeros. Preserve candidate output keys even when their totals are zero.

### Built-in cleaning

Candidate removal computes a replacement set for each observed code, interns equivalent result
sets, and applies the resulting lookup to the ranking array. This performs Python set work once
per distinct code in the NumPy reference. It must never replace the padding code with an explicit
empty-set code or vice versa merely because both contain no remaining candidate.

Compaction removes positions according to the specific cleaning function and pads to the required
width. Removing repeated candidates is row-dependent: a candidate seen earlier in the row changes
the meaning of a later set. Track seen candidates per row because a single global replacement table
cannot capture that dependency.

Return numeric result data and provenance together. Classify source rows using their public labels,
then apply the requested row filters. Preserve the baseline treatment of trailing empty positions as
equivalent where the combined removal/condensation functions specify it. Already empty rows may
appear in both unchanged and no-ranking categories, so those sets must allow overlap.

For scored cleaning, candidate removal changes the value columns and candidate metadata with
the same retention rules as the baseline. Preserve all-missing, all-zero, and zero-weight cases
and their distinct classifications. Reconstruct `CleanedScoreProfile` directly from validated
storage and provenance.

### Grouping and addition

Keep row multiplicity until an explicit grouping call. Group by decoded ranking meaning or by codes
from a reconciled table; never compare codes from different tables directly. Preserve group order,
weight reduction behavior, voter-set union, index reset, and the warning/type change when grouping a
cleaned profile. Reproduce the exposed code dictionaries where the baseline result constructor
assigns fresh codes rather than leaking unused entries from a shared table.

Grouping and addition must remain functional after the storage switch, but their full numeric
ports are not prerequisites for native IRV. Retain a fresh DataFrame compatibility implementation
until a numeric replacement passes those ordering and arithmetic contracts. A broad rewrite of
profile equality is similarly outside this stage.

### Callbacks

Keep `clean_rank_profile` and `clean_score_profile` compatible with arbitrary Python cleaning
functions. Preserve the callback's tuple contents, order of invocation, number of calls,
exceptions, and treatment of the returned row. That path may materialize labeled rows.

For built-in profile types, built-in operations call their numeric implementations directly. Unknown
profile subtypes retain the dynamic `.df` and `.ballots` accesses used by the baseline, including
during validation. Do not attempt to recognize equivalent user lambdas, inspect arbitrary callback
code, or run a callback during dispatch to discover its capabilities. Later native dispatch follows
the same rule.

## Implementation boundaries

Place numeric implementations beside the relevant existing operation, or in a small private
`pref_profile/_operations.py` module when multiple callers need the same function. Public
wrappers retain signatures and handle validation and labeled return values. Trusted result
construction uses stage 2's private storage constructor.

Update the scoring entry points in `utils/common_utils.py`, the built-in profile cleaning
functions, and validators in `RankingElection` and `GeneralRating`. Validators should inspect
numeric rows and construct an offending ballot only when an error message requires it. Sharing
structural facts does not eliminate election-specific checks or authorize earlier public errors.

Integrate `NumpySTVBase` through its consumer-specific adapter. Use checked candidate indices and
position indices wide enough for the accepted dimensions. Remove the `int8` conversion and 256-entry
materialization lookup assumptions together because changing only the input cast leaves history
reconstruction dependent on the old range.

Integrate `pairwise_dict` through its own adapter. Preserve its validation of ballot presence,
candidate selection, pair order, and current undefined-set handling. Do not silently begin
rejecting ties because the STV adapter requires singleton rankings. Keep its existing Numba
signature and arithmetic until a separate native checkpoint proves an equivalent replacement.

## Implementation checkpoints

1. Activate first-preference, positional, and score totals separately, with their existing
   wrappers and numeric validators. Each scorer is an independent release unit.
2. Activate ranked candidate removal/compaction and repeated-candidate removal with full cleaned
   provenance. Scored candidate removal is a separate release unit.
3. Integrate FastSTV conversion and wide-index reconstruction as one checkpoint, then integrate
   pairwise conversion without replacing the Numba tally kernel.

The relevant stage-2 profile store is a prerequisite for each activation. Pure numeric grouping can
follow if its cost warrants the extra implementation. Compatibility grouping remains available while
the election migration proceeds.

## Verification

Use differential tests that compare output keys and order, exact values where specified,
warnings and prints, errors, and every field of a cleaned result. Test tied sets larger than two
and score-vector slices crossing a tie boundary. Include internal empty sets, trailing padding,
repeated candidates within and across tied sets, empty profiles, zero weights, and declared
candidates absent from the votes.

Use callback fixtures that record invocation order and raise at a chosen row. Confirm that
built-in acceleration does not change the public callback path or invoke it speculatively.
Check custom profile subtypes with overridden accessors as well as election subclasses.
Check cleaned parent chains, category overlaps, candidate retention flags, and public index
labels that differ from physical row positions. Compare result code dictionaries as well as
decoded ballots for operations that rebuild profiles.

Exercise candidate and ranking counts on both sides of the old 8-bit range, including history
materialization. These are capacity checks where the baseline may fail; successful ordinary
inputs still require class-specific parity. Invalid private adapter payloads must fail before
indexing or allocation based on their codes.

Benchmark standalone operations and complete Borda, plurality, rating, IRV, and STV elections.
Measure pairwise construction before and after Numba warmup. Confirm that successful numeric
paths do not populate public `.df` or `.ballots` caches, while intentional error and callback
materialization remains possible. Apply the stage-1 time and memory gates.

## Exit criteria

The activated scoring and cleaning operations work directly on profile arrays and preserve
their public contracts. FastSTV and pairwise conversion no longer reconstruct labeled rankings
before recoding them. The ordinary and exact Python paths remain available, and stage 4 can use
the resulting numeric operations without depending on pandas inside its transition loop.

## Validation ownership

Built-in numerical validators use Rust scans where eligible and preserve the Python error boundary.
Construct an offending ballot only when its representation is needed in the exception. Structural
profile validity and eligibility for a singleton election remain distinct checks. Never skip invalid
metadata or constructor errors because a numeric kernel happens to accept the rankings. Unknown
profile subclasses retain dynamic accessor behavior for validation as well as scoring and cleaning.
