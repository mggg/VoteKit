# Stage 2: authoritative array-backed profiles

## Goal

Make the existing numeric ranking representation authoritative outside pandas. Ranked profiles own a
ballots-by-ranking-positions integer array, while scored profiles own separate ordered score columns
that retain their individual dtypes. The public classes continue to accept ballots and DataFrames
and produce the same labeled representations, with the snapshot semantics defined in the engineering
notes.

Ranked and scored storage are separate activation checkpoints. Each must include its constructor,
materialization, copy, persistence, cleaned-profile metadata, and affected internal readers. A
partially migrated profile with two writable sources of truth is not eligible for release.

Status: ranked and scored storage, detached materialization, and persistence are implemented.
The numeric constructor below was added with native generator integration.

## Constraints inherited from the baseline

`PreferenceProfile.__new__` dispatches to ranked or scored classes based on the supplied ballots
or DataFrame columns. Preserve that factory behavior, including empty construction and rejection
of mixed ballot families. Neither a new public storage argument nor a new profile class is
needed.

Ranked cells preserve candidate sets exactly, including explicit empty sets and repeated
candidates in separate positions. Preserve width checks based on `max_ranking_length` and
`max_candidates_ranked`; do not confuse the number of ranking positions with the number of
candidates contained in those positions. Input validation must retain its baseline ordering.

Scored DataFrames may preserve zero entries which `ScoreBallot` materialization subsequently drops.
Candidate inference, column order, weight types, and public DataFrame dtypes can depend on the input
path. The storage constructor must carry enough information to reconstruct those values. Converting
every numeric column to `float64` would lose those distinctions.

`CleanedRankProfile` and `CleanedScoreProfile` override the displayed row indices after ordinary
profile initialization. Those public indices, their parent links, and their index categories
are part of the input to this migration, not optional reporting information.

## Relationship to existing acceleration

Use the stage-1 numeric interface as the access boundary. Its ranked codes now refer directly to
owned arrays instead of extracted DataFrame columns. Existing election adapters can share these
immutable codes and construct their own candidate-index workspace when needed.

The baseline FastSTV padded `int8` matrix does not become profile storage. Its extra sentinel column
and exhaustion conventions are engine details. Stage-3 integration replaced its unchecked narrow
casts with wider private candidate and position indices.

## Design decisions and tradeoffs

### Storage and constructors

A small ranked storage value owns the code array, code meanings, weights, and row metadata. A
corresponding scored value owns score columns, weights, and metadata. Their constructors enforce
alignment so callers cannot independently replace a weight vector or code table. The profile keeps
its public candidate declarations and existing derived properties. Unknown profile subtypes retain
the dynamic-access compatibility route specified in the engineering notes. Direct numeric consumers
recognize the built-in ranked, scored, and cleaned types explicitly.

The minimum ranked layout is conceptually:

```text
ranking_codes[B, L] : int64
ranking_sets       : code -> frozenset of public candidate labels
weights[B]         : original numeric values, including Fraction where present
ballot_index[B]    : public index labels and index metadata
voter_sets[B]      : privately owned immutable membership
```

`B` counts stored rows, not total ballot weight or distinct rankings. `L` follows the baseline
width contract. Empty profiles need explicit dimensions and public column metadata; an empty
array alone does not establish whether construction should yield a ranked, scored, or base
profile.

Ranked codes use the signed 64-bit representation in the engineering notes. Weight and score storage
preserves accepted values in separate columns, including extension arrays and object arrays when
needed. A homogeneous matrix is only an operation-local conversion after eligibility and coercion
checks; mixed signed/unsigned integers can otherwise lose precision. Preserve per-column dtype
metadata needed for public DataFrame reconstruction, including accepted extension dtypes. Converting
a large integer or extended-precision input through a float and casting it back is not lossless
preservation.

Preserve accepted extra DataFrame columns and their order as auxiliary column data. The ranked
DataFrame constructor can retain columns beyond the declared ranking width, so silently dropping
them would change `.df` and some consumers. Keep nonstandard schemas on the compatible operation
path until their numeric interpretation is established. Auxiliary columns do not enter the ordinary
election matrix and do not require another authoritative DataFrame.

Although this compatibility path is slower than ordinary `float64` storage, it preserves the public
DataFrame constructor's accepted inputs while keeping the common case compact. Later checkpoints may
accelerate additional numeric cases.

### Checked numeric construction

Generators and numeric callers can avoid labeled DataFrames and ballot objects through:

```python
@classmethod
def from_rankings_array(
    cls, rankings: np.ndarray, weights: np.ndarray, *,
    candidates: Sequence[Candidate],
) -> RankProfile: ...
```

This built-in `RankProfile` constructor accepts two-dimensional `int64` candidate indices and
aligned one-dimensional numeric weights. It checks bounds, trailing `-1` padding, no repeated
candidate within a row, canonical labels, and finite nonnegative weights. It rejects subclass
construction. These stricter inputs do not change the general labeled constructors' ability to
represent ties, repeats, or explicit empty sets.

The constructor owns copies, derives profile-local singleton codes, preserves weight dtype and
safe totals, and reconstructs ordinary public metadata. Dense candidate indices are not exposed
as public ranking-set codes. Generated grouped counts therefore stay integers until an eligible
election explicitly converts its workspace weights.

### Materialization and remaining Python consumers

Implement `.df` directly from the storage, translating each ranking code through its table.
Implement `.ballots` from rows and metadata without calling `.df` first. The two properties remain
lazy and may be cached independently. Preserve public column order, index name, scalar types,
padding labels, and ballot construction behavior.

Keep the authoritative mappings private and expose the existing dictionary attributes as
detached snapshots. Internal code must use the private table. Calling `copy()`, grouping,
addition, equality, and serialization after a caller edits a public cache must still operate on
the authoritative profile.

Some consumers will remain DataFrame-based until stage 3 or later. Give those consumers a private
fresh materialization route that reads the storage and returns a detached DataFrame. Move all
internal readers identified in stage 1 either to numeric access or to that route before activating
the profile family. For built-in profiles they must not consult an edited `.df` or `.ballots`
cache. For unknown subtypes, preserve the baseline overridden accessors instead of forcing fresh
materialization from private storage. Compatibility materialization is an explicit cost, not a
second authoritative store.

Retain `_df` temporarily as a derived encoded-frame adapter for legacy internal readers, with
detached metadata and no write-through behavior. Do not cache another complete encoded frame unless
a measured consumer requires it. Remove production reads and then the adapter as the later
checkpoints replace them. Deserializing old `_df` state remains supported independently.

### Provenance and profile operations

Move public ballot index labels into the storage rather than reproducing the cleaned-profile
override on a cached DataFrame. Update cleaned constructors and category validation to use that
metadata. The category sets must remain subsets covering the parent labels as the baseline requires,
without assuming they are disjoint.

Copy may share immutable ranking and score buffers, but the returned profile must have independent
mutable public snapshots. Addition reconciles profile-local code tables and reproduces the
baseline candidate order and width rules. Grouping remains explicit and returns the baseline
profile type, index reset, summed weights, and voter-set unions. Until their numeric ports are
ready, these operations can use the fresh compatibility materialization route.

## Implementation boundaries

### Module ownership

Put concrete storage values and checked construction in `pref_profile/_storage.py`. Keep public
factory dispatch, CSV methods, and profile properties in `pref_profile.py`, and update
`cleaned_pref_profile.py` to obtain provenance from storage. A private constructor such as
`_from_storage` can wrap a validated result without reconstructing ballots or rerunning public
parsing. It must still establish the subclass metadata required by a cleaned profile.

Do not eagerly precompute every possible election cache. Retain cheap construction facts needed
by several consumers, such as dimensions and observed code sizes. Derive specialized candidate
indices when the corresponding consumer asks for them, with cache ownership tied to immutable
storage rather than a mutable public dictionary.

### Persistence transition

Introduce explicit versioned pickle state for profiles using Python and NumPy values. Persist
authoritative data and subclass metadata, not public caches, native handles, or an election
workspace. The new release reads the pinned baseline format and the new format; old releases
are not required to read the new format. CSV remains the existing interchange representation.

For baseline pickles without a format version, recover rankings or scores and weights from `_df` and
the stored code tables. For cleaned subclasses, recover public index labels from `df_index_column`
and their parent/category attributes rather than the old internal frame index. To preserve snapshot
semantics, ignore ordinary cached `.df` and `.ballots` values when they disagree with authoritative
data. Restore objects through a private checked path before freezing attributes and discarding stale
cached totals.

Support baseline payloads with no caches, either cache populated, both populated, and cleaned parent
chains. Structurally inconsistent payloads, such as a ranking code absent from its table or a
mismatched provenance length, receive a clear deserialization error. Do not attempt to guess a
repaired ranking from conflicting caches. The baseline tables themselves are the only available
record of code meanings, so values lost to prior in-place corruption cannot be recovered.

Pickle remains a trusted Python serialization mechanism. These structural checks do not turn it
into a safe format for untrusted data. The migration does not change the URL-loading interface.

## Implementation checkpoints

1. Implement owned storage, private construction, and round-trip tests without switching public
   construction.
2. Activate ranked storage together with ranked materialization, exact weights, cleaned indices,
   old-pickle loading, and the complete inventory of ranked internal readers.
3. Activate scored storage with the corresponding numeric, missing-value, dtype, metadata, and
   reader guarantees.

The ranked and scored activation order may be reversed if implementation dependencies favor it.
Both storage checkpoints depend on stage 1. Their representation does not require Rust, although
the current mixed distribution includes the native extension.

## Verification

Run profile, ballot, cleaning, CSV, pickle, mixed-label, and exact-STV tests. Compare DataFrames
structurally, including dtypes and index metadata, and compare ballot sequences without first
grouping them. Check empty profiles and explicit unused candidates as separate cases.

Test edits to the original constructor input and to each returned public representation. Verify
that a numeric tally, `copy()`, serialization, and a second election remain unchanged. Include
mutable sets inside DataFrame cells and mutable score dictionaries inside ballot snapshots.
Verify that requesting `.ballots` does not populate `.df`, and that numeric consumers need neither.

Load the frozen stage-1 pickle fixtures and assert metadata, values, types, parents, and row
indices directly. Round-trip the new format with and without caches populated. Exercise the
existing generators, loaders, plots, metrics, and profile utilities through their public APIs,
including the remaining compatibility consumers.

Measure construction, memory before and after each public cache, repeated elections on one profile,
and old-pickle load time. Include the temporary compatibility adapters in end-to-end measurements.
Apply the stage-1 regression gates to total process memory, including retained snapshots, so a
smaller array cannot conceal an increase in overall memory use.

## Exit criteria

Each activated profile family has one authoritative store, preserves the baseline representation
and public structure, and implements the documented snapshot change consistently. Existing
pickle state remains readable. Internal consumers of built-in profiles no longer rely on a
writable public cache, and ordinary numeric access constructs no labeled DataFrame. Unknown
profile subtypes retain their documented dynamic-access compatibility route.
