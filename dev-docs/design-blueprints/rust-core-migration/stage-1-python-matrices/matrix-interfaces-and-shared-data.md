# Matrix interfaces and shared data

[Stage 1 overview](README.md) ·
[Shared completion checks](README.md#compatibility-and-completion-checks)

## Dependencies

- Existing Python/NumPy matrices, shared weight storage, and profile metadata. No Rust
  implementation or completed consumer migration is required.
- A concrete caller for each proposed operation, so its return values, ordering, and ownership
  contract answer an actual use case.

## Used by

- All [consumer categories](README.md#implementation-guides), which adopt individual capabilities
  as they become available.

## Checklist

Sources: [profile_matrix/](../../../../src/votekit/pref_profile/profile_matrix/). Tests:
[test_profile_matrix.py](../../../../tests/pref_profile/test_profile_matrix.py),
[test_matrix_row_queries.py](../../../../tests/pref_profile/test_matrix_row_queries.py), and
[test_profile_weights.py](../../../../tests/pref_profile/test_profile_weights.py).

- [ ] **Callers:** Reuse `RankMatrix.has_rankings()`, `has_ranked_candidates()`, `has_ties()`, and
      `get_ranking(row)` where their contracts match. These methods already exist in
      `rank_matrix.py`; this is caller wiring, not a request to implement them again. The consumer
      guides distinguish direct substitutions from operations that still need a matrix
      implementation.
- [ ] **Matrix + callers:** Develop a per-row length query for `ballot_lengths()` and separate
      candidate coverage where a caller needs it. A `lengths` property is a possible interface,
      not an existing method; define what it counts before replacing caller logic. Follow the
      [ranked length and coverage layout](#ranked-length-and-coverage-implementation-layout).
- [ ] **Callers:** Reuse `ScoreMatrix.get_scores()`, `score_bounds_by_ballot()`, and
      `sum_scores_by_ballot()`. These already exist in `score_matrix.py`; `GeneralRating` already
      uses the bounds and totals. Preserve declared candidate order, column mapping, missing
      scores, explicit zeros, and zero-weight undeclared columns when migrating other consumers.
- [ ] **Matrix:** Add only the missing total-weight, weight-kind, row-weight, and row-label access
      needed by identified callers. Own these operations in the concrete matrices or
      `shared_data.py` as appropriate. Follow the
      [implementation layout](#small-queries-and-bulk-access-implementation-layout) below for
      ownership, caller wiring, and bulk access.
- [ ] **Matrix + callers:** Complete missing selection and stable-order operations inside the
      matrix modules. Reuse existing grouping, concatenation, selected-row export, and ranked
      weight replacement; move consumers such as `_ordered_profile_rows()` off raw storage access.
      Keep source positions distinct from public index labels.
- [ ] **Matrix + callers:** Have matrix transformations retain or return source-row
      correspondence. Let the Python facade or cleaning wrapper use it to reconstruct voters,
      labels, and provenance without exporting and matching complete profiles. Specify both sides
      together.
- [ ] **Matrix:** Extend `protocol.py` only for operations with a common ranked/scored contract
      that shared callers need. Keep ranked membership in `rank_matrix.py` and score-column
      interpretation in `score_matrix.py`.
- [ ] **Matrix + callers:** Establish consistent column ordering across construction,
      transformations, and exports. Centralize schema ordering and make any change to callback
      tuple order explicit; follow the column-order layout below.

## Ranked length and coverage: implementation layout

A length query could serve more callers than separate `is_complete` and `is_short` predicates, but
ranking positions and candidate coverage are different quantities. `ballot_lengths()` counts
positions before the end sentinel, including explicit empty positions. A tie occupies one such
position. Candidate coverage instead counts distinct members of the relevant candidate universe;
repeated appearances do not add coverage, and undeclared candidates must not make a ballot appear
complete.

For example, with declared candidates A, B, and C, the row `({A, B}, {}, {A}, ~)` has three
ranking positions but covers only two candidates. Summing candidate-set sizes also yields three,
so neither that sum nor position count establishes completeness. Equality with the size of the
universe is meaningful only for the distinct-candidate coverage count.

| Owner                                  | Work                                                                                                                                                                                                                                                                    |
| -------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `rank_matrix.py`                       | Provide position counts for the existing `ballot_lengths()` consumer. Explore a separate coverage query when a caller needs it, with an explicit universe and treatment of repeated candidates. These are concrete ranked operations, not common protocol requirements. |
| `common_utils.py`: `ballot_lengths()`  | Aggregate row weights by position count while preserving its current rejection of rows with no ranking positions.                                                                                                                                                       |
| Completeness or short-ballot consumers | Choose the query that matches their domain. `BallotGraph.from_profile()` currently operates on decoded singleton rankings and completes length `n-1` ballots; retain its tie checks and completion rules rather than substituting a coverage predicate blindly.         |

The interface name can be settled alongside its first caller. If `lengths` is retained, document
its unit as ranking positions and keep candidate coverage distinguishable at the call site.

## Column order: implementation layout

Column order should follow one documented rule rather than depend on the construction route.
Ranked input already sorts ranking columns by their numeric suffix. Score input preserves stored
column order, ballot decoding follows declared candidate order, and concatenation aligns columns
to the result's candidate universe. These orders can currently differ.

Use increasing rank positions for ranked payloads. For scored payloads, the proposed canonical
order is declared candidate order followed by any retained undeclared columns in their existing
relative order. Keep absent scores missing when aligning columns; do not fill them with zero or
drop data from zero-weight rows. Both DataFrame exports should continue placing `Voter Set` and
`Weight` after the payload, with `Ballot Index` as the index.

| Owner                                  | Work                                                                                                                                                                                                                              |
| -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `_df_schema.py`                        | Own shared export-column order and ranking-label ordering conventions alongside the existing names. Keep score-array alignment in `ScoreMatrix`.                                                                                  |
| `rank_matrix.py` and `score_matrix.py` | Establish the payload order at construction and preserve it through grouping, selection, cleaning, and concatenation. Reorder labels and values together; candidate-set IDs are unrelated to column positions.                    |
| Profile adapters and export callers    | Reuse the matrix/schema order instead of independently sorting columns. Full and selected-row DataFrame exports should agree. Keep the separate CSV file schema explicit.                                                         |
| Score-cleaning callbacks               | Account for their existing stored-column tuple order before canonicalizing storage. Preserve that order through an adapter or document an intentional callback-contract change; do not silently reinterpret positional callbacks. |

Check differently ordered inputs, numeric ranking suffixes such as `Ranking_2` and `Ranking_10`,
mixed string/integer candidate names, missing declared columns, and retained undeclared columns.
Assert both labels and corresponding values after each transformation. Column normalization must
not change the declared-candidate accumulation order used by score totals.

## Small queries and bulk access: implementation layout

Give callers an operation for the result they need rather than requiring them to retrieve a
storage array. The following is a suggested Python layout for that boundary. The method names can
change, but numerical ownership, metadata ownership, and the distinction between scalar and bulk
access should remain explicit.

**Inside the matrix package**

| File                                                                                                                                                                            | Implementation responsibility                                                                                                                                                                                                                                                                                    |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [shared_data.py](../../../../src/votekit/pref_profile/profile_matrix/shared_data.py)                                                                                            | Implement the shared weight queries once on `ProfileVoterWeightData`: `total_weight()`, `get_weight(row)`, and access to the existing `weight_kind`. Resolve row labels here while this component owns them. Keep the arithmetic in one implementation rather than duplicating it in ranked and scored matrices. |
| [rank_matrix.py](../../../../src/votekit/pref_profile/profile_matrix/rank_matrix.py) and [score_matrix.py](../../../../src/votekit/pref_profile/profile_matrix/score_matrix.py) | Expose those queries through the concrete matrix interface, delegating to shared data in Python. Keep storage fields out of the callers. After the port, numerical queries delegate to the Rust owner while label lookup remains in the Python metadata component.                                               |
| [protocol.py](../../../../src/votekit/pref_profile/profile_matrix/protocol.py)                                                                                                  | Declare the shared query signatures used through `ProfileMatrix`, with the same return and indexing contracts for both concrete classes. Do not add the numerical implementation or ranked/scored interpretation to the protocol.                                                                                |

The intended caller-facing shape is:

```python
matrix.total_weight()       # float or Fraction under the total-weight contract
matrix.weight_kind          # Existing WeightKind enum; no array dtype inspection
matrix.get_weight(row)      # One Python float or Fraction
matrix.get_row_label(row)   # One Python label; may be implicit or duplicated
```

These are new interface suggestions, not methods already available on both matrices.
`ProfileVoterWeightData` already stores `weight_kind` and exposes `has_fraction_weights`. Define
the scalar accessors using positional row indices, including the existing negative-index
convention and `IndexError` for out-of-range rows. When labels are implicit, normalize a valid
negative position before returning its label: for a five-row source, position `-1` has label `4`,
not `-1`. Retain explicit labels without assuming they are integers or unique. A selected result
must preserve source labels according to its selection contract even if its new positions differ.

Total-weight validation and the public total should ultimately use the same reduction. At present,
`_validate_total_ballot_weight()` uses `math.fsum` while
`PreferenceProfile._find_total_ballot_wt()` uses `np.sum`. NumPy uses partial pairwise summation
in this float reduction, not a running correction as Kahan or Neumaier does. Route both Python
operations through `math.fsum` and check for changed decisions near election thresholds. Keep this
work distinct from the selected Rust scalar Neumaier implementation in ADR 0005. Exact totals
remain exact, and empty-input behavior must be specified with the return-type contract.

**Changes in callers**

| Caller                                                                                                                                          | Wiring change                                                                                                                                                                      |
| ----------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `PreferenceProfile._find_total_ballot_wt()` in [preference_profile.py](../../../../src/votekit/pref_profile/pref_profile/preference_profile.py) | Replace `np.sum(self._matrix.voter_weight_data.weights)` with `self._matrix.total_weight()`. Keep the facade's existing cached summary lifecycle.                                  |
| `RankProfile.get_ballot()` and `ScoreProfile.get_ballot()` in the [profile classes](../../../../src/votekit/pref_profile/pref_profile/)         | Obtain the requested row's weight through `get_weight(row)`. Continue using `get_ranking(row)` or `get_scores(row)` for its payload and Python metadata for its voters.            |
| `STV._stv_validate_profile()` in [stv.py](../../../../src/votekit/elections/election_types/ranking/stv/stv.py)                                  | Replace its direct weight-vector indexing when constructing the tied-ballot diagnostic. Preserve the diagnostic's existing content rather than adding voter metadata incidentally. |
| Ranked and scored `to_csv()` in the [profile classes](../../../../src/votekit/pref_profile/pref_profile/)                                       | Use the shared `weight_kind` query for formatting decisions instead of reaching through the matrix to its storage component.                                                       |
| Parent-index validation in [cleaned_pref_profile.py](../../../../src/votekit/pref_profile/cleaned_pref_profile.py)                              | Use the Python metadata interface to retrieve the required labels in bulk. `get_row_label()` is for a single-row request, not a prescription to perform one call per parent row.   |

**Bulk consumers keep bulk operations**

Do not rewrite a scoring or sorting loop as repeated scalar queries. Numerical consumers request a
complete matrix operation; Python consumers that need many values request a batch. In particular:

- `_ordered_profile_rows()` should request stable row ordering from the matrix instead of
  exporting weights or sorting values obtained through repeated `get_weight()` calls.
- Scoring, mentions, and total-weight validation run against owned storage. They must not call
  back into Python once per weight after the Rust port.
- `iter_ballots()` must remain a streaming public interface. Its Rust-backed implementation can
  obtain decoded rows and weights in batches, then attach Python metadata while yielding ballots.
  Do not turn its current per-row delegation to `get_ballot()` into a binding call for every field
  of every ballot since this requires crossing the Python/Rust boundary repeatedly.
- A Python weight-conversion callback may require Python scalar values. Export or iterate them in
  suitable batches and submit the replacement vector once, preserving callback order and failures.
  Keep the existing owned, validated weight-replacement path.

A single requested ballot may justify a scalar binding call. A million-ballot computation does not
justify a million scalar calls merely because those calls avoid an array export. Introduce a batch
interface when a concrete consumer needs it.

Extend the existing weight and row-query tests with scalar type/indexing checks, explicit and
implicit labels, and agreement between total validation and the exposed total. When Rust bindings
arrive, verify that scalar queries do not export whole vectors and bulk consumers do not fall back
to per-row binding calls.

Raw ID handling inside `rank_matrix.py` remains appropriate because that module owns the encoding.
Keep ranked interpretation in that concrete class rather than expanding the common protocol to
expose its storage layout.
