# Election validation and Python counting

[Stage 1 overview](README.md) · [Matrix interfaces](matrix-interfaces-and-shared-data.md) ·
[Shared completion checks](README.md#compatibility-and-completion-checks)

## Dependencies

- [Matrix queries](matrix-interfaces-and-shared-data.md) for ranking presence, ties, score bounds,
  budget totals, and diagnostic row access.
- [Ranked cleaning](ranked-profile-cleaning.md) for ordinary STV candidate removal and
  condensation.
- [Scoring and partitions](scoring-and-profile-transformations.md) for float/exact tallies and
  first-candidate ballot groups.

Validation and transfer cleanup need not wait for every shared helper. The Rust runner and
retained history remain stage 4 work.

## Checklist

Sources: [ranking elections](../../../../src/votekit/elections/election_types/ranking/),
[rating.py](../../../../src/votekit/elections/election_types/scores/rating.py),
[transfers.py](../../../../src/votekit/elections/transfers.py), and
[models.py](../../../../src/votekit/models.py). Tests:
[test_matrix_validation.py](../../../../tests/elections/test_matrix_validation.py) and
[tests/elections/election_types/](../../../../tests/elections/election_types/).

- [ ] **Callers:** Migrate the remaining veto ranking-presence checks to `has_rankings()`,
      preserving error precedence and materializing only diagnostic ballots. `RankingElection` and
      ordinary STV already use `has_ranked_candidates()`; STV also uses `has_ties()`. Keep those
      narrower rules rather than imposing them on every validator.
- [ ] **Verify:** Check `GeneralRating` uses score bounds and computes score totals only when a
      budget exists. Preserve error precedence, limit comparisons, and the ballot shown in
      diagnostics.
- [ ] **Matrix + callers:** Replace plurality-veto's stored DataFrame, decoded rankings, and row
      iteration with matrix queries and count-local arrays. Preserve cumulative-weight selection
      and RNG draw behavior.
- [ ] **Matrix + callers:** Replace simultaneous-veto's stored DataFrame and repeated `iloc` reads
      with row queries and count-local state, preserving veto order, exhaustion, replay inputs,
      and diagnostics.
- [ ] **Callers:** Rewire ordinary STV's validation, scoring, partitioning, cleaning, and
      weight-conversion consumers through the shared operations. Leave workspace orchestration and
      retained history to stage 4.
- [ ] **Callers:** Keep built-in transfer paths from rebuilding profiles merely to recover ballot
      tuples where the same validated result can be constructed directly. Preserve public transfer
      signatures, custom callbacks, weight types, filtering, and random-transfer draw order.
- [ ] **Verify:** Check composite and pairwise elections through their shared dependencies. Retain
      custom `_run_election`, `_run_step`, scoring, transfer, and tiebreak hooks rather than
      bypassing them.
- [ ] **Verify:** Leave FastSTV, `NumpySTVBase`, and their private encoding untouched, including
      when they share a source file with ordinary STV. Their planned deprecation is not a reason
      to alter them here.

## Election implementation layout

`RankingElection`, ordinary STV, and `GeneralRating` already use structural or score-row queries.
GeneralRating also computes totals only when a budget exists. The remaining validation
replacements are listed below; do not duplicate these completed validators.

| Location                                                        | Remaining work and owner                                                                                                                                                                                               |
| --------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `plurality_veto.py`: initialization and counting                | Replace `_df`, decoded ranking storage, and `iterrows()` with matrix reads. Cumulative-weight selection, voter order, and RNG calls remain election behavior.                                                          |
| `simultaneous_veto.py`: initialization and veto updates         | Replace `_df` and `iloc` interpretation with matrix operations. Keep mutable veto positions, scores, and replay state in the election.                                                                                 |
| Ordinary STV and its shared helpers                             | Adopt the scoring, cleaning, partitioning, and weight APIs as they become available. This does not require the stage 4 runner or retained-history implementation.                                                      |
| `transfers.py`: `fractional_transfer()` and `random_transfer()` | Remove `RankProfile(...).ballots` result reconstruction only after accounting for the normalization and validation it performs. Keep tuple results, custom transfer compatibility, and random-transfer sampling order. |

The matrix owns immutable profile interpretation. Mutable counting state and election policy
remain with the election; replacing a DataFrame does not mean moving the veto algorithm into
`RankMatrix`.

## Validation caller inventory

| Caller                                                                                                                                           | Existing behavior to replace                                                                       | Matrix method and constraint                                                                                                                                  |
| ------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `PluralityVeto._pv_validate_input()` in [plurality_veto.py](../../../../src/votekit/elections/election_types/ranking/plurality_veto.py)          | Materializes all ballots and checks `ranking is None`, interleaved with integer-weight validation. | Use `has_rankings()` for presence, keeping the first invalid row and the existing within-row error order. Do not replace this with `has_ranked_candidates()`. |
| `SimultaneousVeto._sv_validate_input()` in [simultaneous_veto.py](../../../../src/votekit/elections/election_types/ranking/simultaneous_veto.py) | Materializes ballots for `ranking is None`.                                                        | Use `has_rankings().all()` at the same validation point.                                                                                                      |

An explicit empty set supplies a ranking position even if it ranks no candidate. `has_rankings()`
preserves that distinction; `has_ranked_candidates()` would tighten these validators. Ordinary STV
still reads a raw weight for a diagnostic ballot. That is a
[shared-data access task](matrix-interfaces-and-shared-data.md#small-queries-and-bulk-access-implementation-layout),
not missing ranking validation.
