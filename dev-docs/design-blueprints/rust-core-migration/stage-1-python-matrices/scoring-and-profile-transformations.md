# Scoring, mentions, and profile transformations

[Stage 1 overview](README.md) · [Matrix interfaces](matrix-interfaces-and-shared-data.md) ·
[Shared completion checks](README.md#compatibility-and-completion-checks)

## Dependencies

- [Ranked and scored matrix interfaces](matrix-interfaces-and-shared-data.md) for structural
  queries, candidate mapping, and access to aligned weights inside matrix operations.
- Result construction and metadata correspondence for transformations that produce new profiles.

Reductions, partitions, and transformations are work in this guide. Each can proceed without
waiting for the others.

## Used by

- [Election consumers](election-validation-and-counting.md), which adopt scoring and
  first-candidate partitions as each operation becomes available.

## Checklist

Sources: [common_utils.py](../../../../src/votekit/utils/common_utils.py),
[pref_profile/utils.py](../../../../src/votekit/pref_profile/utils.py), and
[exact_fraction_ops.py](../../../../src/votekit/elections/election_types/ranking/stv/exact_fraction_ops.py).
Tests: [test_common_utils.py](../../../../tests/utils/test_common_utils.py),
[tests/pref_profile/utils/](../../../../tests/pref_profile/utils/),
[test_stv_exact.py](../../../../tests/elections/election_types/ranking/test_stv_exact.py), and
[test_stv_weight_conversion.py](../../../../tests/elections/election_types/ranking/test_stv_weight_conversion.py).

- [ ] **Matrix + callers:** Move storage interpretation from
      `_score_dict_from_rankings_df_no_ties()` and `_first_place_votes_from_df_no_ties()` into
      ranked matrix operations. Rename obsolete private `_df` helpers and update their callers
      without changing public entry points.
- [ ] **Matrix + callers:** Route general `score_dict_from_score_vector()` through matrix
      computation. Preserve tie-size score-vector advancement, `high`/`low`/`average` allocation,
      contribution order, and rounding.
- [ ] **Matrix + callers:** Route `_mentions_from_matrix()` through a matrix reduction with one
      contribution per candidate occurrence, including repeated positions. Keep exact weights
      exact until public conversion.
- [ ] **Matrix + callers:** Replace `score_profile_from_ballot_scores()`'s full-ballot loop with a
      score-matrix operation. Preserve `ScoreBallot` coercion, zero-dropping behavior,
      missing-score errors, and output keys.
- [ ] **Matrix + callers:** Replace `ballot_lengths()`'s ballot loop with a weighted
      ranking-length query. Count ranking positions as the current function does, not distinct
      candidates or candidate-set sizes.
- [ ] **Matrix + callers:** Separate `ballots_by_first_cand()`'s row partitioning from ballot
      assembly. Keep the public dictionary of ballot lists while allowing internal callers to
      consume row groups directly.
- [ ] **Matrix + callers:** Move rank-to-score conversion in
      `convert_rank_profile_to_score_profile_via_score_vector()` behind the matrix boundary,
      preserving candidate/column alignment and its conversion policy.
- [ ] **Matrix + callers:** Route `exact_first_place_votes()` and `exact_borda_scores()` through
      exact matrix operations. Preserve their separate empty-set and end-sentinel behavior without
      float intermediates.
- [ ] **Verify:** Check `convert_profile_weights()` validates only replacement weights and
      preserves unchanged rankings, voters, labels, candidate metadata, and source ownership.
- [ ] **Matrix + callers:** Wire `add_missing_cands()` through a ranked transformation. For
      `resolve_profile_ties()`, stream source rows and construct expanded output directly,
      preserving weights, voters, and ordering. Do not change the expansion's combinatorial
      behavior as part of the wiring.

## Scoring implementation layout

The first-place helper delegates to `_score_dict_from_rankings_df_no_ties()`, so move that shared
computation once. `mentions()` already uses the matrix consistently; its remaining work is to move
the reduction into the matrix owner. Neither change requires another public scoring API.

| Implementation owner                                                                  | Remaining caller work                                                                                                                                                                                    |
| ------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `score_matrix.py`: weighted candidate totals with the existing ballot-score semantics | Replace the `.ballots` loop in `score_profile_from_ballot_scores()`. Account for `ScoreBallot` dropping zero scores and treating an all-zero row as unscored; `get_scores()` deliberately retains zeros. |
| `rank_matrix.py`: weighted position counts and ranked transformations                 | Rewire `ballot_lengths()` and `add_missing_cands()`. Length counts explicit empty positions; adding missing candidates may extend the ranking limit.                                                     |
| `rank_matrix.py`: conversion to aligned score storage                                 | Rewire `convert_rank_profile_to_score_profile_via_score_vector()` in `pref_profile/utils.py`. Keep public argument validation and result construction in the wrapper.                                    |
| Existing ranked `with_weights()` path                                                 | `convert_profile_weights()` already uses `_with_weights()`. Move its raw weight access behind a conversion/batch interface while preserving converter calls and replacement validation.                  |
| Python tie expansion with matrix result assembly                                      | `resolve_profile_ties()` should stream inputs and assemble expanded rows without a complete source ballot tuple. Preserve its current candidate inference as well as expansion order and weights.        |

## Ranked scoring caller inventory

Use the existing structural query for the conversion precheck:

| Caller                                                                                                                               | Existing behavior to replace                                   | Matrix method and constraint                                                                  |
| ------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| `convert_rank_profile_to_score_profile_via_score_vector()` in [pref_profile/utils.py](../../../../src/votekit/pref_profile/utils.py) | Scans referenced set IDs and decodes sets just to detect ties. | Use `matrix.has_ties().any()`. Its conversion loop is a separate matrix-operation task below. |

The remaining operations need matrix computation before their callers can delegate:

| Caller                                                                                                                                                                | Work inside the matrix implementation                                                                                          | Work in the caller                                                                                                                                                           |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ballots_by_first_cand()` in [common_utils.py](../../../../src/votekit/utils/common_utils.py)                                                                         | Own first-position interpretation and row partitioning, including first-position ties, empty sets, and padding.                | Assemble the requested ballot lists from row groups. It already uses `get_ranking()` for full-row decoding.                                                                  |
| `_score_dict_from_rankings_df_no_ties()` and `score_dict_from_score_vector()` in [common_utils.py](../../../../src/votekit/utils/common_utils.py)                     | Own ID-to-set interpretation and score accumulation, preserving tie allocation and arithmetic.                                 | Retain argument validation and public result/error construction. `_first_place_votes_from_df_no_ties()` delegates to the former helper; it has no independent decoding loop. |
| `_mentions_from_matrix()` in [common_utils.py](../../../../src/votekit/utils/common_utils.py)                                                                         | Own set membership, occurrence counting, and weighted reduction.                                                               | Keep the existing `has_rankings()` check and public return behavior.                                                                                                         |
| `exact_first_place_votes()` and `exact_borda_scores()` in [exact_fraction_ops.py](../../../../src/votekit/elections/election_types/ranking/stv/exact_fraction_ops.py) | Own exact accumulation and reserved-ID handling. Borda's empty-position check is not answered by either ranking-presence mask. | Preserve singleton-profile preconditions and each scorer's distinct empty-position behavior.                                                                                 |
| `convert_rank_profile_to_score_profile_via_score_vector()` in [pref_profile/utils.py](../../../../src/votekit/pref_profile/utils.py)                                  | Own candidate-column mapping and construction of scores from ranking IDs.                                                      | Validate public inputs and wrap the resulting score matrix and metadata.                                                                                                     |

`_mentions_from_matrix()` already uses `has_rankings()` for its initial check, and
`ballots_by_first_cand()` already uses `get_ranking()` for full-row decoding. Keep those calls
while moving storage interpretation and computation into the matrix owner. A bulk calculation
should not become repeated `get_ranking()` calls.
