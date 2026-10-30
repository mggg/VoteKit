# Ranked cleaning

[Stage 1 overview](README.md) · [Matrix interfaces](matrix-interfaces-and-shared-data.md) ·
[Shared completion checks](README.md#compatibility-and-completion-checks)

## Dependencies

- [Matrix selection and metadata contracts](matrix-interfaces-and-shared-data.md) for aligned
  filtering and source-row correspondence.
- Ranked structural queries and decoding, including the distinction between explicit empty
  positions and trailing padding.

Implement the cleaning transformations within this guide. Candidate removal and condensation do
not need to wait for generic callback cleaning.

## Used by

- [Ordinary STV wiring](election-validation-and-counting.md), which needs candidate removal and
  condensation.

## Checklist

Source: [rank_profiles_cleaning.py](../../../../src/votekit/cleaning/rank_profiles_cleaning.py).
Tests: [tests/cleaning/rank_profiles/](../../../../tests/cleaning/rank_profiles/).

- [ ] **Matrix:** Implement candidate removal by transforming referenced candidate sets, interning
      equal results, and remapping ranking IDs. Preserve explicit empty sets when condensation is
      not requested.
- [ ] **Matrix:** Implement condensation by shifting retained positions left and restoring
      trailing padding. Preserve ranking width and the distinction between empty positions and an
      entirely null row.
- [ ] **Matrix:** Implement repeated-candidate removal with per-row tracking across singleton and
      tied sets. Keep the established first-occurrence behavior.
- [ ] **Callers:** Wire `remove_cand_rank_profile()`, `condense_rank_profile()`, and
      `remove_repeat_cands_rank_profile()` through those operations.
- [ ] **Callers:** Move `remove_and_condense_rank_profile()` onto matrix transformations while
      preserving its existing single result construction with the final candidate metadata.
- [ ] **Matrix + callers:** Wire `remove_ballots_with_cand_rank_profile()` through membership
      queries and row selection instead of using a matrix mask to filter a freshly exported
      DataFrame.
- [ ] **Matrix + callers:** Rework generic `clean_rank_profile()` to decode callback inputs one
      row at a time, validate returned rankings, and assemble the result without a full DataFrame
      intermediary.
- [ ] **Verify:** Preserve all removal flags, retained/inferred candidate rules, source labels,
      unchanged versus altered classifications, null/zero-weight categories, and callback
      invocation order.

## Ranked cleaning implementation layout

The built-in cleaners still share a DataFrame-based transformation path. Candidate removal
followed by condensation already constructs only one cleaned profile; its remaining cost is row
decoding and DataFrame assembly inside that path.

| Implementation owner                                                                             | Callers and division of responsibility                                                                                                                                                                                                                               |
| ------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `rank_matrix.py`: set transformation, ID remapping, condensation, and repeated-candidate removal | `remove_cand_rank_profile()`, `condense_rank_profile()`, `remove_repeat_cands_rank_profile()`, and `remove_and_condense_rank_profile()` select the operation and candidate-retention policy. Return row correspondence and change information for cleaning metadata. |
| `rank_matrix.py`: candidate-membership masks and aligned row selection                           | `remove_ballots_with_cand_rank_profile()` currently builds a raw-ID mask and then exports the profile to filter it. The caller should choose removal flags and attach provenance to the selected result.                                                             |
| Matrix row decoding and validated result assembly                                                | `_iterate_and_clean_ranking_tuples()` and `_clean_rank_profile_with_candidates()` retain callback invocation and cleaning classifications in Python. Preserve their padded callback tuples and classify the returned rows, as described below.                       |

Keep null-row removal distinct from the altered-empty classification: the current filter removes
all-tilde rows, whereas the classification also recognizes rows containing explicit empty sets.

## Ranked cleaning query semantics

| Caller                                                                                                                               | Work inside the matrix implementation                     | Work in the caller                                                                                                                                     |
| ------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `remove_ballots_with_cand_rank_profile()` in [rank_profiles_cleaning.py](../../../../src/votekit/cleaning/rank_profiles_cleaning.py) | Own candidate-membership masks and aligned row selection. | Preserve removal options, candidate metadata, and cleaning provenance. Replace its all-tilde null filter with `has_rankings()` on the selected result. |

The generic cleaning path needs particular care. `_clean_rank_profile_with_candidates()` checks
whether the **transformed** row contains only tildes, and `_iterate_and_clean_ranking_tuples()`
separately classifies changed rows containing only tildes or empty sets. Those correspond to
`has_rankings()` and `has_ranked_candidates()` on the transformed result, respectively, with the
existing unchanged-row exclusion retained. Applying either query to the original matrix would
classify the wrong data. Its callback currently receives padded rows; `get_ranking()` removes
padding, so it is not a drop-in replacement for that callback input.
