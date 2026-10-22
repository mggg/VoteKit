# Score cleaning

[Stage 1 overview](README.md) · [Matrix interfaces](matrix-interfaces-and-shared-data.md) ·
[Shared completion checks](README.md#compatibility-and-completion-checks)

## Dependencies

- [Matrix selection and metadata contracts](matrix-interfaces-and-shared-data.md) for aligned
  filtering and source-row correspondence.
- Score-column mapping and stored-row decoding that preserve missing values, explicit zeros, and
  the positional callback contract.

Implement score erasure and cleaning result assembly within this guide. Ranked cleaning and
election counting are independent work.

## Checklist

Source: [score_profiles_cleaning.py](../../../../src/votekit/cleaning/score_profiles_cleaning.py).
Tests: [tests/cleaning/score_profiles/](../../../../tests/cleaning/score_profiles/).

- [ ] **Matrix + callers:** Wire `remove_cand_score_profile()` through stored-column mapping,
      score erasure, and column selection. Retaining the original candidate list keeps removed
      candidates' columns with missing values; otherwise select the remaining declared columns.
      Avoid exporting once to find columns and again to construct the result.
- [ ] **Matrix + callers:** Rework generic `clean_score_profile()` to stream its documented
      score-tuple callback inputs and construct aligned matrix output directly.
- [ ] **Verify:** Preserve missing scores versus zeros, empty-row and zero-weight removal flags,
      candidate retention, cleaning categories, and parent provenance.
- [ ] **Verify:** Keep public ballot-level cleaning functions intact. Use them where individual
      ballots are actually requested rather than forcing every profile operation through them.

## Score cleaning implementation layout

`ScoreMatrix.get_scores()` serves ballot decoding, but generic cleaning currently receives a tuple
of every stored score column in column order, including NaNs and any undeclared columns retained
on zero-weight rows. Using `get_scores()` would change that callback contract.

| Implementation owner                                                                   | Callers and division of responsibility                                                                                                                                                                                |
| -------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `score_matrix.py`: stored-row decoding and validated assembly                          | `_iterate_and_clean_score_tuples()` invokes callbacks in row order. Preserve NaN-aware unchanged detection and the classification of all-zero/all-missing rows.                                                       |
| `score_matrix.py`: score erasure, optional column selection, and aligned row selection | `remove_cand_score_profile()` chooses candidate retention and removal flags. Replace its current clean/export/reconstruct sequence with one result construction that retains the original parent and classifications. |
| `score_profiles_cleaning.py`: callback and provenance policy                           | `clean_score_profile()` determines which empty or zero-weight rows to remove. Matrix operations should supply masks and correspondence without deciding those public flags.                                           |
