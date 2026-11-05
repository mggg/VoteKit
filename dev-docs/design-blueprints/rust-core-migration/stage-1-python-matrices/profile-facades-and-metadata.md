# Profile facades and metadata

[Stage 1 overview](README.md) · [Matrix interfaces](matrix-interfaces-and-shared-data.md) ·
[Shared completion checks](README.md#compatibility-and-completion-checks)

## Dependencies

- [Shared matrix queries](matrix-interfaces-and-shared-data.md) for totals, weight kind, row
  decoding, and aligned metadata access.
- Matrix grouping, concatenation, and weight replacement, with trusted result construction and
  preserved row correspondence. Reuse the existing implementations.

## Used by

- [Distance distributions](pairwise-matrices-and-metrics.md), which consume grouped profiles.
- [Loader and generator outputs](input-adapters-and-generators.md), which use profile grouping and
  assembly.

## Checklist

Sources: [pref_profile/](../../../../src/votekit/pref_profile/pref_profile/),
[cleaned_pref_profile.py](../../../../src/votekit/pref_profile/cleaned_pref_profile.py), and
[utils.py](../../../../src/votekit/pref_profile/utils.py). Tests:
[tests/pref_profile/](../../../../tests/pref_profile/).

- [ ] **Verify:** Keep `_find_num_ballots()` on its existing `len(self._matrix)` path.
- [ ] **Callers:** Route `_find_total_ballot_wt()` through the matrix total query. Reconcile the
      total used for validation with the exposed profile total, recording existing discrepancies
      separately from the Rust Neumaier change in ADR 0005.
- [ ] **Verify:** Check `get_ballot()` and `iter_ballots()` for both profile types. A single-row
      request should not need full-array exports, and the public `.ballots` tuple should use the
      same decoding.
- [ ] **Verify:** Check that both profile types' `copy()` and `group_ballots()`, ranked
      `_with_weights()`, and `sum_profiles()` use matrix operations and trusted result
      construction rather than public-constructor round trips.
- [ ] **Verify:** Preserve candidate declarations, active-candidate ordering, ranking limits,
      scalar weight types, frozen profile attributes, and the documented result types for
      cleaned/subclass inputs.
- [ ] **Callers:** Route cleaned-profile parent-index validation and label replacement through
      aligned metadata access. Preserve duplicate labels, parent chains, and overlapping cleaning
      categories.
- [ ] **Callers:** Remove incidental full-ballot materialization from equality while preserving
      the current grouping and ballot-equality semantics. Equality redesign remains separate work.
- [ ] **Verify:** Preserve detached `to_df()` and `to_matrix()` exports and the existing `.df`
      deprecation. DataFrame input normalization and schema validation remain legitimate boundary
      operations.

## Profile implementation layout

Both profile types already decode ballots through their matrices, group through matrix operations,
and copy through trusted construction. Profile addition already calls the concatenation helpers in
the matrix modules. These paths need compatibility checks, not another implementation. Ranked
weight replacement is also wired; there is no corresponding score-profile `_with_weights()` to
verify.

| Location                                                     | Remaining work                                                                                                                                                                        |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `preference_profile.py`: `_find_total_ballot_wt()`           | Use the shared total operation in the [matrix query layout](matrix-interfaces-and-shared-data.md#small-queries-and-bulk-access-implementation-layout).                                |
| `rank_profile.py` and `score_profile.py`: `get_ballot()`     | Keep existing ranking/score decoding; replace raw shared-data access with aligned row metadata access. Keep iteration suitable for batching at the Rust boundary.                     |
| `cleaned_pref_profile.py`: `_indices_must_match_parent_df()` | Despite its name, this already reads matrix labels. Route it through the metadata owner rather than treating it as a remaining DataFrame consumer.                                    |
| `cleaned_pref_profile.py`: `_reindex_matrix()`               | Replace dataclass field replacement with a metadata-owned label update that preserves numeric storage and row alignment. The cleaned profile still owns parent/category validation.   |
| `preference_profile.py`: `__eq__()`                          | Remove cached `.ballots` materialization from the grouped comparison. Preserve ballot equality, including voter and weight semantics; structural matrix equality is not a substitute. |
