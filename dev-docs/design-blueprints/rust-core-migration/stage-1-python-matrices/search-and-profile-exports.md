# Search, display, dictionaries, graphs, and CSV

[Stage 1 overview](README.md) · [Matrix interfaces](matrix-interfaces-and-shared-data.md) ·
[Shared completion checks](README.md#compatibility-and-completion-checks)

## Dependencies

- [Matrix selection and ordering](matrix-interfaces-and-shared-data.md) for search and head/tail
  exports, including retained row labels.
- [Profile row decoding and ballot iteration](profile-facades-and-metadata.md) for CSV, ballot
  dictionaries, and ballot graphs. Reuse the existing streaming interface.

Selected-row exports and ballot streaming are independent branches; CSV work need not wait for
search kernels.

## Checklist

Sources: [search_utils.py](../../../../src/votekit/utils/search_utils.py),
[pref_profile/utils.py](../../../../src/votekit/pref_profile/utils.py),
[profile classes](../../../../src/votekit/pref_profile/pref_profile/),
[ballot_graph.py](../../../../src/votekit/graphs/ballot_graph.py), and
[plots/](../../../../src/votekit/plots/). Tests:
[test_search_utils.py](../../../../tests/utils/test_search_utils.py),
[tests/pref_profile/](../../../../tests/pref_profile/),
[tests/graphs/ballot/](../../../../tests/graphs/ballot/), and
[tests/plots/](../../../../tests/plots/).

- [ ] **Matrix + callers:** Move search membership/location queries behind the matrix boundary and
      export only matching rows. Preserve query validation, clone handling, unranked behavior,
      distances, and labels.
- [ ] **Matrix + callers:** Verify `profile_df_head()` and `profile_df_tail()` select and order
      rows before export. Move weight-order selection behind the matrix interface while preserving
      stable ties and summaries.
- [ ] **Callers:** Stream both profile types' CSV rows using `iter_ballots()` or equivalent row
      decoding. Use row count for emptiness checks, and preserve headers, Fraction formatting,
      voters, and ordering.
- [ ] **Callers:** Stream `rank_profile_to_ballot_dict()` and `score_profile_to_ballot_dict()`
      without populating the source profile's complete ballot tuple. Preserve aggregation and
      normalization semantics.
- [ ] **Callers:** Build ranking and score dictionaries from decoded rows without constructing
      unnecessary ballot objects, preserving key order, zero/missing treatment, and scalar
      conversion.
- [ ] **Callers:** Rework `BallotGraph.from_profile()` to consume streamed rankings or matrix
      queries. Preserve graph-node identities, weight aggregation, and ballot-completion behavior.
- [ ] **Verify:** Check profile bar plots and MDS through their scoring and distance helpers. Keep
      plotting and user-supplied metric callbacks in Python; do not introduce a second
      implementation in plots.
- [ ] **Verify:** Keep explicit DataFrame/ballot conversion utilities available for callers that
      request those representations. Remove private helpers only after their remaining callers are
      accounted for.

## Export and presentation implementation layout

Both matrices already support `to_df(rows=...)`, and `profile_df_head()`/`profile_df_tail()`
already select rows before exporting. Reuse that selected-row interface rather than adding another
exporter.

| Location                                                       | Remaining work and owner                                                                                                                                                                                                     |
| -------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `search_utils.py`: `search_profile_for_rank_pattern()`         | Pass matching row positions to the existing selected-row export instead of filtering `profile.to_df()`. This caller change can precede moving search kernels into `RankMatrix`.                                              |
| `pref_profile/utils.py`: `_ordered_profile_rows()`             | Move stable weight ordering into the matrix owner. `_export_profile_rows()` already delegates selected export correctly.                                                                                                     |
| Both profile classes: `to_csv()`                               | Replace `.ballots` checks with row counts and stream decoded rows into the writer. Removing `.ballots` alone is insufficient while `ballot_rows` still collects the whole export in a list.                                  |
| `pref_profile/utils.py`: ballot dictionary builders            | Use existing `iter_ballots()`. Preserve voter-bearing keys and each function's arithmetic: the ranked builder converts weights to float, whereas the score builder retains their numeric type before optional normalization. |
| `pref_profile/utils.py`: ranking and score dictionary builders | Decode rows through matrix operations while preserving key order and missing/zero treatment. Score dictionary keys currently inherit `ScoreBallot`'s zero-dropping behavior.                                                 |
| `ballot_graph.py`: `BallotGraph.from_profile()`                | Use ballot streaming as the direct caller change. Keep tie rejection, short-ballot completion, node construction, and weight aggregation in the graph code.                                                                  |
| Profile plots and MDS                                          | Exercise the migrated scoring and distance helpers. No separate matrix implementation belongs in plotting wrappers.                                                                                                          |

## Search caller inventory

| Caller                                                                                                                                                                                                | Work inside the matrix implementation                                             | Work in the caller                                                                                                                     |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `_get_candidate_set_ids()`, `_get_candidate_set_id_locations()`, `_include_unranked_in_cand_locations()`, and `_boolean_matrix()` in [search_utils.py](../../../../src/votekit/utils/search_utils.py) | Own membership, exact-set matching, position masks, and end-of-ranking locations. | Retain query composition and search policy, then export matching rows only. A row-level presence mask cannot supply ranking positions. |
