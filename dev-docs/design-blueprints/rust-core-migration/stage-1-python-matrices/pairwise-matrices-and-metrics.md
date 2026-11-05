# Pairwise, candidate matrices, distances, and representation

[Stage 1 overview](README.md) · [Matrix interfaces](matrix-interfaces-and-shared-data.md) ·
[Shared completion checks](README.md#compatibility-and-completion-checks)

## Dependencies

- [Matrix queries and encoding contracts](matrix-interfaces-and-shared-data.md) for candidate
  membership, ranking positions, and aligned weights.
- [Profile grouping](profile-facades-and-metadata.md) for normalized distribution construction.
- [Ranking dictionary construction](search-and-profile-exports.md) for the `lp_dist()` path
  through `profiles_to_ndarrys()`.

Pairwise encoding and candidate measures can proceed independently of the distribution paths.

## Checklist

Sources:
[pairwise_comparison_graph.py](../../../../src/votekit/graphs/pairwise_comparison_graph.py),
[matrices/candidate/](../../../../src/votekit/matrices/candidate/),
[distances.py](../../../../src/votekit/metrics/distances.py), and
[representation_scores.py](../../../../src/votekit/representation_scores.py). Tests:
[tests/graphs/pairwise_comparison/](../../../../tests/graphs/pairwise_comparison/),
[tests/matrices/candidate/](../../../../tests/matrices/candidate/),
[tests/metrics/](../../../../tests/metrics/), and
[tests/representation_scores/](../../../../tests/representation_scores/).

- [ ] **Matrix + callers:** Build `pairwise_dict()`'s encoded inputs directly from the matrix.
      Preserve candidate ordering, ballot-presence validation, pair ordering, float conversion,
      and the kernel's skip/stop rules.
- [ ] **Verify:** Keep the existing Numba pairwise tally kernel and its private encoding behind a
      dedicated adapter. Profile set IDs must not be passed through as though they were kernel
      candidate IDs.
- [ ] **Matrix + callers:** Replace full-ballot traversal in `candidate_distance_matrix()`,
      `comentions_matrix()`, and `boost_prob()`/`boost_matrix()` with matrix operations. Preserve
      their distinct rank-distance, directional-comention, weighting, and undefined-result rules.
- [ ] **Matrix + callers:** Build `__build_simultaneous_profile_distribution()` from grouped
      matrices without exporting either profile. Align candidate identities across matrices and
      preserve normalization and explicit-empty-ranking rejection.
- [ ] **Callers:** Trace `earth_mover_dist()`, `lp_dist()`, and `profiles_to_ndarrys()` through
      their shared distribution/dictionary builders so indirect ballot materialization is also
      removed.
- [ ] **Matrix + callers:** Compute `r_representation_score()` from top-position membership and
      weights, counting a ballot at most once. Reuse that operation in
      `winner_sets_r_representation_scores()` without changing winner-set enumeration, warnings,
      or normalization.

## Metrics implementation layout

| Implementation owner                                               | Callers and distinctions to preserve                                                                                                                                                                                                                                                       |
| ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Ranked encoding operation with a dedicated pairwise adapter        | `pairwise_dict()` builds inputs for the existing Numba kernel. Keep its kernel-specific IDs and sentinel rules within that adapter.                                                                                                                                                        |
| `rank_matrix.py`: presence, position, and weighted pair operations | `candidate_distance_matrix()`, `comentions_matrix()`, and `boost_prob()` currently traverse ballots. Presence counts each ballot once; directional comention and candidate distance use the last occurrence of a repeated candidate. Occurrence-based mentions cannot replace these rules. |
| Grouped matrix decoding and normalized distribution construction   | `__build_simultaneous_profile_distribution()` serves `earth_mover_dist()`. `lp_dist()` instead reaches `rank_profile_to_ranking_dict()` through `profiles_to_ndarrys()`. Wire both paths; fixing one does not migrate the other.                                                           |
| `rank_matrix.py`: top-position membership and weighted reduction   | `r_representation_score()` counts each matching ballot once. Keep warnings, winner-set enumeration, and public result formatting in `representation_scores.py`.                                                                                                                            |

Profile grouping already runs through matrices. Distribution builders still need to align
candidate identities across profiles rather than compare their local candidate-set IDs. Transport
solvers, NetworkX operations, and user-supplied distance callbacks retain their existing owners.

## Pairwise validation caller

| Caller                                                                                                           | Existing behavior to replace                          | Matrix method and constraint                                                                                  |
| ---------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `pairwise_dict()` in [pairwise_comparison_graph.py](../../../../src/votekit/graphs/pairwise_comparison_graph.py) | Materializes ballots for ranking-presence validation. | Use `has_rankings().all()`. Keep the pairwise kernel's distinct encoding; input preparation is separate work. |

Keep explicit empty positions valid for this presence check. Requiring ranked candidates would
change accepted inputs. The Numba adapter owns its encoding; its sentinel values are not profile
candidate-set IDs.
