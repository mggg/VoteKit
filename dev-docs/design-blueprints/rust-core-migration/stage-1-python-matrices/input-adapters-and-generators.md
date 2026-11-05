# Input adapters and producer output paths

[Stage 1 overview](README.md) · [Matrix interfaces](matrix-interfaces-and-shared-data.md) ·
[Shared completion checks](README.md#compatibility-and-completion-checks)

## Dependencies

- [Profile grouping and concatenation](profile-facades-and-metadata.md) for aggregate and by-bloc
  outputs, including candidate metadata and result construction.
- Existing parsing and input validation at loader and generator boundaries.

Direct buffer construction and sampler changes remain in
[stage 6](../stage-6-native-generators.md) and do not block this consumer wiring.

## Checklist

Sources: [cvr_loaders/](../../../../src/votekit/cvr_loaders/),
[csv_utils/](../../../../src/votekit/pref_profile/csv_utils/), and
[ballot_generator/](../../../../src/votekit/ballot_generator/). Tests:
[tests/cvr_loaders/](../../../../tests/cvr_loaders/), profile CSV tests, and
[tests/ballot_generators/](../../../../tests/ballot_generators/).

- [ ] **Callers:** Check `load_scottish()`, `load_csv()`, and profile `from_csv()` paths for
      internal profile reconstruction after parsing. Route subsequent grouping, metadata queries,
      and transformations through the same matrix operations as other callers.
- [ ] **Verify:** Retain parser DataFrames and individual ballot construction where they establish
      validated external input. Removing pandas from input parsing is not required by this stage.
- [ ] **Callers:** Audit generator calls to grouping, concatenation, and profile utilities so
      producer outputs exercise the shared interfaces. Keep direct sampled-buffer construction,
      bloc capture, prepared sampling, and ChaCha8 integration in stage 6.
- [ ] **Verify:** Leave generator algorithms and their random streams unchanged during consumer
      wiring. The Python/NumPy BT implementation remains the separate starting assumption
      described in stage 6.
- [ ] **Verify:** Check indirect consumers such as animations and plotting wrappers without
      confusing election summary `.scores`, ballot `.scores`, or configuration DataFrames with
      profile storage.

## Producer integration layout

`load_scottish()` already calls `group_ballots()`. The name/slate PL and BT generators, cumulative
generators, and Cambridge generators also use profile grouping; slate BT and Cambridge combine
bloc profiles through addition. Those calls already reach matrix grouping or concatenation through
the profile facade. Verify those integrations rather than replacing them with direct matrix
internals.

| Owner                                        | Work in this stage                                                                                                                                                                                 |
| -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `cvr_loaders/` and `pref_profile/csv_utils/` | Keep parsing and external-input validation. Review operations after parsing for avoidable export/reconstruction; an initial DataFrame or ballot constructor is not itself a migration defect.      |
| `ballot_generator/bloc_slate_generator/`     | Exercise existing grouping and addition with both aggregate and by-bloc outputs. Preserve grouping flags, candidate metadata, and RNG state.                                                       |
| Stage 6 generator implementations            | Direct sampled-buffer construction, configuration capture, prepared samplers, and Rust RNG integration remain later work. Consumer wiring does not imply that those producer changes are complete. |
