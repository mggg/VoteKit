# Rust election ownership and sampling

The [matrix migration blueprint](rust-core-migration/master.md) coordinates profile storage,
election execution, and generator integration.

- [Profile storage](rust-core-migration/stage-2-array-storage.md) defines construction, ownership,
  exports, and persistence.
- [Election state](rust-core-migration/stage-4-election-state.md) develops the Rust runner, Python
  policy hooks, immutable history, and retained replay.
- [Generators](rust-core-migration/stage-6-native-generators.md) covers direct matrix construction
  and Bradley–Terry completion sampling.
