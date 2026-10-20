# Native election ownership and prepared sampling

The implemented follow-up design is incorporated into the
[core migration blueprint](rust-core-migration/master.md). Its authoritative architectural detail
now lives in:

- [Election ownership and Python execution hooks](rust-core-migration/stage-4-election-state.md):
  Rust control of ordinary STV/IRV, lazy mutable history, replay, GC, and persistence.
- [Profile storage](rust-core-migration/stage-2-array-storage.md): checked numeric construction and
  safe integer-weight handling.
- [Native generators](rust-core-migration/stage-6-native-generators.md): prepared Bradley–Terry
  tables, deterministic streams, grouping, profile boundaries, and generator memory admission.

The [implementation report](../../reports/native-election-core-2026-09-17.md) retains benchmark
results and review findings. Further speed experiments are distinguished from the implemented
architecture in the master blueprint.
