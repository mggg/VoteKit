# 0002: Retained election workspaces and deterministic replay of STV

## Context

An STV count changes weights, candidate activity, and each ballot's current position while its
underlying rankings remain fixed. Rebuilding profiles between rounds repeatedly materializes that
unchanged data. Reconstructing rounds by rerunning the election also risks repeating randomness or
callback side effects.

## Decision

### Counting and replay

Rust controls supported counts and implements the existing built-in STV policy. Each count retains
a workspace that shares immutable profile storage and owns changing weights, cursors, candidate
activity, live row order, and transitions. Ordinary rounds do not construct public profiles.

An internal `STVRuleSet` supplies the quota, next-action, transfer, and tiebreak policies. Each
policy uses a Rust implementation or a Python callback with the same operation-specific contract.
Rust implementations execute without entering Python. The rule set selects actions and their
order; the runner validates and applies them to the workspace. Its first implementation preserves
existing STV behavior.

The transition log records transfer factors, row transformations, callback effects, and terminal
completion as they occur. By applying these records to the original input, replay preserves row
multiplicity, exhaustion behavior, and cleaned-profile provenance without repeating decisions,
random draws, or callbacks. Each count retains its own workspace and replay state even when
several counts share an input.

### Public history

Published `ElectionState` values, including their owned score and tiebreak mappings, form an
immutable sequence of completed rounds. All queries, election length, and negative indexing refer
to that sequence, so `get_step()` pairs each reconstructed profile with its recorded summary.
History assembly remains private; callers receive detached, editable values when they export a
state as a dictionary.

### Python extensions

Custom Python behavior enters the Rust runner at defined policy events, including tiebreaks and
transfers. Election and elimination selection invoke the configured tiebreak only when a tie
prevents selection, supplying the pending action as part of the context. The runner releases
incompatible Rust borrows before invoking a callback, then validates its result and records its
effects before continuing. Callbacks may inspect completed rounds and receive materialized
profiles or ballots when needed, but cannot advance the count recursively or access a partially
applied transition.

Execution eligibility is determined before counting or RNG consumption. Custom subclasses that
replace the counting algorithm retain Python execution unless they participate in the runner.
Their existing reconstruction behavior may repeat callbacks and is outside the deterministic
replay guarantee. Once Rust execution starts, failures propagate without a Python retry.

## Consequences

Counting and history materialization have separate costs. Retaining callback results may require
additional storage, especially when a custom transfer cannot be represented by a scalar factor.
Those results become part of the persistence contract.

Immutable history removes the ability to edit published states or reorder completed rounds.
Composite elections must construct replacement summaries when renumbering rounds. Extending
retained replay to legacy custom algorithms will require records or snapshots sufficient to
reproduce their outputs.

The [election-state plan](../design-blueprints/rust-core-migration/stage-4-election-state.md)
specifies the transition records, callback interface, and compatibility checks. Its
[workflow and dispatch sketch](../design-blueprints/rust-core-migration/stage-4-election-state.md#rule-set-orchestration)
shows the Rust/Python boundary. Municipal rule implementations and a public rule-registration API
remain separate work. FastSTV is unchanged during this migration, with deprecation planned for
4.0.
