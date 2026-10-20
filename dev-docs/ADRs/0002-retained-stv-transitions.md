# 0002: Retain ordinary IRV and fractional STV transitions

Status: implemented; independent review reconciled. The later native election-core migration
supersedes the Python round-orchestration portion of this decision, as described below.

## Context

Ordinary STV rebuilt profiles every round and reconstructed history by rerunning policy decisions.
Those decisions include Python label ordering and random tiebreaks. Native code cannot substitute
its own candidate order without changing outcomes or subsequent random draws.

## Decision

For eligible built-in IRV/STV elections, Python coordinates existing quota, winner/loser, ordering,
and tiebreak policy. An election-local numeric workspace retains input rankings, weights, active
candidates, live row order, and typed transitions. Rust handles ballot-scale tallying, transfers,
filtering, and replay. Python retains labels and provenance. Each checked transition returns its
new generation and candidate summary; invalid or stale transitions fail before mutation.

History replays recorded transitions into a fresh workspace without invoking policy helpers.
Public profiles are materialized on demand. Exact arithmetic, custom subclasses and callbacks,
random transfers, zero-weight input rows, and unsupported inputs retain their compatible Python
election path.

Record finish as an explicit transition. In the baseline, finishing below quota because all
remaining candidates fill the seats eagerly returned an empty RankProfile, but history could
instead rerun elimination, retain an elected candidate, and consume another random draw. Retained
history now returns the actual empty RankProfile, including width zero and no cleaning parent.

## Consequences

Policy still crosses the Python boundary each round with candidate-scale data. This avoids a
second suspend/resume policy protocol while keeping numerical state in Rust. Full history remains
more expensive than the election alone because it constructs public profiles and parent chains.
The finish correction deliberately changes defective baseline history structure; winners and
forward execution remain unchanged. Adversarial review accepted both decisions with transition,
ordering, replay, and RNG invariance checks required.

## Evolution: native round orchestration

The retained transition and replay contracts remain in force. The subsequent implementation moves
ordinary IRV/STV progression into a synchronous Rust runner, with attached Python label ordering
and call-scoped tiebreak helpers. It does not introduce a suspend/resume protocol. A PyO3
`ElectionCore` also owns common queries and lazily materializes the mutable public history list;
Python election subclasses retain their existing execution hooks. See the updated
[election architecture](../design-blueprints/rust-core-migration/stage-4-election-state.md) for
current ownership, eligibility, history mutation, and persistence contracts.
