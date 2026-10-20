# 0001: Profiles own arrays and expose detached snapshots

Status: implemented; independent review reconciled.

## Context

Internal consumers depended on cached public DataFrames and ballots. Mutating a cache could
change later results while other cached attributes described the original profile. Ranked codes
identify entire ranking sets, including ties and explicit empty positions; they are not candidate
indices. Scored frames can mix integer, unsigned, nullable, floating, and object columns.

## Decision

Built-in profiles own private arrays, row metadata, and a complete frame schema. Ranked codes use
signed int64 with profile-local set meanings and -1 reserved for padding. Each scored column keeps
its own dtype. A native float64 matrix is constructed only after eligibility checks. Public frames,
ballots, and code dictionaries are detached snapshots. Internal readers use private data or fresh
compatibility adapters; unknown profile subclasses retain dynamic public accessors.

Persist the authoritative storage in versioned pickle state, ignoring derived public caches.
Continue loading the old encoded-frame state, including cleaned parents and index metadata.

## Consequences

Mutating a public snapshot no longer changes computation, copies, or serialized profile values.
Native operations pay a checked copy at their boundary. Per-column score storage avoids the
precision loss of stacking uint64(2**63 + 1) and signed integers into float64, but adds temporary
matrix memory for eligible native operations. General frame schemas and exact weights remain
available through compatibility adapters. Successful common numeric operations avoid public frame
and ballot construction.

The per-column variation from the blueprint was accepted by an adversarial design reviewer after
reproducing the precision-loss case. See the [implementation report](../../reports/rust-migration.md).
