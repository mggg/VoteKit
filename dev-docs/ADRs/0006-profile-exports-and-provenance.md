# 0006: Detached exports and explicit row provenance

## Context

Profile exports must preserve both ownership and row identity. A mutable DataFrame cannot serve as
the authority for a frozen profile because edits can leave cached totals inconsistent with
subsequent computations. Nor can an array position stand in for a source label: filtering,
grouping, and concatenation must carry labels and cleaning relationships through their numeric
transformations.

## Decision

`to_df()` constructs an independent snapshot on each call. `.df` delegates to it and is deprecated
as of 3.6.0, with removal scheduled for the next major release. Profiles retain no authoritative
or permanently cached `_df`. Exports may share immutable candidate and voter sets, but their
writable array storage is independent.

Matrix exports accept positional row selections and decode only those rows, preserving source
labels, output order, repeated selections, and the specified negative-index behavior. `get_ballot`
constructs one row and `iter_ballots` streams rows; the existing `ballots` tuple remains
available. Editing an exported frame or mutable ballot payload cannot change the profile or
another export.

Public labels may repeat or be nonconsecutive, so they remain distinct from physical positions
even when they are integers. Each transformation carries the correspondence it needs: selected
source positions for filtering, group membership for weight aggregation and voter unions before
index reset, and reconciled candidates with aligned metadata for concatenation. Cleaned profiles
also retain their parents and classification categories, including permitted overlaps.

Numeric results and metadata are published together after alignment checks. Persistence stores a
versioned representation of authoritative values and provenance, excluding derived caches and live
handles to Rust objects. Restoration validates the stored relationships before publishing an owned
result. Compatibility includes older pickles and cleaned parent chains. Conflicting caches cannot
substitute for missing authoritative values. Pickle remains a trusted-input format.

## Considered Alternatives

Caching a DataFrame and copying it for each caller would isolate edits but retain another full
representation for the profile's lifetime. Fresh exports let callers decide when to incur and
retain that allocation.

## Consequences

Repeated pandas work should reuse a caller-held export. Internal computation and serialization use
authoritative matrix values while Python can continue to retain rich labels and provenance.
Transformations must return enough row correspondence to reconstruct those relationships without
inferring them from a new default index.

The [storage plan](../design-blueprints/rust-core-migration/stage-2-array-storage.md) covers
exports and persistence.
[Election replay](../design-blueprints/rust-core-migration/stage-4-election-state.md) adds the
provenance required by recorded transitions.
