# Developer documentation

This directory holds engineering documentation that is not published with the user documentation
in `docs/`.

- `ADRs/` records architectural decisions and their rationale.
- `specifications/` describes the current system.
- `design-blueprints/` organizes proposed work into evolving outlines.

An ADR records a focused architectural decision: the constraint that motivates it, the choice
made, and its operational consequences. Use a “Considered Alternatives” section when a rejected
option helps explain the choice. Implementation sketches, progress, and experimental results
belong in the blueprint or its supporting documents. ADRs use sequential numbers and descriptive
filenames without status lines.

Blueprints serve as working outlines for implementation. They develop the approach; explain how
its parts relate; and identify dependencies, compatibility concerns, and questions worth
investigating. Their structure gives an implementer a place to start and room to revise the design
as the work reveals better options. Reorder or combine sections, change the proposed interfaces,
and pursue a small working path across several areas when that helps resolve uncertainty.

ADRs record the architectural decisions behind an outline; diagrams and code sketches explore how
to realize them. Keep the blueprint aligned with what is learned during implementation. When that
learning calls an architectural decision into question, revisit its rationale in the ADR as well.

The [matrix migration blueprint](design-blueprints/rust-core-migration/master.md) covers the
remaining Python integration and the transition to Rust-owned storage. The
[example migration](design-blueprints/example-rust-core-migration/master.md) is a structural and
stylistic reference from another package.

The repository [coding style guide](../STYLE-GUIDE.md) covers Python and Rust implementation
conventions, including types, ownership, and documentation.

## Accepted matrix migration decisions

- [0001: Rust ownership of profile storage](ADRs/0001-profile-array-ownership.md)
- [0002: Retained election workspaces and deterministic replay](ADRs/0002-retained-stv-transitions.md)
- [0003: A private Rust extension](ADRs/0003-private-rust-distribution.md)
- [0004: Matrix-owned ballot interpretation](ADRs/0004-matrix-representation-and-operations.md)
- [0005: Numeric representation, accumulation, and comparison](ADRs/0005-profile-numeric-contracts.md)
- [0006: Detached exports and row provenance](ADRs/0006-profile-exports-and-provenance.md)
- [0007: Editable bloc-slate configuration](ADRs/0007-bloc-slate-configuration.md)
