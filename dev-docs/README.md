# Developer documentation

This directory contains engineering documentation that belongs in the repository but is not
published with the user documentation in `docs/`.

- `ADRs/` records accepted architectural decisions.
- `specifications/` describes the current system.
- `design-blueprints/` holds proposals and implementation plans in progress.

Structure blueprints around stages or milestones that can be reviewed independently. After an
implementation is approved, record its architectural decisions in an ADR while the rationale is
still fresh. Include the status, context, decision, and consequences, and use a sequential number
and short description for the filename, such as `0001-rust-core-layout.md`.

The [preference profile and election migration](design-blueprints/rust-core-migration/master.md)
plans the move from encoded DataFrames to array-backed profiles and a private Rust core. The
[example migration](design-blueprints/example-rust-core-migration/master.md) remains a structural
and stylistic reference from another package.
