# 0007: Editable bloc-slate configuration and owned generation inputs

## Context

`BlocSlateConfig` supports incremental editing through mappings and pandas tables. Intermediate
configurations may be incomplete, and direct table edits bypass property setters. Generation needs
a stricter boundary: once inputs have been captured, subsequent edits must not alter the sample or
its prepared tables.

Structural edits affect slate membership, bloc proportions, preference and cohesion columns, and
retained Dirichlet parameters. These changes require a single owner to keep the configuration
consistent when an update fails.

## Decision

### Editable configuration

Keep `BlocSlateConfig` as the public Python editing interface, including direct table access.
Editing distinguishes invalid values and structural changes from temporarily incomplete
probability totals, preserving warnings for supported intermediate states and the existing
`silent` behavior. Suppressing those warnings does not relax generation validation, and neither
normalization nor sampling occurs without an explicit request.

Collection operations delegate structural changes to the configuration, which prepares dependent
tables and recipe changes before publishing them. This coordinated update leaves the configuration
unchanged on failure and keeps existing collection wrappers attached without recursive setters or
an alpha-invalidation flag. New probability entries start unset while references to replaced
DataFrames retain ordinary pandas behavior.

The configuration owns caller-supplied mutable collections and tables. `copy()` produces
independent editable state, preserving incomplete or malformed tables introduced through direct
pandas edits. Validation occurs at `is_valid()` and generation input preparation; copying does not
repair the data before those boundaries can report it.

### Dirichlet recipes

Retained alphas specify future explicit resampling rather than the provenance of current
preferences.

- Manual preference edits retain the recipe. Explicit resampling overwrites those edits.
- Candidate additions, removals, and renames within existing slates retain each slate's scalar
  alpha.
- Removing blocs or slates trims the recipe to surviving entries. An empty bloc or slate structure
  clears it.
- Adding blocs or slates clears the recipe because the new alpha values are unknown.
- Setting alphas and resampling publish the recipe and preferences only after the complete result
  is ready. Failure preserves both although the caller's RNG may have advanced.

### Generation inputs

Each public bloc-slate generator copies the configuration, validates that copy, and constructs
private immutable inputs. They preserve candidate, slate, and bloc order, voter count,
proportions, preferences, cohesion, and zero-support policy. Nested containers own their data and
retain no mutable configuration collections or DataFrames.

Combined and by-bloc generation use the same capture throughout a call without resampling
preferences or caching results on the mutable configuration. Subsequent edits therefore leave
those inputs unchanged and are validated by the next call. Mutation concurrent with capture
remains unsupported.

Python owns the parent RNG and supplies seed material to Rust's ChaCha8 RNG for sampling. Input
capture consumes no randomness. Sampling is reproducible within a fixed package version but may
change between minor versions.

## Consequences

The editing interface and its diagnostics remain in Python. Rust receives validated numeric inputs
independent of the configuration's subsequent lifetime or edits. This boundary can be implemented
before Rust storage or sampling.

Private preparation can reuse captured parameters and completion tables without retaining the
editable configuration. Voter count belongs to the generation call, not the completion-table key.
Seed encoding and stream behavior are developed in
[stage 6](../design-blueprints/rust-core-migration/stage-6-native-generators.md).
