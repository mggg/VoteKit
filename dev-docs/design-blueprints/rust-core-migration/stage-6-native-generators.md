# Stage 6: direct matrix generators and Bradley–Terry sampling

## Goal

Construct matrices directly from sampled numeric results and port the Python/NumPy Bradley–Terry
implementation to Rust. The sampling algorithms are assumed to be implemented before this
migration; this stage develops their ownership, bindings, and integration with profile storage.

Direct construction removes intermediate ballots and DataFrames. Porting preparation and sampling
lets their buffers become Rust-owned profile storage without a Python array round trip.
[ADR 0001](../../ADRs/0001-profile-array-ownership.md) establishes ownership of those buffers.

## Feature flow

One generation call captures independent inputs, prepares the required sampling state, and builds
a profile from sampled buffers. Subgraphs separate those responsibilities while showing which
state crosses the Python/Rust boundary. Solid arrows show execution or returned data; dashed
arrows show retained inputs or preparation. The diagram shows one RNG stream; per-bloc stream and
batching contracts are described below.

```mermaid
flowchart TD
    START(["START: call a generator"])
    subgraph CAPTURE["Configuration and owned input capture"]
        CONFIG["PYTHON: mutable BlocSlateConfig"]
        INPUT["PYTHON: copy and validate inputs once<br/>Owned parameters, labels, counts, and options"]
        EDIT["Later configuration edits"]
        NEXT["Capture new inputs on a subsequent call"]
        CONFIG --> INPUT
        CONFIG --> EDIT --> NEXT
    end

    subgraph PREPARATION["Deterministic BT preparation"]
        MODEL{"BT model?"}
        NAME["RUST: name-BT subset completion tables"]
        SLATE["RUST: slate-BT count completion tables"]
        TABLES["RUST: immutable prepared state<br/>Reusable for identical preparation inputs"]
        MODEL -->|Name| NAME --> TABLES
        MODEL -->|Slate| SLATE --> TABLES
    end

    subgraph SAMPLING["Sampling and RNG ownership"]
        SEED["PYTHON: draw seed material from parent RNG"]
        RNG["RUST: initialize ChaCha8 stream"]
        SAMPLE["RUST: conditional BT sampling<br/>Apply specified zero-support tail policy"]
        OTHER["Other generators: model-specific sampling<br/>Produce candidate IDs, counts, or scores"]
        SEED -->|"Python → Rust: seed"| RNG --> SAMPLE
    end

    subgraph ASSEMBLY["Direct matrix assembly"]
        BUFFERS["RUST: assemble buffers and combine blocs<br/>Group when requested"]
        MATRIX["RUST: checked matrix construction<br/>Own generated buffers directly"]
        PROFILE["PYTHON: attach captured metadata<br/>and publish generated profile"]
        BUFFERS --> MATRIX
        MATRIX -->|"Rust → Python: handle"| PROFILE
    end

    START --> CONFIG
    INPUT -->|"BT: Python → Rust parameters"| MODEL
    TABLES -->|Preparation ready| SEED
    TABLES -.-> SAMPLE
    INPUT -->|Other models| OTHER
    SAMPLE --> BUFFERS
    OTHER -->|Sampled values; cross binding if Python-owned| BUFFERS
    INPUT -.-> PROFILE
    class CONFIG,INPUT,EDIT,NEXT,SEED,PROFILE python;
    class NAME,SLATE,TABLES,RNG,SAMPLE,BUFFERS,MATRIX rust;
    class START start;
    classDef python fill:#e8f0ff,stroke:#356ac3,color:#111;
    classDef rust fill:#fff0df,stroke:#b86b25,color:#111;
    classDef start fill:#d9f2df,stroke:#26783b,color:#111;
```

## Rust work

Port the Python/NumPy name-BT subset tables, slate-BT count tables, and conditional samplers. Rust
sampling uses a seed supplied by Python and returns buffers directly to Rust profile storage. Move
numeric assembly of sampled IDs and counts into Rust matrix construction, including grouping and
combination of bloc outputs.

## Python work

Implement atomic `BlocSlateConfig` edits and owned generation-input capture. Preserve the one-off
generator functions, configuration diagnostics, candidate labels, and parent RNG. Replace
ballot/DataFrame assembly with direct matrix construction where that provides a useful
intermediate step. Use the Python/NumPy sampling implementation as the behavioral reference and
connect its public entry points to the Rust port. Other generators retain their model-specific
sampling code while adopting the same matrix construction path.

## Configuration editing and generation inputs

[ADR 0007](../../ADRs/0007-bloc-slate-configuration.md) describes configuration editing and owned
generation inputs. Sampling needs validated inputs independent of later configuration edits; that
capture boundary can be developed with the sampler. The broader editing improvements can proceed
alongside it without becoming a prerequisite for every generator port.

The following work develops those responsibilities and can be divided to suit the implementation:

1. Centralize structural updates in the configuration. Prepare dependent tables and recipe changes
   before publication, keeping collection wrappers attached without recursive setters or
   invalidation flags.
2. Implement recipe lifetime and independent copying, preserving malformed direct table edits for
   validation to report.
3. Construct private immutable generation inputs from an owned, validated configuration copy.
   Capture ordered candidates, slates, blocs, and their numeric parameters without retaining
   mutable tables or collection wrappers. This is implementation work, not an existing repository
   interface.
4. Capture once per public call. Combined and by-bloc implementations receive the same validated
   inputs rather than recapturing the mutable configuration through nested calls.
5. Test failed updates, retained collection references, intermediate-state warnings, independent
   copies, malformed tables, recipe transitions, and sampling failures. Subsequent edits must
   leave captured inputs unchanged and be checked on the next call. Isolate capture behavior in
   tests so changes to seeded output can be attributed to sampling changes rather than
   configuration edits.

### Input capture flow

Capture occurs once per public call. Configuration remains editable, but subsequent edits cannot
change the inputs or prepared state used by the current call.

## Direct matrix construction

Keep sampled rankings as candidate IDs and counts until assembly, then translate once to
matrix-local singleton set IDs and combine bloc outputs before constructing the profile.
Throughout this process, preserve candidate metadata, requested grouping, and sampling semantics.

Impartial culture, impartial anonymous culture, spatial models, name Plackett–Luce, name BT, and
shared slate assembly provide useful starting points. Port them individually or together where
shared construction makes a combined change easier to understand. Cumulative generators can
construct score matrices from count arrays, and existing NumPy sampling primitives can feed this
construction path directly.

Batching bounds temporary allocations although ungrouped output still grows with requested rows.
When grouping is requested, aggregate duplicate rankings before profile construction while
preserving weight and order. Slate-pattern counts alone do not determine complete ballot counts
because within-slate identity assignments may produce different rankings from the same pattern.

## Porting the Python/NumPy Bradley–Terry implementation

This outline assumes name-BT subset tables, slate-BT count tables, and conditional sampling
already have a Python/NumPy implementation. Use that implementation and its tests as the reference
for the Rust port, including zero-support tails, within-slate identity assignment, numerical
stabilization, and preparation reuse. Algorithm development is separate from this migration.

Port the preparation and sampling operations behind the existing generator functions. Preserve the
sampling distribution and supported input behavior while adapting storage and random streams to
Rust. The implementation may organize tables and traversal differently without turning the port
into another algorithm-design phase.

## Private preparation and reuse

Keep one-off generator functions as the public interface. Their implementation separates input
capture, table preparation, sampling, and matrix construction. Preparation retains owned
parameters rather than a live configuration; name tables are reusable only for identical supports
and indexing. New Dirichlet draws generally require new tables.

Voter count and bloc apportionment belong to each generation call, not to completion-table
identity. Combined and by-bloc entry points share the model implementation and captured inputs. A
public prepared-sampler object and implicit global caches remain deferred.

Start from the preparation and reuse behavior of the Python/NumPy implementation. Conditional
caches, fused preparation, and mixed-radix traversal remain implementation choices within that
sampling contract.

Shared-prefix count sampling is separate algorithm work unless the Python implementation already
provides it. Preserve the existing MCMC option and its burn-in and retained-sample semantics; a
failed allocation must not silently select a different sampler.

### BT preparation flow

The Python/NumPy reference defines the completion weights and conditional probabilities. Rust
preparation preserves those distributions without consuming randomness; name and slate BT use
different state spaces.

## Randomness and Rust integration

Python owns the parent RNG and draws the seed material for Rust's ChaCha8 RNG, after which
sampling proceeds without a Python call per draw. Input capture and deterministic preparation
consume no randomness. Given a documented seed or initial state, sampling is reproducible within a
fixed package version though samples may change between minor versions.

As the sampler is integrated, define seed encoding and draw boundaries, per-bloc streams,
zero-ballot bloc behavior, batching, and random tail tiebreaks. Test the relationship between
combined/by-bloc and grouped/ungrouped calls. Post-sampling grouping can preserve the draw
sequence; direct count sampling requires its own documented stream behavior. Neither a global RNG
nor implicit reseeding is introduced. Election tiebreak streams retain their separate contract.

Port completion tables and conditional sampling to Rust, using the Python/NumPy implementation and
its small enumerated cases to check the distribution. Python retains configuration validation and
labels; Rust owners retain tables and numeric results. Those results enter a profile directly,
with alignment and candidate/weight invariants checked before publication. Generators remain
independent of quotas, election policies, and mutable workspaces.

### Sampling and matrix assembly flow

For BT, sampling starts with prepared completion tables. Other generators can enter matrix
assembly with their own sampled IDs, counts, or scores. This diagram shows one stream; seed
encoding, per-bloc streams, batching, and random tail tiebreaks retain the contracts described
above.

Preparation is independent of the sampling RNG. Reusing valid tables does not require retaining
the editable configuration or reusing an advanced RNG.

## Resource admission

Carry forward the Python implementation's resource policy, adapting allocation estimates to Rust
storage for preparation, all resident bloc tables, temporary allocations, grouping, and output.
One float64 value per name subset requires 8 MiB at 20 candidates, 256 MiB at 25, and 8 GiB at 30,
before other storage. Batching cannot reduce this table requirement.

Use checked arithmetic to calculate state products, byte counts, and platform conversions,
followed by fallible large allocations where possible. Grouped output must be budgeted for the
worst-case number of distinct rankings rather than an assumed duplicate rate. When allocating a
new sample, check current availability without charging resident tables twice.

Host and applicable cgroup limits can guide conservative refusal but do not reserve memory.
Specify which failures precede caller RNG consumption; later allocation failures need not roll
back an advanced RNG. Resource errors propagate without changing the model or sampler. Select
concrete limits from the production representation and its allocation requirements.

## Verification

Compare Rust preparation and conditional probabilities against the Python/NumPy implementation.
Reuse its small exhaustive-enumeration cases to check name and slate normalizers and ranking
probabilities. Check conditional sums and products along positive-support rankings. Include uneven
supports, zero-support tails, single candidates/slates, multiplicity, and underflow-resistant
calculations. Test multiple-zero cases against the Python implementation's explicit tail rule
rather than the undefined unmodified model. Different RNG algorithms need not produce identical
Python and Rust samples; verify each implementation's seeded behavior and their common
distribution separately.

Verify captured-input independence, repeated batches, deterministic seeds, parent RNG state,
grouping, by-bloc composition, MCMC retention, and refusal before large allocations. Generated
profiles must work through ordinary election APIs.

Verify the full generation-to-election path, including float/Fraction weights, sampled voter
count, and stored row count after grouping. Rust activation requires reference correctness and the
specified tail/RNG behavior. Public prepared samplers and implicit caches remain outside the
initial migration.
