# Stage 6: native generators and prepared Bradley–Terry sampling

## Decision and scope

Implemented: costly Bradley–Terry model construction, exact and MCMC sampling, random draws,
and ranking assembly execute in Rust. Prepared exact sampling also groups rows natively and uses
numeric profile construction. MCMC retains DataFrame-based profile construction/combination and
public profile grouping. Python retains configuration validation, labels, and bloc apportionment. Other generators use selected native kernels
where appropriate, including spatial distances, bounded impartial-culture decoding, slate
Plackett–Luce operations, and cumulative assembly. Their Python/NumPy paths remain useful;
custom spatial metrics and unsupported numerical cases retain compatible behavior.

Generator acceleration is an explicit extension of the original profile/election migration.
It does not imply rewriting every generator or replacing a fast NumPy primitive merely because
a Rust implementation is possible.

## Exact model and reusable ownership

Exact name Bradley–Terry sampling uses subset dynamic programming instead of enumerating every
permutation. For remaining candidates `S` and supports `w`, the partition recurrence is
`Z(S) = sum(w[i] ** (len(S) - 1) * Z(S - {i}) for i in S)`, with `Z(empty) = 1`.
Conditional sampling uses those partitions in log space. Setup takes `O(C * 2**C)` work and
`O(2**C)` table storage rather than factorial enumeration. “Exact” describes the sampled model,
not rational arithmetic; this algorithm uses floating-point calculations and is distinct from MCMC.

Slate Bradley–Terry uses a count-state table with `product(slate_size + 1)` states. Its states
represent sequences of slate symbols, so adding a multiplicity factor for interchangeable copies
would change the distribution. Candidate ordering within each slate and the slate sequence
assembly also run natively. Positive supports use stable log-space calculations; zero-support
candidates within a slate form a uniformly shuffled tail. A name model with multiple zero supports
has zero total mass and is rejected; a sole zero can occupy the final position.

`PreparedBradleyTerry` snapshots a validated configuration, including the candidate universe,
bloc order, model, and voter count. It apportions voters once and owns all bloc partition tables
at the same time. Repeated samples reuse those tables. Later edits to the input configuration do
not change the sampler. There is no global cache, implicit cache invalidation, or per-call voter
count override. MCMC remains a separate native sampling route rather than using exact partitions.

Build partition masses and cumulative conditional probabilities in the same ascending-state pass,
sharing the shifted exponentials. Name states use subset bitmasks. Slate states reuse the interaction
sums of the slower-changing counts within each mixed-radix block; calculate the fastest-changing
count's contribution directly to avoid accumulating add/subtract error across states.

Cache dense conditional tables when they fit the 32 MiB cap. Larger tables retain log-space
conditional calculation during sampling; the partition state limit is unchanged. Use the same
checked size calculation for allocation and preparation admission. Sampling cached tables uses
one uniform draw and a short cumulative-probability scan per position without recomputing
exponentials. The cache changes setup and memory costs, not the model, slate multiplicities, or
the generator-to-profile boundary. Ordinary nonprepared native sampling allocates no conditional
cache. Compact transition storage was tested and removed; the retained construction improvements
and complete before/after timings are recorded in the
[probability construction follow-up](../../../reports/bt-probabilities-first-three-2026-09-18.md).

The important public signatures are:

```python
class PreparedBradleyTerry:
    def __init__(self, config: BlocSlateConfig,
                 model: Literal["name", "slate"] = "name"): ...

    def sample(self, *, group_ballots: bool = True,
               rng_seed: Optional[int] = None,
               rng: Optional[random.Random] = None) -> RankProfile: ...
```

`sample_by_bloc` takes the same options as `sample` and returns a dictionary of profiles using the
same ballot draws. Default one-shot exact generators use the prepared implementation internally, so model
logic and validation have one home even when setup cannot be amortized.

## Randomness and grouped output

Passing `rng` or `rng_seed`, including zero, produces deterministic results for a fixed package
version and initial RNG state. Supplying both is invalid. Bradley–Terry uses a native ChaCha12
stream initialized from the caller's Python RNG. Prepared sampling consumes one native seed per
bloc, including blocs apportioned zero ballots. This deliberately does not reproduce the old
Python stream across package versions. Native generator randomness is separate from ordinary
STV's preserved Python tiebreak RNG.

Prepared exact sampling accumulates grouped counts natively and emits rankings in first-occurrence
order. The default one-shot exact generators use this same path.
Grouping does not change the random draws; grouped and ungrouped samples represent the same
weighted rankings. This scoped BT ordering change avoids inheriting pandas sorting of ranking
sets. It does not change the public `group_ballots` operation's separate compatibility contract.

MCMC streams proposals rather than allocating complete proposal/random buffers. Preserve its
specified burn-in, retention positions, and deterministic consumption within the version. Model
errors and native execution failures surface directly, without a retry through another sampler.

## Profiles as the generator/election boundary

Prepared samples and the default one-shot exact generators pass rankings and counts to the checked
numeric profile constructor from
[stage 2](stage-2-array-storage.md), avoiding DataFrame and ballot-object round trips. Owned copies
and public metadata semantics still apply; generated arrays are not trusted merely because their
producer is native.

Generators expose profiles, not election results. Keep generator model parameters and sampling
state independent of election rules, quotas, tiebreaks, and workspaces. Election methods consume
the common profile representation without knowing which generator produced it. A caller composes
the operations through their ordinary APIs:

```python
sampler = PreparedBradleyTerry(config)
profile = sampler.sample(rng_seed=123)
election = STV(profile, n_seats=3, tiebreak="random", rng_seed=456)
winners = election.get_elected()
```

Both sampling and supported election execution remain native internally. The intervening profile
owns NumPy-backed storage; numeric construction avoids ballot objects and DataFrames. Preserve this
boundary when optimizing storage or conversion instead of adding a generator-specific election API.

## Memory admission

Rust owns size estimation because it knows the table, grouping, and output layouts.
Use checked arithmetic for state products, row counts, byte counts, and platform-size conversion,
and fallible allocation for large buffers. Exact table-state and output-size limits bound work
before allocating; MCMC also has a checked work limit. The current constants and detailed estimates
live with the implementation rather than becoming public API guarantees.

Preparation budgets all bloc tables concurrently, not only the largest single table. Each sample
checks current available memory again. Resident prepared tables are already reflected in that
availability and must not be charged twice. Grouping estimates assume worst-case distinct rankings
and include retained keys/counts alongside output arrays. Elections separately admit their own
workspace and history allocations against current availability, which already reflects any
resident sampler tables and profiles. Generator estimates must not depend on an election algorithm.

Admission uses a conservative share of available host memory and discoverable Linux cgroup limits,
including constrained ancestors. It is a heuristic, not a reservation: concurrent allocation and
platform information gaps mean no absolute OOM guarantee is possible. Initial sample preflight
runs before extracting seeds from a supplied RNG. The binding rechecks admission afterward, so a
later failure need not leave that RNG untouched.

## Verification and measured consequences

Check small exact distributions against exhaustive mathematical references, and test slate
multiplicity, zero supports, underflow, deterministic seeds, caller-RNG advancement, MCMC retention,
and errors. Compare grouped/ungrouped weighted rankings and one-shot/prepared sampling. Verify
configuration snapshots, repeated calls, independent samplers, by-bloc results, and memory refusal
before large allocation. Check that generated profiles work through ordinary election APIs,
including native and Python execution, without changing their input profiles.

Measure table preparation separately from repeated draws, numeric profile construction, STV, and
the complete pipeline. Report voter count and stored row count separately: 100,000 sampled voters
may collapse to far fewer grouped rankings. Include integer weights produced by the generator in
public STV benchmarks. Memory checks should be tested with constrained availability and fresh
process measurements, not only small inputs that happen to fit.

Prepared tables and numeric profile construction reduce repeated setup and object construction. After those
changes, generation can still dominate a BT-to-STV workload, so a faster STV kernel may produce
only a modest pipeline improvement. The
[Bradley–Terry report](../../../reports/rust-bradley-terry-2026-09-16.md) and
[prepared sampling report](../../../reports/native-election-core-2026-09-17.md) preserve the
measurements and model corrections behind these choices. Their direct generator-to-election
experiments are historical; that API was removed as recorded in the
[separation report](../../../reports/generator-election-separation-2026-09-17.md).
