# Stage 4: Rust election ownership and Python extensibility

## Decision and status

Implemented: `ElectionCore` owns common history/query behavior, and supported ordinary IRV and
fractional STV, including exact rational mode, run their control loop in Rust. Python `Election`
remains the public class and extension point. The initial retained-engine decision in
[ADR 0002](../../ADRs/0002-retained-stv-transitions.md) established replay and transition semantics;
the subsequent migration moved round orchestration from Python into Rust without replacing them.

A Python implementation remains a first-class election implementation. There is no new public
backend registry, abstract engine protocol, or mandatory native counterpart for a user election.

## Execution boundary

`Election.__init__` performs its public setup and invokes `self._run_election()` through ordinary
Python dispatch. The existing extension hooks remain (annotations shown for clarity; `P` is the profile type):

```python
def _run_election(self) -> None: ...

def _run_step(self, profile: P, prev_state: ElectionState,
              store_states: bool = False) -> P: ...
```

The default Python loop uses the subclass's step and completion logic. Built-in STV/IRV select
native execution only for their complete supported configuration. A custom subclass or override
keeps Python dispatch; accelerating a base class must not bypass overridden behavior.
Python elections store their existing `ElectionState` objects, including `Fraction` values,
in the core without converting them to floating point. `NumpySTVBase` is a separate hierarchy and
is not forced into this ownership model.

Rust's native runner is synchronous. It owns quota comparisons, election/elimination decisions,
termination, transitions, and compact round recording. Numerical scans and transfers detach from
the interpreter. Python set/frozenset ordering runs through the attached PyO3 boundary, and rare
tiebreak events invoke a call-scoped Python callback. There is no suspend/resume request protocol.

Normal progression constructs no ballot objects or public profiles per round. A supported Borda
or current-first-place winner tiebreak may materialize an array-backed profile at that event.
Loser ties use the initial scores and existing backup behavior. The ordinary election's Python
`random.Random` instance remains the source of random choices, with its established draw order.

## State ownership

The master blueprint gives the recommended
[`RoundState` and `Workspace` layouts](master.md#recommended-core-types) and their invariants.
The boundary owner connects those numeric types to Python labels and history. These are selected
fields of the implemented owner, not a complete binding API:

```rust
pub struct ElectionCore {
    pub labels: Vec<Py<PyAny>>,
    native: NativeHistory,
    history: Option<Py<PyList>>,
}

enum NativeHistory {
    Empty,
    Floating(CompletedRun<f64>),
    Exact(CompletedRun<ExactWeight>),
}
```

`RoundState` records candidate-index groups for remaining/elected/eliminated candidates, ordered
scores, and resolved tiebreaks. `TransitionData` is the binding/serialization representation of
the numeric transition log. Python-driven elections can instead supply Python states; the
native variant is empty because common query ownership does not require native execution.
Each completed native run owns its rounds, workspace and transitions for one numeric type.

The PyO3 owner deliberately retains labels and public states. It implements garbage-collection
traversal and clearing so cycles through user objects or history can be collected. By contrast,
the numerical `Workspace` owns only Rust values: initial rankings and weights, current weights,
active candidates, live row order, and transitions. Labels and callbacks never enter detached
numeric loops. Call-scoped callbacks are not retained on the core.

Replay through a Python subclass can call back into election queries. Release the mutable core
borrow before invoking that Python hook. During a native STV run, policy callbacks operate on
supplied context rather than reentering the borrowed running core.

## Lazy, mutable public history

Before `election_states` is requested, a native election answers common queries from compact Rust
rounds. On first access, it materializes and caches the existing Python `ElectionState` objects in
a real list. Do not replace that list with a proxy: callers can append, replace, reorder, and edit
its states, or assign a replacement list whose identity must be preserved.

After materialization or replacement, query methods read the public states and observe their
edits. The election's construction-time `length` remains fixed, as in the existing public model;
it is not silently recomputed whenever a caller edits the list. The immutable execution log
continues to describe the actual native run. Editing a summary does not rewrite ballots or
transfer factors. This distinction is essential when a user edits history and later asks for a
round profile.

`get_status_df` remains Python presentation over core query results. Exact/custom states keep
original objects and numeric values rather than round-tripping through native floating-point
summaries.

## Numeric transitions and profile replay

The workspace's important transition variants are:

```rust
pub enum Transition {
    Eliminate(usize),
    Elect {
        removed: Vec<usize>,
        groups: Vec<usize>,
        factors: Vec<f64>,
    },
    Finish,
}
```

Record actual transfer factors and group order. Do not recompute them from rounded public scores,
rerun winner selection, or consume RNG during native history access. Python-derived set order
can change live row order and subsequent floating-point reductions, so it is computational state,
not presentation metadata.

Keep original rankings and weights plus a transition log instead of retaining every round's
ballot matrix. Replay builds a fresh numerical workspace, and the Python adapter constructs the
requested profile and cleaned parent chain. Requests can be repeated or made out of order without
mutating the source profile or another election. Full public history can still be expensive
because it intentionally constructs profiles; it is measured separately from execution.

Filtering must match each transition's semantics. In simultaneous transfers, rows retained before
peer removal can become positive-weight empty rankings. A generic exhausted-row cleanup would
change history. `Finish` records the actual terminal empty `RankProfile`, with width zero and no
cleaning parent, instead of rerunning a baseline elimination when all remaining candidates win.

## Eligibility and persistence

Ordinary native dispatch requires the built-in IRV/STV types, supported built-in profile/schema,
canonical singleton rankings, fractional transfer, a positive usable quota, and strictly positive
finite weights. Supported weights are `float64` or integers with a Python-integer total at most
`2**53`. Workspace conversion preserves exact integer values in that domain and leaves the
source profile's original dtype unchanged. Zero-weight rows, unsafe ordinary-mode integer totals,
custom transfers/subclasses, or unsupported metadata select compatible Python execution.

Exact native dispatch uses the same layout and transfer restrictions, with positive `Fraction`
weights and a positive exact quota. Python validates/converts exact inputs and computes the quota;
Rust performs unrounded rational tallies, comparisons and transfers. Custom configurations and a
forced Python backend keep the eager exact implementation. Exact history and replay return
`Fraction` values, including across pickle/deepcopy. See [exact STV](exact-stv.md).

The private numeric workspace accepts some inputs outside ordinary public dispatch, including
zero weights. These are explicit contracts, not reasons to broaden ordinary STV eligibility
without validation. If a Hare outcome
would elect more candidates than available seats, native execution raises an explicit `ValueError`
instead of the eager implementation's eventual `IndexError`. This accepted error clarification does
not change valid outcomes or authorize other validation-timing changes.

Versioned election serialization contains labels, Python history, numeric workspace inputs, and
transitions rather than a live native handle. Materializing history for pickle/deepcopy preserves
aliases within the serialized object graph. Restoration validates numeric payloads and recreates
query/replay ownership without new policy decisions or random draws. Legacy election state is
restored into the same core. Profile persistence has its separate contract in stage 2.

## Verification and performance consequences

Compare every round and query, not only winners: candidate and dictionary order, scores, ties,
row order, weights, metadata, cleaned parents, terminal profiles, index errors, and RNG state.
Include mutable/replaced history, exact states, custom execution hooks, callback reentrancy,
garbage collection, pickle/deepcopy aliases, and repeated/out-of-order profile queries.

Retaining the workspace removes the repeated profile-construction cost. Lazy state conversion is
useful for ownership and compatibility, but measured conversion cost is small relative to sampling
and tallying. Benchmark ordinary execution, numeric replay, and public materialization separately.

The optional `stv-cursors` feature adds per-row positions and cached summaries to the workspace.
It is disabled by default and still recomputes tallies in live-row order. Neither cursor indexing
nor incremental tally maintenance is an architectural prerequisite. See the
[cursor experiment report](../../../reports/faststv-cursor-benchmark-2026-09-17.md) before choosing
further traversal changes.
