# VoteKit coding style

VoteKit code should make the underlying election or data operation easy to follow. A reader needs
both the steps of the calculation and enough context to understand its assumptions, valid inputs,
and effects. We favor explicit domain types, focused functions, and interfaces that make those
relationships visible. In Python, this means drawing on the discipline of static type systems
without losing the directness of ordinary Python. The same principles guide the Rust
implementation.

Use this guide for new and substantially changed code, with judgment about the surrounding module.
A focused change should preserve public compatibility and leave unrelated style differences alone.
Where a local convention differs from this guide, retain it if a migration would obscure the work
at hand. Correctness and scientific meaning come first, including the validation and documentation
needed to establish them.

[CONTRIBUTING.md](CONTRIBUTING.md) covers setup, checks, contribution policy, and election
integration. The [developer documentation](dev-docs/README.md) records architectural decisions.
This guide concerns how we express those decisions in code.

## Contents

- [Organization and abstraction](#organization-and-abstraction)
- [Names, layout, and control flow](#names-layout-and-control-flow)
- [Python types and valid states](#python-types-and-valid-states)
- [Ownership, validation, and failures](#ownership-validation-and-failures)
- [Numerical and tabular code](#numerical-and-tabular-code)
- [Rust and the Python boundary](#rust-and-the-python-boundary)
- [Comments, docstrings, and prose](#comments-docstrings-and-prose)
- [Tests and review](#tests-and-review)

## Organization and abstraction

Aim for one thought per function, one larger thought per file, and one cohesive responsibility per
submodule. The point is to let a reader understand an operation without repeatedly leaving it to
recover context. Helpers usually belong near their primary callers, in the same file, and should
name a recognizable responsibility or take ownership of an invariant. An extraction earns its
place when the caller has less to understand afterward. Moving a few lines elsewhere merely to
reduce indentation can have the opposite effect.

Length is a useful prompt to reconsider cohesion. A function exceeding roughly 40 lines in Python
or 70 in Rust deserves a look, though a longer function may be easier to follow when its steps
form one argument. Preserve that context, along with useful docstrings, rather than compressing
statements or creating incidental helpers to meet a size target.

At the module level, organize code around domain responsibilities, as `cvr_loaders`,
`pref_profile`, and `elections` do. File and module names should identify the behavior they own.
Tutorial notebooks and development scripts should call reusable library operations rather than
become their only implementation. Command-line parsing and release or documentation orchestration
belong in the relevant scripts, with additional directories introduced when a responsibility needs
its own home.

Before adding an abstraction, find the existing owner of the behavior and consider what a new
layer would remove from its callers. A wrapper, registry, callback, or base class can be
worthwhile when it centralizes domain knowledge or manages a resource. If it merely adds another
stop in the reading path, a direct operation and a concrete record will usually serve better.
Existing extension points, including election policy callbacks, have callers whose contracts must
survive a refactor. New extension points need a similarly concrete purpose in the current code.

## Names, layout, and control flow

Names should tell readers which operation is happening and what role each value plays.
`build_df_from_ballot_samples`, `validate_weight`, and `load_ranking_csv`, for example,
distinguish construction, validation, and I/O before the reader opens a function. Verbs help make
those operations explicit, though ordinary properties and accessors need no forced action phrase.
For Python, use `snake_case` for functions and variables, `PascalCase` for classes, and
`UPPER_SNAKE_CASE` for module-level constants.

The same care applies to values that share a primitive type but have different meanings. Counts,
weights, indices, and labels should be distinguishable by name, with units or scope included where
readers would otherwise have to infer them. A table name such as `ballot_weights_df` gives more
context than bare `df`. Short mathematical names remain appropriate when nearby equations or the
algorithm establish their meaning.

Within a function, whitespace should separate immediate operations. Preparing inputs, checking a
condition, updating state, and returning a result often form distinct groups, including inside
loops and branches. Keep coordinated assignments together and give substantial control-flow blocks
space around them. Ruff handles formatting, import order, and linting, but the author still has to
choose these logical paragraph breaks. Follow the configured 100-character width for code,
comments, and docstrings, and wrap prose similarly where practical.

Validation and guard clauses belong before the work that depends on them. Early returns and
`continue` often make the main path easier to follow, especially when they prevent several levels
of nesting. Aim for three indentation levels or fewer where practical without scattering a
cohesive operation. Short comprehensions work well for a single transformation or selection. When
validation, branching, side effects, or several accumulated results need explanation, an ordinary
loop gives those decisions room to be read.

At call sites, keyword arguments help distinguish parameters with similar types. Within an
operation, direct comparisons between known attributes make the relevant fields visible whereas
iterating over attribute names with `getattr` makes the reader reconstruct that information.
Implementation files use absolute imports, such as `from votekit.elections.election_state import
ElectionState`; relative imports are reserved for re-exports in `__init__.py`.

## Python types and valid states

Types are part of an interface's design: they tell callers which inputs are supported and what
states a result can represent. Annotate parameters and return values, including those of
nontrivial helpers, so those choices remain visible through the operation. Obvious local values
can be inferred while empty containers and ambiguous intermediates may benefit from an explicit
annotation. New syntax must remain compatible with the supported Python baseline.

### Choose representations that explain the domain

Choose a type that expresses what the operation actually supports. A ranked-only operation should
accept `RankProfile` since accepting `PreferenceProfile` would leave its assumptions about
rankings hidden in the body. Shared behavior belongs behind a shared interface: the existing
`ProfileMatrix` protocol serves that role for common matrix operations, with rank-specific work
retaining a rank-specific interface.

A dataclass works well when named fields form one record, especially where a long positional tuple
would leave their roles unclear. If alternative states carry different data, a union of records
can express that relationship without optional fields and Boolean flags that permit contradictory
combinations. Enums suit shared named choices while `Literal` is often enough for a small fixed
set. A Boolean remains appropriate for an independent binary choice.

Candidate scores are naturally a typed key/value mapping, and `TypedDict` can describe a fixed
dictionary schema, as it does for `ElectionPlay`, although it provides no runtime validation.
Configuration rules belong with the class that owns the configuration, such as `BlocSlateConfig`
and its validation helpers. Reuse those rules rather than introducing a separate validation
framework for an internal record. Likewise, generics should express a useful relationship between
types and protocols should describe interchangeable behavior. Their value lies in the contract
they reveal to a caller.

Consider a private tie-selection operation whose callers must distinguish a selected candidate
from a tie that still needs resolution. The following sketch makes those outcomes explicit without
proposing a new public VoteKit API:

```python
from dataclasses import dataclass

from votekit.types import Candidate


@dataclass(frozen=True)
class SelectedCandidate:
    """
    A candidate selected for the available seat.
    """

    candidate: Candidate


@dataclass(frozen=True)
class UnresolvedTie:
    """
    Candidates still tied for the available seat.
    """

    candidates: frozenset[Candidate]


Selection = SelectedCandidate | UnresolvedTie
```

After narrowing with `isinstance` or `match`, a caller has exactly the payload relevant to that
outcome. Construction must still establish that `UnresolvedTie` contains at least two valid
candidates, which its annotation cannot guarantee. If the operation simply returns a candidate or
raises an exception, that simpler interface is sufficient without introducing a result hierarchy
to imitate Rust.

### Preserve distinctions through the workflow

Once an operation has a typed record, retain it across internal interfaces and convert to a
dictionary, serialized form, or binding payload where the consumer requires that representation.
Repeatedly unpacking and reconstructing it loses the benefit of having made its structure
explicit. A type alias can give that structure a readable name, but it does not create a distinct
type. Where confusing candidate indices with candidate-set IDs would cause a real error, `NewType`
can provide a static distinction. A small validated class can enforce runtime constraints as well.
Choose the level of protection the operation needs, reusing existing domain types where possible.

Absence needs an equally clear meaning. Within one contract, `None` should not stand for several
unrelated states such as missing, invalid, pending, and failed. Use `is None` when zero, an empty
collection, or another false-valued result is valid. Once a branch has established a narrower
state, retain it rather than making later code infer the same state again from flags.

Keep `Any` confined to boundaries where untyped libraries make it unavoidable, then validate or
convert values before they enter domain operations. For a value whose type is genuinely unknown,
`object` requires the caller to narrow it before use. A `cast` only supplies information to the
checker, so it cannot substitute for conversion or runtime validation. Casts and targeted checker
suppressions are appropriate when an established guarantee cannot be expressed to the checker,
with a brief explanation where the reason would otherwise be unclear.

Container interfaces should reflect how the operation uses its inputs. `Mapping` and `Sequence`
are suitable when their interfaces suffice while a concrete container can make a representation or
mutation requirement explicit. Neither a read-only interface nor an annotation prevents another
owner from changing the underlying data. An `Iterable` also permits single-use inputs, so
accepting one requires an implementation that does not assume indexing or repeated traversal.

## Ownership, validation, and failures

Pure transformations are usually easiest to reason about because their inputs account for their
results. Pass dependencies explicitly and keep file access and random-state advancement at
identifiable boundaries. Mutable globals need a specific shared lifetime to justify them. Resource
cleanup should belong to the operation that opens the resource, usually through a context manager.

When callers depend on mutation, copying, sharing, or lifetime guarantees, make those guarantees
part of the contract. A frozen dataclass prevents field reassignment but leaves embedded lists,
DataFrames, and NumPy arrays mutable. Similarly, an array view may expose shared storage through
an otherwise read-only interface. The applicable storage contract must establish ownership rather
than relying on annotations or a frozen wrapper to imply it.

Validation belongs where the meaning of the constraint is known. A shape check cannot establish
candidate identity, for instance, and a type check cannot establish nonnegative finite weights.
Validate user inputs, loaded data, and binding inputs before relying on them, using explicit
exceptions for invalid external values. Python can disable assertions, so reserve them for
internal invariants established by earlier validation. When several callers rely on the same
invariant, share its validation. An owned value that already satisfies the contract can then pass
through internal operations with checks on the constraints each transformation affects, avoiding
repeated parsing as though it were new external input.

Failure handling should preserve the distinction between an expected domain outcome, invalid
input, and a programming error. Keep established exception types, including `ProfileError` where
the public API requires it, and catch failures only where the operation can handle them
meaningfully. Empty results and zero values can conceal an unexpected failure. Where translating
an exception helps the caller, retain useful context and chaining without exposing sensitive
input.

A useful error message identifies the violated constraint and the observed value. For example,
`ValueError(f"n_seats ({n_seats}) must be positive.")` tells a caller what to correct, unlike
`ValueError("Invalid input.")`. When rejecting an option, include the supported choices from the
same constants used by validation so the explanation remains accurate as those choices change.

## Numerical and tabular code

Numerical code needs to preserve the meaning of its values as well as their representation. Ballot
count and voter weight answer different questions, candidate labels serve a different role from
storage indices, and a missing score may have a different interpretation from zero. Ordering,
ties, provenance, and empty-input behavior are part of the contract wherever they affect results.

Use NumPy and pandas operations directly when they express the calculation clearly. If a chain
combines several independent decisions, named intermediate values can make those decisions easier
to inspect. Units, denominators, exclusions, and normalization rules belong in the explanation
where they affect interpretation. Array shape, dtype, alignment, and value constraints still need
validation at the boundary responsible for them because `NDArray` annotations cannot establish
those properties on their own.

A readability refactor must preserve exact versus floating arithmetic, reduction order, rounding,
tolerance, and tie handling. Even numerically close results can lead to different election
decisions. The migration's [numeric contracts](dev-docs/ADRs/0005-profile-numeric-contracts.md)
record the relevant arithmetic choices and remain their authoritative home. When performance calls
for less obvious code, retain the invariant or reason that explains it and verify the behavior it
protects.

## Rust and the Python boundary

The shared principles of organization, readability, and numerical correctness apply to Rust as
well as Python. The following conventions guide their application as Rust enters the
implementation. Storage, execution, and distribution follow the
[migration decisions](dev-docs/README.md), with Cargo tooling and the native extension to be
configured as that work proceeds.

### Rust types and valid states

Structs and enums should make the relationships between values explicit. Named struct fields help
when positional values are easy to confuse while an enum with payloads expresses states that carry
different information. Where two concepts share a primitive representation but must not be mixed,
a newtype supplies the distinction that a `type` alias cannot. Derived traits also make promises
to callers: choose them when their semantics fit the domain and they serve a current use. A
`Default` implementation, for example, should produce a meaningful valid state.

The Python selection example can be expressed in Rust as one enum. As before, this is a design
illustration rather than a proposed public API or a choice of index representation for the port:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CandidateIndex(usize);

#[derive(Debug, PartialEq, Eq)]
enum Selection {
    Selected { candidate: CandidateIndex },
    UnresolvedTie { candidates: Vec<CandidateIndex> },
}
```

The enum associates each outcome with its data while construction must still establish index
bounds, distinct candidates, and the minimum size of a tie. Those checks belong where the
candidate universe is known. If callers outside the owning module must not bypass an invariant,
private fields and checked construction can enforce that boundary, provided mutation preserves it
too.

When handling these states, an exhaustive `match` makes each case visible and requires a decision
when a new variant is added. Use a wildcard arm only when all remaining cases, including future
ones, should receive the same treatment. For a single relevant case, `if let` often expresses the
intent more directly.

Concrete functions and types are a good starting point for an API. A trait becomes useful when
callers need a shared behavioral contract, and generics when the implementation operates across
supported types. Runtime-selected implementations or heterogeneous values may call for
`dyn Trait`. Each choice should answer a current need, with bounds limited to the capabilities the
operation uses. A trait for every struct, parameters for hypothetical backends, or a macro around
ordinary control flow adds complexity without necessarily clarifying a contract.

### Borrowing, mutation, and resource ownership

A signature should reveal the operation's relationship to its inputs. `&T` permits borrowed
access, `&mut T` permits exclusive mutation, and `T` transfers the value when the operation
consumes or retains ownership. Small `Copy` values such as indices usually work well by value.
When only contents matter, prefer `&[T]` to `&Vec<T>` and `&str` to `&String`. For mutation, a
slice is enough to change existing elements whereas resizing requires access to the vector itself.

Keep mutable bindings and borrows close to the work that needs them. If the borrow checker seems
to demand a clone, first reconsider ownership and the order of operations. An independent copy may
be the correct choice, but substantial allocation should be apparent at the call site. Cloning an
`Arc<T>`, for example, shares its allocation rather than copying `T` into independent storage.

Immutable profiles borrowed or shared by separately owned election workspaces give each count
control over its changing state. Adding `Arc`, locks, or interior mutability should follow an
identified sharing requirement, not postpone deciding who owns an operation.

### Errors, absence, and conversions

`Option<T>` represents meaningful absence while `Result<T, E>` lets a caller respond to a
recoverable failure. Preserve that distinction instead of silently substituting an empty
collection or default value for an error. If callers need to distinguish failure kinds, a small
error enum gives them that information without parsing message strings. Python-specific exceptions
belong in the binding layer.

At a fallible call, decide what responsibility the current function has for the failure. It may
need to handle a particular case, add context, translate the error into a domain failure, or pass
it to its caller. Use `?` only when the resulting error already satisfies the function's contract
and an early return leaves its state valid. Where this layer knows the affected candidate, ballot,
or round, include that context when it helps the caller understand the failure. An explicit
`match` is useful when error variants require different responses. Choose the control flow after
deciding how the error should be handled.

Malformed public inputs must produce errors rather than reach `unwrap`, `expect`, or a panic. An
internal `expect` can be justified when an established invariant makes failure impossible and its
message explains why success is expected. Where practical, express that guarantee in the type or
control flow instead. Tests have a different purpose: `unwrap` or `expect` is appropriate when an
unexpected failure should fail the test. The Rust Book's
[guidance on panics](https://doc.rust-lang.org/book/ch09-03-to-panic-or-not-to-panic.html)
develops this distinction.

Conversions deserve the same attention to failure and meaning. Use `From` for an infallible,
value-preserving conversion and `TryFrom` when conversion can fail, implementing those traits
rather than `Into` or `TryInto` directly as described by the
[standard conversion traits](https://doc.rust-lang.org/std/convert/index.html). A named method is
clearer when conversion requires a policy such as rounding or normalization. Avoid `as` casts that
silently truncate, wrap, or lose precision, and remember that an integer bounds check alone cannot
establish an exact integer-to-float conversion. The operation's numeric contract determines
conversion and overflow behavior, including consistency between debug and release builds.

### API shape, modules, and iterators

Rust names follow the usual language conventions: `snake_case` for functions, methods, variables,
and modules, `PascalCase` for types, traits, and enum variants, and `UPPER_SNAKE_CASE` for
constants. Visibility should reflect who needs the interface. Start with private items, widening
to `pub(super)`, `pub(crate)`, or a public interface as consumers require. Within that boundary, a
module can hold a coherent responsibility and its helpers together without giving every type a
separate file or every field a public getter and setter.

For conversion methods, the
[Rust naming conventions](https://rust-lang.github.io/api-guidelines/naming.html#c-conv)
communicate both cost and ownership: `as_` supplies a cheap borrowed view, `to_` retains the
source and may allocate, and `into_` consumes the source. Other operations should use domain verbs
that explain their work. Constructors, in particular, should establish the guarantees their names
and types advertise so callers need not remember a separate validation step.

Iterator chains suit a clear transformation or reduction. As several decisions accumulate, named
intermediates or a loop often make the calculation easier to inspect. Stateful algorithms and
coordinated updates usually benefit from that explicit sequence. Collect into a `Vec` when reuse
or ownership requires materialization, avoiding an intermediate allocation whose only purpose is
to be iterated once. Whether the operation uses loops or iterators, its specified accumulation
order and arithmetic must remain intact.

### Rust documentation and verification

Rust documentation should explain the contract beyond the signature, using `///` for items and
`//!` for module or crate context. Add `# Examples`, `# Errors`, `# Panics`, and `# Safety`
sections where they apply. A public unsafe function must document the safety conditions callers
need to uphold. Keep unsafe blocks small and place a `// SAFETY:` explanation nearby to show why
the relevant preconditions hold, including allocation lifetime, alignment, bounds, and aliasing as
applicable. The enclosing abstraction must prevent safe callers from violating those guarantees.

Examples should compile and demonstrate an observable result. Hidden setup lines can keep the
published example focused while preserving a complete test. Where the environment prevents an
example from running, `no_run` still permits compilation checks; `ignore` must not conceal a
broken example. The
[documentation test guide](https://doc.rust-lang.org/rustdoc/write-documentation/documentation-tests.html)
explains these controls, and the
[Rustdoc guide](https://doc.rust-lang.org/rustdoc/how-to-write-documentation.html) covers API
prose. Mathematical explanations and references belong alongside the operation whose
interpretation depends on them.

Focused unit tests belong near the owning module, with integration tests exercising the public
boundary. Relevant invariants may include invalid construction, ownership isolation, arithmetic
edge cases, and state transitions. For the Python/Rust port, combine differential tests with small
independently checked fixtures because two implementations can agree on the same mistake. Compare
winners, ties, ordering, and exception behavior as well as numerical values, and verify seeds and
random-state advancement against the operation's reproducibility contract.

As the Rust build is established, use `rustfmt`, Clippy, unit tests, and documentation tests under
the repository's configuration. Any lint suppression should be narrow and explain its reason.
Crate layout, dependencies, supported toolchains, and detailed lint settings belong with the
implementation and build configuration, where those choices can be made against actual code.

### Python and Rust interoperability

The migration design places Python normalization and public diagnostics in adapters, with numeric
operations behind the native interface. Expected Rust errors therefore need translation into the
established Python exceptions. Shape, value, ownership, ordering, and arithmetic contracts must
survive that translation and the accompanying data conversions.

The [profile ownership decision](dev-docs/ADRs/0001-profile-array-ownership.md) specifies copying
and sharing guarantees. Before exposing an array or running without the interpreter, establish
that writable aliases and allocation lifetimes satisfy those guarantees. Where practical, use the
same behavioral fixtures on both sides of the binding to check that the interface preserves what
callers depend on.

## Comments and docstrings

Documentation should supply the contract and reasoning a reader cannot recover from a signature.
Public classes, functions, and methods need docstrings, and nontrivial internal operations often
benefit from them too. Retain mathematical explanations, invariants, and useful examples even when
they make the function longer.

### Python docstrings

VoteKit uses a modified Google style with a 100-character line limit and the summary on its own
line inside the docstring. Include `Args`, `Returns` or `Yields`, `Raises`, and class `Attributes`
where they apply, documenting types, meaningful defaults, and optional behavior. The following
example omits the body to show the format:

```python
def validate_seat_count(n_seats: int, candidate_count: int) -> None:
    """
    Require a positive seat count no greater than the number of candidates.

    Args:
        n_seats (int): Number of seats to fill.
        candidate_count (int): Number of available candidates.

    Raises:
        ValueError: The seat count is nonpositive or exceeds the candidate count.
    """
    ...
```

A docstring should help callers use the operation and interpret its result. Depending on the API,
that may require accepted states, ordering, mutation, ownership, side effects, or failure
behavior, with examples for non-obvious usage. Scientific APIs may also need formulas, references,
units, weights, and tie-breaking rules. Rust APIs need the same substantive information in
Rustdoc's conventions rather than Python's section syntax.

When editing, ask whether removing a sentence would make misuse, misinterpretation, or mishandling
a failure more likely. If so, retain it. Descriptions of temporary variables, helper handoffs, or
facts already evident in the signature rarely carry that weight. Trim conservatively, preserving
useful sections and explanations when their value is uncertain. Check the documentation against
return paths, validation, and exceptions so its guarantees match the implementation.

### Comments

Clear names and direct code reduce the need for commentary, leaving comments to explain reasons
the code cannot show. A terse inline explanation is usually enough, though a non-obvious invariant
may warrant a short paragraph. Concrete worked cases generally belong in named tests unless they
are needed as public API examples. Comments should describe durable behavior and constraints,
independent of the plan or review that prompted them, with references where an external source
explains the rule. Describe failure modes precisely, using terms such as “malformed input” or
“corrupt data” when those are the actual concerns.

## Tests

Tests should establish observable behavior and the invariants on which correctness depends. Small,
independently understood examples are useful for numerical results, ties, empty inputs, and
invalid data where those cases bear on the change. Include meaningful failure behavior as well as
success, without tying tests to incidental helper structure or repeating the calculation from the
implementation. The [testing guidelines](CONTRIBUTING.md#testing-guidelines) cover test placement
and commands.
