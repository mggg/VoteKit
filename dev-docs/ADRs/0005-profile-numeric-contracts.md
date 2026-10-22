# 0005: Numeric representation, accumulation, and comparison

## Context

Election outcomes can depend on conversion, accumulation order, and rounding as well as comparison
tolerance. The Rust implementation needs explicit arithmetic contracts to distinguish intended
numerical changes from regressions in ties, threshold decisions, and exact results.

## Decision

Weights use either float64 or an all-Fraction representation. Any Fraction input selects exact
storage for the vector, allowing accompanying integers and integral floats because they convert
exactly, but rejecting non-integral floats. Without Fractions, inputs must convert losslessly to
float64, and empty input uses floating storage. Booleans, negative values, and nonfinite weights
are invalid in either representation; other floating dtypes require explicit caller conversion.

Total ballot weight is capped at `2**50`, including after addition or weight replacement. The cap
leaves headroom below float64's consecutive-integer limit but does not prevent the loss of small
fractional contributions through naive accumulation. Rust uses scalar Neumaier compensated
summation for total float weight and float mentions. Each reduction retains a running sum and its
correction in the operation's existing contribution order, returning their sum when complete.
Empty reductions return zero. Exact totals remain rational and obey the same value cap.

This replaces the Python reference's `math.fsum` for those two operations. Neumaier reduces
accumulation error but does not guarantee correct rounding or bitwise agreement with `math.fsum`.
The same scalar algorithm applies on Windows, Linux, and macOS. SIMD implementations and parallel
reductions are outside this implementation's scope.

GMP provides exact arithmetic for the Rust implementation, preserving arbitrary-size rational
values and public Python Fraction results. Explicit rational-to-float conversion follows Python's
rounding and overflow behavior, including nearest-even ties and subnormal values.

`votekit.constants` defines `TOL_EPSILON = 1e-10` and `FLOAT_DECIMAL_PLACES = 10`. Tolerance-based
comparisons are absolute unless an operation specifies otherwise. Rating validation rejects a
budget excess when `total > budget + TOL_EPSILON`; candidate limits and limit/budget consistency
use the same allowance. Exact comparisons, integrality checks, election ties, and quotas retain
their own rules.

Accumulation and rounding follow these operation-specific rules:

| Operation                             | Arithmetic contract                                                                            |
| ------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Total float weight and float mentions | Scalar Neumaier summation in the existing contribution order                                   |
| Grouped float weights                 | Weighted `np.bincount`, with source-order contributions to each bucket                         |
| Score totals per ballot               | Float64 `np.nansum` over declared columns; all-missing rows total zero                         |
| General positional scoring            | NumPy float64 rounding of each weighted allocation to ten places before accumulation           |
| No-ties positional scoring            | Weighted `np.bincount` over row-major contributions, then NumPy float64 rounding to ten places |
| Exact operations                      | Rational arithmetic and exact comparisons without float intermediates                          |

NumPy scalar rounding is part of this reference; substituting Python float rounding can change the
result. The Neumaier decision does not replace the other reductions listed above. Comparison
tolerance is not an accumulation-error bound, and even a correctly rounded float64 result can
differ from the exact sum by more than `TOL_EPSILON`. Changes to reduction order or rounding
require explicit validation of election decisions. Numerical closeness alone does not establish
equivalent behavior.

## Consequences

These rules require distinct floating and exact storage variants with checked conversion between
them. They narrow historical acceptance through the total cap and mixed-input rejection although
the cap limits rational value alone: arbitrarily large numerators and denominators can still make
exact arithmetic expensive.

The
[engineering notes](../design-blueprints/rust-core-migration/engineering-notes.md#numeric-policy)
specify numerical verification. GMP integration and distribution requirements are developed in the
[exact-STV plan](../design-blueprints/rust-core-migration/exact-stv.md).
