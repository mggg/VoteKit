# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]


## Added
- Add print statement to `BallotGraph` so that when you draw the graph without labels, it prints a dictionary of candidate labels for you.
- Add an IRV election class, which is just a wrapper for STV with 1 seat.
- Add default option to Borda election class, so users do not have to input a score vector if they want to use the traditional Borda vector.
- Add several methods to `PairwiseComparisonGraph`. Added two boolean methods that return True if there is a condorcet winner or if there is a condorcet cycle. Added two get methods that return the winner or the cycles. Cached the results of `dominating_tiers` and `get_condorcet_cycles`.

- Added optional to_float method to `first_place_votes` if users want to see them as floats instead of Fractions.

## Changed
- Change the way the `condense_ballots()` method works in profiles. Rather than altering the original profile, it returns a new profile. This gives users the option to preserve the original profile.

-Alter STV class so that the remaining candidates are always listed in order of current first place votes.

## Fixed
- Fixed an error in the `PreferenceProfile` tail method.

## [1.1.0] - 2023-11-28

## Added

- Added `to_dict()`, `to_json()` functions for `ElectionState` (107).
- Scores attribute to `ElectionState` (113).
- Post init to `Ballots`, handles conversion from float/int types to Fraction (113).
- `_rename_blocs` method to `CambridgeSampler` to rename blocs to historical names. 
- Added `reset` and `run_to_step` methods to `Election` class (116).
- `sort_by_weight` parameter to `head` and `tail` methods of `PreferenceProfile` (115).
- `received_votes` parameter to `get_candidates` method of `PreferenceProfile`. If True, only return 
    candidates that received votes (115).
- `__str__` method for `Ballot` (114).
- Social choice theory documentation (112).

## Changed

- Method names for `ElectionState` methods (107).
- Caches for the results of `run_election()` (107).
- Optimized `random_transfer` function (113).
- `Ballot` objects made immutable, using Pydantic dataclass (113).
- `PrefrenceProfile`'s `get_ballots()` returns a copy of the ballots list (113).
- `compute_votes()` returns dictionary and list of tuple to avoid recomputing values (113).
- Parameter names for `CambridgeSampler`.
- Where the edge labels for `PairwiseComparisonGraph` print, from 1/2 to 1/3.
- The `BallotGraph` now always displays vote totals for a ballot (116).
- `ElectionState` now prints the current round whenever printing (116).
- `load_blt` became `load_scottish` (116).
- `plot_summary_stats` always displays candidates in order inputted (115).
- `PreferenceProfile` data frame now has Percent column, removed the Weight Share column (115).
- `Ballot` object can take floats and ints as weights (114).
- `Ballot` attribute `voter` changed to `voter_set` (114).


## Fixed

- Multiple winners correctly ordered based off previous rounds vote totals (113)
- `CambridgeSampler` correctly computes the frequency of opposing bloc ballots.
- `PreferenceProfile` no longer rounds all weights to integer when printing (114).


## [1.0.2] - 2023-09-09

### Fixed

- Bug fixes for election state

### Changed

- Lowered pandas version requirements for wider compatibility


## [1.0.1] - 2023-09-03

### Added

- Elections submodule (e.g from votekit.elections import STV, fractional_transfer)
- More documentation for ballot generators

### Changed

- Renamed CVR loaders to load_csv and load_blt


## [1.0.0] - 2023-09-01

### Added

- User determined options for breaking ties
- Ballot Simplex object, refactored construction for ballot generator classes
- IC/IAC with generalized arguments
- Improved handling of tied rankings within ballots

## Fixed

- Allow imports from the module level

### Changed

- Optimization for election helper functions


[unreleased]: https://github.com/mggg/VoteKit
[1.1.0]: https://github.com/mggg/VoteKit/releases/tag/v1.1.0
[1.0.2]: https://github.com/mggg/VoteKit/releases/tag/v1.0.2
[1.0.1]: https://github.com/mggg/VoteKit/releases/tag/v1.0.1
[1.0.0]: https://github.com/mggg/VoteKit/releases/tag/v1.0.0
