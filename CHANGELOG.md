# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## Added
- Created a read the docs page.
- Add `scale` parameter to `ballot_graph.draw()` to allow for easier reading of text labels.

## Changed
- Updated tutorial notebooks; larger focus on slate models, updated notebooks to match current codebase.
- Removed the seq-RCV transfer rule since it is a dummy function, replaced with lambda function.
- Update plot MDS to have aspect ratio 1, remove axes labels since they are meaningless in MDS.
- Update all BLT files in scot-elex repo to be true CSV files, updated `load_scottish` accordingly.

## Fixed
- Fixed bug by which slate-PlackettLuce could not generate ballots when some candidate had 0 support.
- Updated various functions in the ballot generator module to only generate ballots for non-zero candidates.
- Fixed one bloc s-BT pdf, which was incorrectly giving 0 weight to all ballot types.

## [2.0.0] - 2024-03-04

## Added
- A `PreferenceInterval` class.
- MCMC sampling for both `BradleyTerry` ballot generators.
- Add print statement to `BallotGraph` so that when you draw the graph without labels, it prints a dictionary of candidate labels for you.
- Add an IRV election class, which is just a wrapper for STV with 1 seat.
- Add default option to Borda election class, so users do not have to input a score vector if they want to use the traditional Borda vector.
- Add several methods to `PairwiseComparisonGraph`. Added two boolean methods that return True if there is a condorcet winner or if there is a condorcet cycle. Added two get methods that return the winner or the cycles. Cached the results of `dominating_tiers` and `get_condorcet_cycles`.

- Added optional to_float method to `first_place_votes` if users want to see them as floats instead of Fractions.
-Added a `by_bloc` parameter to `generate_profile`. If True, this returns a tuple, the first entry of which is a dictionary of PreferenceProfiles by bloc. The second entry is the aggregated profile. This is very helpful for analyzing the behavior of a single bloc of voters. Defaults to False for backwards compatibility.
- Created a `Cumulative` ballot generator class. The `Cumulative` class works like PL, but samples with replacement instead of without. The ranking order does not matter here, simply that candidates are listed on the ballot with multiplicity.
- Created a `HighestScore` election class. This takes in a profile and a score vector, and returns the candidates with highest scores. There is a lot of flexibility in the score vector, so this class can run things like Borda, cumulative, etc.
- Created a `Cumulative` election class which is just a subclass of `HighestScore` with the score vector set to all 1s. Thus anyone appearing on the ballot gets one point for each time they appear.
- Wrote an `__add__` method for `PreferenceProfile` that combines the ballot lists of two profiles.
- Created utility functions to compute the winners of a profile given a score vector, as well as to validate a score vector (non-negative and non-increasing).
- Created a `shortPlackettLuce` class which allows you to generate ballots of arbitrary length in the style of PL.
- Added tests for  `__add__` method of `PreferenceProfile`.
- Added `SlatePreference` model and tests.

## Changed
- Change the way the `condense_ballots()` method works in profiles. Rather than altering the original profile, it returns a new profile. This gives users the option to preserve the original profile.

- Alter `STV` class so that the remaining candidates are always listed in order of current first place votes.

- Made `PlackettLuce` a subclass of `shortPlackettLuce`.

- Change `PlackettLuce`, `BradleyTerry`, and `Cumulative` ballot generators to have `name_` prefix. This is in contrast to the `slate_` models we have introduced.

- Speed improvements for various ballot generators.

- `pref_interval_by_bloc` is now `pref_intervals_by_bloc` in all ballot generators. This is now a dictionary of dictionaries, where the both sets of keys are the blocs, and the values of the sub-dictionaries are PreferenceInterval objects. The slate models require that we sample from the uncombined PreferenceInterval objects, while the name models require that we combine the PreferenceInterval objects using cohesion parameters.

- MDS plot functionality, splitting it into `compute_MDS` which computes the coordinates, and `plot_MDS` which plots them. Made because the computation is the most time intensive.

## Fixed
- Fixed an error in the `PreferenceProfile` tail method.

- Errors in bloc labeling in `CambridgeSampler`.

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
