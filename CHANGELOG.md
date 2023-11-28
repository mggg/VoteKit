# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## Added

- Added to_dict(), to_json() functions for election_state (107)
- Scores attribute to election state (113)

## Changed

- Method names for election state methods (107)
- Caches for the results of run_election() (107)
- Optimized random_transfer function (113)
- Ballot objects made immutable, using Pydantic dataclass (113)
- PrefrenceProfile's get_ballots() returns a copy of the ballots list (113)
- compute_votes() returns dictionary and list of tuple to avoid recomputing values (113)

## Fixed

- Multiple winners correctly ordered based off previous rounds vote totals (113)


## [1.0.2] - 2023-09-10

### Fixed

- Bug fixes for election state

### Changed

- Lowered pandas version requirements for wider compatibility


## [1.0.1] - 2023-10-03

### Added

- Elections submodule (e.g from votekit.elections import STV, fractional_transfer)
- More documentation for ballot generators

### Changed

- Renamed CVR loaders to load_csv and load_blt


## [1.0.0] - 2023-10-01

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
[1.0.2]: https://github.com/mggg/VoteKit/releases/tag/v1.0.2
[1.0.1]: https://github.com/mggg/VoteKit/releases/tag/v1.0.1
[1.0.0]: https://github.com/mggg/VoteKit/releases/tag/v1.0.0
