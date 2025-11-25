# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.3.1] - 2025-11-24
## Added

## Changed 
- implemented speed improvements to sBT, sPL, and CS
- refactored ballot generator tests

## Fixed
- fixed a floating point error in STV transfers (#311)
- fixed a cohesion issue in Cambridge sampler (#313)

## [3.3.0] - 2025-10-02
## Added
- Created a `BlocSlateConfig` class for ballot generators. This validates that all of the inputs to a ballot generator are valid.
- `FastSTV`, a new implementation of the `STV` class that relies more heavily on numpy and should be faster.
- `RankBallot` and `RankProfile`, and `ScoreBallot` and `ScoreProfile`, now allowing us to distinguish between the two types of input. The old
`PreferenceProfile` and `Ballot` constructors still work and create instances of `Rank` and `Score` for you.
- `RankedPairs`, `StarVoting` and `OpenList`, all new election methods.

## Changed 
- refactored the codebase to use the new updates.
- deprecated the ballot generator classes in favor of functions.
- `PreferenceProfile` and cvr imports can now use urls.

## Fixed
- Issues 252, 229, 258, 248, 164, 205, 269, 233, 169, 285, 282, 238, 303, 302


## [3.2.2] - 2025-07-02
## Added

## Changed 

## Fixed
- fixed an error with the `STV` class. In edge cases where not enough candidates receive votes to pass threshold, the class was not properly counting the number of remaining candidates.



## [3.2.1] - 2025-06-17
## Added

## Changed 
- removed the `PreferenceProfile` warning raised when `max_ranking_length>0` but no rankings provided.
We have decided an empty profile can still have positive max ranking length.

## Fixed
- fixed an error raised by an empty ranking of `~` symbols being passed to `utils.ballots_by_first_cand` .

## [3.2.0] - 2025-06-13
## Added
- created a `PreferenceProfile.df` attribute that is a pandas `DataFrame` representation of the profile. The df is in bijection with the profile, and using the df allows for great speed improvements throughout the codebase.
- a new tutorial notebook replicating the Portland, OR election case study.
- added a `to_pickle` and `from_pickle` method to `PreferenceProfile`.

## Changed 
- separated the computation of the pairwise comparison dictionary from the pairwise comparison graph.
- renamed `dominating_tiers` method of pairwise comparison graph to `get_dominating_tiers`.
- renamed `cleaning.deduplicate_profile` to `cleaning.remove_repeated_candidates`.
- moved `remove_cand` from `utils` to `cleaning`, and removed the older function `remove_noncands`.
- renamed `PreferenceProfile.condense_ballots()` to `PreferenceProfile.group_ballots()`.
- removed `Ballot.id` attribute.
- rewrote many cleaning functions and utilities to use the underlying dataframe of `PreferenceProfile` for speed improvements.
- removed the use of `Fraction` and use floats instead for speed improvement.
- altered the various plotting functions for profiles, now in the `plots.profiles` module.

## Fixed
- `PreferenceProfile.group_ballots()` now also groups the `voter_set` attribute of ballots.

## [3.1.0] - 2025-03-03
## Added
- added support for three types of averaging conventions within `score_profile_from_rankings`: average,
low, and high.
- r-representation scores: compute how "satisfied" voters are with a given winners set.
- matrices: create three kinds of matrices based on profiles: boost, mentions, and average distance. Accompanying heatmap
code that plots them.
- support for Python 3.12, 3.13
- contributing guidelines and community resources to our docs.

## Changed 
- changed the default Borda scoring to use low averaging, where tied rankings receive the lowest possible
points
- changed the sampling method for `boosted_random_dictator` and `random_dictator` to more clearly
use the first place votes distribution
- changed the structure of `plot_summary_stats`. Is now split into many functions, all called
`profile_STAT_plot` or `multi_profile_STAT_plot`, where `STAT` can be `fpv`, `borda`, `mentions`, 
and `ballot_lengths`. Built on top of more general `bar_plot` and `multi_bar_plot` functions.
- removed support for Python 3.9.

## Fixed

## [3.0.0] - 2024-08-15

## Added
- The election methods thanks to @kevin-q2. The newest election methods are `PluralityVeto`,
  `RandomDictator`, and `BoostedRandomDictator`.
- More comprehensive tests for the `Ballot` class.
- More comprehensive tests for all of the election types.
- More comprehensive tests for the `PreferenceProfile` class.
- Tests for potential errors in the `BallotGenerator` classes.

## Changed
- Moved the following election methods to sorted folders in `src/votekit/elections/election_types`:
    - `approval`
        - `Approval`
        - `BlockPlurality`
    - `ranking`
        - `RankingElection`
        - `Alaska`
        - `BoostedRandomDictator`
        - `Borda`
        - `CondoBorda`
        - `DominatingSets`
        - `PluralityVeto`
        - `Plurality`
        - `SNTV`
        - `RandomDictator`
    - `scores`
        - `GeneralRating`
        - `Limited`
        - `Rating`
        - `Cumulative`

- Updated / added the following methods to `src/votekit/utils.py`:
    - `ballots_by_first_cand`
    - `remove_cand`
    - `add_missing_cands`
    - `validate_score_vector`
    - `score_profile`
    - `first_place_votes`
    - `mentions`
    - `borda_scores`
    - `tie_broken_ranking`
    - `score_dict_to_ranking`
    - `elect_cands_from_set_ranking`
    - `expand_tied_ballot`
    - `resolve_profile_ties`

- Changed the way that the `ElectionState` class operates. It now operates as a dataclass
    storing the following data:
    - `round_number`: The round number of the election.
    - `remaining`: The remaining candidates in the election that have not been elected.
    - `elected`: The set of candidates that have been elected.
    - `eliminated`: The set of candidates that have been eliminated.
    - `tiebreak_winners`: The set of candidates that were elected due to a tiebreak within a round.
    - `scores`: The scores for each candidate in the election.

- The `PreferenceProfile` class is now a frozen dataclass with the idea being that, once the
    ballots and candidates for an election have been set, they should not be changed.
    - Several validators have also been added to the `PreferenceProfile` class to ensure that
        the ballots and candidates are valid.

- Updated all documentation to reflect major changes in the API of the package. 


## Fixed



## [2.0.1] - 2024-06-11
## Added
- Created a read the docs page.
- Add `scale` parameter to `ballot_graph.draw()` to allow for easier reading of text labels.
- Allow users to choose which bloc is W/C in historical Cambridge data for CambridgeSampler.

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
- Added a `by_bloc` parameter to `generate_profile`. If True, this returns a tuple, the first entry of which is a dictionary of PreferenceProfiles by bloc. The second entry is the aggregated profile. This is very helpful for analyzing the behavior of a single bloc of voters. Defaults to False for backwards compatibility.
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
