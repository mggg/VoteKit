---
hide:
  - navigation
---

# API Reference

## Objects 
### ::: votekit.ballot
    rendering:
    heading_level: 4

### ::: votekit.pref_interval
    rendering:
    heading_level: 4

### ::: votekit.pref_profile
    rendering:
      heading_level: 4

### ::: votekit.election_state
    rendering:
      heading_level: 4

### ::: votekit.graphs.ballot_graph
    rendering:
      heading_level: 4

### ::: votekit.graphs.pairwise_comparison_graph
    rendering:
      heading_level: 4

## CVR Loaders
### ::: votekit.cvr_loaders
    rendering:
      heading_level: 4

## Ballot Generators
### ::: votekit.ballot_generator
    rendering:
        heading_level: 4
    options:
        members:
            - BallotGenerator
            - BallotSimplex
            - slate_PlackettLuce
            - name_PlackettLuce
            - slate_BradleyTerry
            - name_BradleyTerry
            - AlternatingCrossover
            - CambridgeSampler
            - OneDimSpatial
            - ImpartialCulture
            - ImpartialAnonymousCulture
            - name_Cumulative

## Elections
### ::: votekit.elections.election_types
    rendering:
      heading_level: 4

## Cleaning
### ::: votekit.cleaning
    rendering:
      heading_level: 4

## Metrics
### ::: votekit.utils
    rendering:
        heading_level: 4
    options:
        members:
            - borda_scores
            - first_place_votes
            - mentions
### ::: votekit.metrics.distances
    rendering:
        heading_level: 4
    options:
        members:
            - earth_mover_dist
            - lp_dist
            - em_array
### ::: votekit.elections.transfers
    rendering:
        heading_level: 4
    options:
        members:
            - fractional_transfer
            - seqRCV_transfer
            - random_transfer

## Plotting
### ::: votekit.plots.mds
    rendering:
        heading_level: 4
### ::: votekit.plots.profile_plots
    rendering:
        heading_level: 4

## Utils
### ::: votekit.utils
    rendering:
        heading_level: 4
    options:
        members:
            - compute_votes
            - remove_cand
            - fix_ties
            - elect_cands_from_set_ranking
            - scores_into_set_list
            - tie_broken_ranking
            - candidate_position_dict



