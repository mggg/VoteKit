API Reference
=============

.. module:: votekit

.. contents:: Table of Contents
    :local:



Ballot
------

.. autoclass:: votekit.ballot.RankBallot
    :members:

.. autoclass:: votekit.ballot.ScoreBallot
    :members:

Preference Profiles
-------------------

.. autoclass:: votekit.pref_profile.RankProfile
    :members:

.. autoclass:: votekit.pref_profile.ScoreProfile
    :members:

.. autoclass:: votekit.pref_profile.CleanedRankProfile
    :members:

.. autoclass:: votekit.pref_profile.CleanedScoreProfile
    :members:

.. autofunction:: votekit.pref_profile.rank_profile_to_ballot_dict

.. autofunction:: votekit.pref_profile.score_profile_to_ballot_dict

.. autofunction:: votekit.pref_profile.rank_profile_to_ranking_dict

.. autofunction:: votekit.pref_profile.score_profile_to_scores_dict

.. autofunction:: votekit.pref_profile.profile_df_head

.. autofunction:: votekit.pref_profile.profile_df_tail

.. autofunction:: votekit.pref_profile.convert_row_to_rank_ballot

.. autofunction:: votekit.pref_profile.convert_rank_profile_to_score_profile_via_score_vector

Preference Intervals
--------------------

.. autoclass:: votekit.pref_interval.PreferenceInterval
    :members:

Ballot Generators
--------------------

.. automodule:: votekit.ballot_generator
    :members:
    :show-inheritance:

Cleaning
---------

.. automodule:: votekit.cleaning
    :members:
    :show-inheritance:

Elections
---------

.. autoclass:: votekit.elections.ElectionState
    :members:

.. autoclass:: votekit.elections.Election
    :members:



Approval-based
~~~~~~~~~~~~~~~

.. autoclass:: votekit.elections.Approval
    :members:
    :show-inheritance:

.. autoclass:: votekit.elections.BlockPlurality
    :members:
    :show-inheritance:


Ranking-based
~~~~~~~~~~~~~~

.. autoclass:: votekit.elections.RankingElection
    :members:
    :show-inheritance:

.. autoclass:: votekit.elections.Alaska
    :members:
    :show-inheritance:

.. autoclass:: votekit.elections.Borda
    :members:
    :show-inheritance:

.. autoclass:: votekit.elections.BoostedRandomDictator
    :members:
    :show-inheritance:

.. autoclass:: votekit.elections.CondoBorda
    :members:
    :show-inheritance:

.. autoclass:: votekit.elections.DominatingSets
    :members:
    :show-inheritance:

.. autoclass:: votekit.elections.Plurality
    :members:
    :show-inheritance:

.. autoclass:: votekit.elections.SNTV
    :members:
    :show-inheritance:

.. autoclass:: votekit.elections.PluralityVeto
    :members:
    :show-inheritance:

.. autoclass:: votekit.elections.SerialVeto
    :members:
    :show-inheritance:

.. autoclass:: votekit.elections.SimultaneousVeto
    :members:
    :show-inheritance:

.. autoclass:: votekit.elections.RandomDictator
    :members:
    :show-inheritance:

.. autoclass:: votekit.elections.RankedPairs
    :members:
    :show-inheritance:

.. autoclass:: votekit.elections.Schulze
    :members:
    :show-inheritance:

.. autoclass:: votekit.elections.STV
    :members:
    :show-inheritance:

.. autoclass:: votekit.elections.FastSTV
    :members:
    :show-inheritance:

.. autoclass:: votekit.elections.IRV
    :members:
    :show-inheritance:

.. autoclass:: votekit.elections.SequentialRCV
    :members:
    :show-inheritance:

.. autoclass:: votekit.elections.TopTwo
    :members:
    :show-inheritance:

Score-based
~~~~~~~~~~~~

.. autoclass:: votekit.elections.GeneralRating
    :members:
    :show-inheritance:

.. autoclass:: votekit.elections.Rating
    :members:
    :show-inheritance:

.. autoclass:: votekit.elections.Cumulative
    :members:
    :show-inheritance:

.. autoclass:: votekit.elections.Limited
    :members:
    :show-inheritance:

Graphs and Viz.
---------------

.. autoclass:: votekit.graphs.BallotGraph
    :members:

.. autoclass:: votekit.graphs.PairwiseComparisonGraph
    :members:
    :show-inheritance:

.. autofunction:: votekit.graphs.pairwise_dict

.. autofunction:: votekit.graphs.restrict_pairwise_dict_to_subset

.. autofunction:: votekit.plots.compute_MDS

.. autofunction:: votekit.plots.plot_MDS

.. autofunction:: votekit.plots.bar_plot

.. autofunction:: votekit.plots.multi_bar_plot

.. autofunction:: votekit.plots.profile_bar_plot

.. autofunction:: votekit.plots.profile_borda_plot

.. autofunction:: votekit.plots.profile_ballot_lengths_plot

.. autofunction:: votekit.plots.profile_fpv_plot

.. autofunction:: votekit.plots.profile_mentions_plot

.. autofunction:: votekit.plots.multi_profile_bar_plot

.. autofunction:: votekit.plots.multi_profile_ballot_lengths_plot

.. autofunction:: votekit.plots.multi_profile_borda_plot

.. autofunction:: votekit.plots.multi_profile_fpv_plot

.. autofunction:: votekit.plots.multi_profile_mentions_plot

Cast Vote Records
-----------------

.. automodule:: votekit.cvr_loaders
    :members:
    :show-inheritance:

Matrices
--------

.. autofunction:: votekit.matrices.matrix_heatmap

.. autofunction:: votekit.matrices.boost_prob

.. autofunction:: votekit.matrices.boost_matrix

.. autofunction:: votekit.matrices.candidate_distance

.. autofunction:: votekit.matrices.candidate_distance_matrix

.. autofunction:: votekit.matrices.comention

.. autofunction:: votekit.matrices.comention_above

.. autofunction:: votekit.matrices.comentions_matrix


Misc.
-----

.. autofunction:: votekit.elections.fractional_transfer

.. autofunction:: votekit.elections.random_transfer

.. automodule:: votekit.utils
    :members:
    :show-inheritance:

.. autofunction:: votekit.metrics.earth_mover_dist

.. autofunction:: votekit.metrics.lp_dist

.. autofunction:: votekit.metrics.euclidean_dist

.. automodule:: votekit.representation_scores.representation_score
    :show-inheritance:
