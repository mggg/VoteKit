API Reference
=============

.. module:: votekit

.. contents:: Table of Contents
    :local:



Ballot
------

.. autoclass:: votekit.ballot.Ballot
    :members:

Preference Profiles
-------------------

.. autoclass:: votekit.pref_profile.PreferenceProfile
    :members:

.. autoclass:: votekit.pref_profile.CleanedProfile
    :members:

.. automodule:: votekit.pref_profile.utils
    :members:
    :show-inheritance:

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
.. automodule:: votekit.cleaning.general_profiles.cleaning
    :members:
    :show-inheritance:

Elections
---------

.. autoclass:: votekit.elections.election_state.ElectionState
    :members:

.. autoclass:: votekit.models.Election
    :members:



Approval-based
~~~~~~~~~~~~~~~
.. automodule:: votekit.elections.election_types.approval.approval
    :members:
    :show-inheritance:


Ranking-based
~~~~~~~~~~~~~~

.. automodule:: votekit.elections.election_types.ranking.abstract_ranking
    :members:
    :show-inheritance:

.. automodule:: votekit.elections.election_types.ranking.alaska
    :members:
    :show-inheritance:

.. automodule:: votekit.elections.election_types.ranking.borda
    :members:
    :show-inheritance:

.. automodule:: votekit.elections.election_types.ranking.boosted_random_dictator
    :members:
    :show-inheritance:

.. automodule:: votekit.elections.election_types.ranking.condo_borda
    :members:
    :show-inheritance:

.. automodule:: votekit.elections.election_types.ranking.dominating_sets
    :members:
    :show-inheritance:

.. automodule:: votekit.elections.election_types.ranking.plurality_veto
    :members:
    :show-inheritance:

.. automodule:: votekit.elections.election_types.ranking.plurality
    :members:
    :show-inheritance:

.. automodule:: votekit.elections.election_types.ranking.random_dictator
    :members:
    :show-inheritance:

.. automodule:: votekit.elections.election_types.ranking.stv
    :members:
    :show-inheritance:

.. automodule:: votekit.elections.election_types.ranking.top_two
    :members:
    :show-inheritance:

Score-based
~~~~~~~~~~~~

.. automodule:: votekit.elections.election_types.scores.rating
    :members:
    :show-inheritance:

Graphs and Viz.
---------------
.. autoclass:: votekit.graphs.ballot_graph.BallotGraph
    :members:

.. automodule:: votekit.graphs.pairwise_comparison_graph
    :members:
    :show-inheritance:

.. automodule:: votekit.plots.mds   
    :members:
    :show-inheritance:

.. automodule:: votekit.plots.bar_plot
    :members:
    :show-inheritance:

.. automodule:: votekit.plots.profiles.profile_bar_plot
    :members:
    :show-inheritance:

.. automodule:: votekit.plots.profiles.multi_profile_bar_plot
    :members:
    :show-inheritance:

Cast Vote Records 
-----------------

.. automodule:: votekit.cvr_loaders   
    :members:
    :show-inheritance:

.. automodule:: votekit.cleaning   
    :members:
    :show-inheritance:

Matrices
--------

.. automodule:: votekit.matrices.heatmap
    :members:
    :show-inheritance:


.. automodule:: votekit.matrices.candidate.boost
    :members:
    :show-inheritance:

.. automodule:: votekit.matrices.candidate.candidate_distance
    :members:
    :show-inheritance:

.. automodule:: votekit.matrices.candidate.comentions
    :members:
    :show-inheritance:


Misc. 
-----

.. automodule:: votekit.elections.transfers   
    :members:
    :show-inheritance:

.. automodule:: votekit.utils   
    :members:
    :show-inheritance:

.. automodule:: votekit.metrics.distances   
    :members:
    :show-inheritance:

.. automodule:: votekit.representation_scores.representation_score
    :show-inheritance:
