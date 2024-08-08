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

Preference Intervals
--------------------

.. autoclass:: votekit.pref_interval.PreferenceInterval
    :members:

Ballot Generators
--------------------

.. automodule:: votekit.ballot_generator
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

.. automodule:: votekit.elections.election_types.ranking.condo_borda
    :members:
    :show-inheritance:

.. automodule:: votekit.elections.election_types.ranking.dominating_sets
    :members:
    :show-inheritance:

.. automodule:: votekit.elections.election_types.ranking.plurality
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

.. autoclass:: votekit.graphs.pairwise_comparison_graph.PairwiseComparisonGraph
    :members:

.. automodule:: votekit.plots.mds   
    :members:
    :show-inheritance:

.. automodule:: votekit.plots.profile_plots   
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
