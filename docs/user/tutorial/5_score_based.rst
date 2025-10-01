Score-based Voting
==================

As we saw in the introductory notebook, in addition to ranking-based
voting, there are also a host of election systems that make use of
score-based ballots. By the end of this section, you should be
comfortable with score ballots, Rating elections, and Cumulative
elections and generators.

Score ballots
-------------

First, let’s revisit how to define score ballots.

.. code:: ipython3

    from votekit import Ballot
    
    score_ballot = Ballot(scores={"A": 4, "B": 3, "C": 4}, weight=3)
    print(score_ballot)


.. parsed-literal::

    ScoreBallot
    A: 4.00
    B: 3.00
    C: 4.00
    Weight: 3.0


Notice that despite the scores inducing the ranking :math:`A,C>B`, the
ballot only knows the scores. This is to conceptually separate score
ballots from ranking ballots. If you want to convert a score ballot to a
ranking, you can use the ``score_dict_to_ranking`` function from the
``utils`` module.

.. code:: ipython3

    from votekit.utils import score_dict_to_ranking
    
    ranking = score_dict_to_ranking(score_ballot.scores)
    print("Ranking:", ranking)
    
    ranked_ballot = Ballot(ranking=ranking, weight=score_ballot.weight)
    print("\n", ranked_ballot)


.. parsed-literal::

    Ranking: (frozenset({'A', 'C'}), frozenset({'B'}))
    
     RankBallot
    1.) A, C, (tie)
    2.) B, 
    Weight: 3.0


If you had an entire profile of score ballots and wanted to convert them
all to ranked, you could do so as follows.

.. code:: ipython3

    from votekit import PreferenceProfile
    
    score_profile = PreferenceProfile(
        ballots=[
            Ballot(scores={"A": 4, "B": 3, "C": 4}, weight=3),
            Ballot(scores={"A": 2, "B": 3, "C": 4}, weight=2),
            Ballot(scores={"A": 1, "B": 5, "C": 4}, weight=5),
            Ballot(scores={"A": 0, "B": 2, "C": 0}, weight=3),
        ]
    )
    
    print(score_profile)
    
    ranked_ballots = [
        Ballot(ranking=score_dict_to_ranking(b.scores), weight=b.weight)
        for b in score_profile.ballots
    ]
    
    ranked_profile = PreferenceProfile(ballots=ranked_ballots)
    
    
    print("Ranked profile\n", ranked_profile.df)


.. parsed-literal::

    ScoreProfile
    Candidates: ('A', 'B', 'C')
    Candidates who received votes: ('A', 'B', 'C')
    Total number of Ballot objects: 4
    Total weight of Ballot objects: 13.0
    
    Ranked profile
                  Ranking_1 Ranking_2 Ranking_3 Voter Set  Weight
    Ballot Index                                                
    0               (A, C)       (B)       (~)        {}     3.0
    1                  (C)       (B)       (A)        {}     2.0
    2                  (B)       (C)       (A)        {}     5.0
    3                  (B)       (~)       (~)        {}     3.0


Score ballots are flexible enough to allow any non-zero score, including
negative scores. Scores of 0 are dropped from the dictionary. Note that
not all election methods support negative scoring, but these elections
in ``VoteKit`` validate your ballots and will raise a ``TypeError`` if
an invalid score is passed.

.. code:: ipython3

    score_ballot = Ballot(scores={"A": -1, "B": 3.14159, "C": 0}, weight=3)
    print(score_ballot)


.. parsed-literal::

    ScoreBallot
    A: -1.00
    B: 3.14
    Weight: 3.0


Rating Election
---------------

In a Rating election, to fill :math:`m` seats, voters score each
candidate independently from :math:`0-L`, where :math:`L` is some
user-specified limit. The :math:`m` winners are those with the highest
total score.

.. code:: ipython3

    from votekit.elections import Rating
    
    score_profile = PreferenceProfile(
        ballots=[
            Ballot(scores={"A": 4, "B": 3, "C": 4}, weight=3),
            Ballot(scores={"A": 2, "B": 3, "C": 4}, weight=2),
            Ballot(scores={"A": 1, "B": 5, "C": 4}, weight=5),
            Ballot(scores={"A": 0, "B": 2, "C": 0}, weight=3),
        ]
    )
    
    # elect 1 seat, each voter can rate candidates up to 5 points independently
    election = Rating(score_profile, m=1, L=5)
    print(election)


.. parsed-literal::

          Status  Round
    B    Elected      1
    C  Remaining      1
    A  Remaining      1


Let’s look at the score totals to convince ourselves B was the winner.

.. code:: ipython3

    print(election.election_states[0].scores)


.. parsed-literal::

    {'A': 21.0, 'B': 46.0, 'C': 40.0}


Now let’s see that the Rating election validates our profile before
running the election. All of these code blocks should raise
``TypeError``\ s.

.. code:: ipython3

    ranking_profile = PreferenceProfile(ballots=[Ballot(ranking=[{"A"}, {"B"}, {"C"}])])
    
    # should raise a TypeError since this profile has no scores
    try:
        election = Rating(ranking_profile, m=1, L=5)
    except Exception as e:
        print(f"Found the following error:\n\t{e.__class__.__name__}: {e}")


.. parsed-literal::

    Found the following error:
    	TypeError: Ballot RankBallot
    1.) A, 
    2.) B, 
    3.) C, 
    Weight: 1.0 must be of type ScoreBallot


.. code:: ipython3

    negative_profile = PreferenceProfile(
        ballots=[Ballot(scores={"A": -1, "B": 3.14159, "C": 0})]
    )
    
    # should raise a TypeError since this profile has negative score
    try:
        election = Rating(negative_profile, m=1, L=5)
    except Exception as e:
        print(f"Found the following error:\n\t{e.__class__.__name__}: {e}")


.. parsed-literal::

    Found the following error:
    	TypeError: Ballot ScoreBallot
    A: -1.00
    B: 3.14
    Weight: 1.0 must have non-negative scores.


.. code:: ipython3

    over_L_profile = PreferenceProfile(ballots=[Ballot(scores={"A": 0, "B": 10, "C": 1})])
    
    # should raise a TypeError since this profile has score over 5
    try:
        election = Rating(over_L_profile, m=1, L=5)
    except Exception as e:
        print(f"Found the following error:\n\t{e.__class__.__name__}: {e}")


.. parsed-literal::

    Found the following error:
    	TypeError: Ballot ScoreBallot
    B: 10.00
    C: 1.00
    Weight: 1.0 violates score limit 5 per candidate.


Cumulative election
-------------------

In a Cumulative election, voters can score each candidate as in a Rating
election, but have a total budget of :math:`m` points, where :math:`m`
is the number of seats to be filled. This means candidates cannot be
scored independently, the total must sum to no more than :math:`m`.

Winners are those with highest total score. Giving a candidate multiple
points is known as “plumping” the vote.

.. code:: ipython3

    from votekit.elections import Cumulative
    
    score_profile = PreferenceProfile(
        ballots=[
            Ballot(scores={"A": 2, "B": 0, "C": 0}, weight=3),
            Ballot(scores={"A": 1, "B": 1, "C": 0}, weight=2),
            Ballot(scores={"A": 0, "B": 0, "C": 2}, weight=5),
            Ballot(scores={"A": 0, "B": 2, "C": 0}, weight=4),
        ]
    )
    
    # elect 2 seat, each voter can rate candidates up to 2 points total
    election = Cumulative(score_profile, m=2)
    print(election)
    print(election.get_ranking())
    print(election.election_states[0].scores)


.. parsed-literal::

          Status  Round
    B    Elected      1
    C    Elected      1
    A  Remaining      1
    (frozenset({'B', 'C'}), frozenset({'A'}))
    {'A': 8.0, 'B': 10.0, 'C': 10.0}


Here, B and C tied for 10 points and are thus elected in the same set.

Again, the Cumulative class does validation for us.

.. code:: ipython3

    over_m_profile = PreferenceProfile(ballots=[Ballot(scores={"A": 0, "B": 2, "C": 1})])
    
    # should raise a TypeError since this profile has total score over 2
    try:
        election = Cumulative(over_m_profile, m=2)
    except Exception as e:
        print(f"Found the following error:\n\t{e.__class__.__name__}: {e}")


.. parsed-literal::

    Found the following error:
    	TypeError: Ballot ScoreBallot
    B: 2.00
    C: 1.00
    Weight: 1.0 violates total score budget 2.


Cumulative generator
--------------------

We have a ballot generator that generates cumulative style ballots from
a preference interval. It samples with replacement, thus allowing for
the possibility that you give one candidate multiple points (this is
known as “plumping”).

.. code:: ipython3

    import votekit.ballot_generator as bg
    from votekit import PreferenceInterval
    
    m = 2
    bloc_proportions = {"all_voters": 1}
    slate_to_candidates = {"all_voters": ["A", "B", "C"]}
    
    # the preference interval (80,15,5)
    preference_mapping = {
        "all_voters": {"all_voters": PreferenceInterval({"A": 0.80, "B": 0.15, "C": 0.05})}
    }
    
    cohesion_mapping = {"all_voters": {"all_voters": 1}}
    
    config = bg.BlocSlateConfig(n_voters=100,
                bloc_proportions=bloc_proportions,
                slate_to_candidates=slate_to_candidates,
                preference_mapping=preference_mapping,
                cohesion_mapping=cohesion_mapping)
    
    # the num_votes parameter says how many total points the voter is given
    # for a cumulative election, this is m, the number of seats
    # in a limited election, this could be less than m
    
    profile = bg.name_cumulative_profile_generator(config, total_points=m)
    
    # profile = cumu.generate_profile(number_of_ballots=100)
    print(profile.df)


.. parsed-literal::

                    A    B    C  Weight Voter Set
    Ballot Index                                 
    0             1.0  1.0  NaN    24.0        {}
    1             1.0  NaN  1.0     7.0        {}
    2             2.0  NaN  NaN    66.0        {}
    3             NaN  1.0  1.0     2.0        {}
    4             NaN  2.0  NaN     1.0        {}


Verify that the ballots make sense given the interval. ``A`` should
receive the most votes.

.. code:: ipython3

    Cumulative(profile, m)




.. parsed-literal::

          Status  Round
    A    Elected      1
    B    Elected      1
    C  Remaining      1



**Try it yourself**
~~~~~~~~~~~~~~~~~~~

   Change the preference interval and rerun the election. Does the
   profile make sense?

Conclusion
----------

You have now seen score ballots, Rating elections, and Cumulative
elections and generators. ``VoteKit`` also implements Limited elections,
as well as approval elections, which are like score-based elections but
each candidate can only be scored 0 or 1.

