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
    
    score_ballot = Ballot(scores = {"A":4, "B": 3, "C":4}, weight = 3)
    print(score_ballot)
    print("ranking:", score_ballot.ranking)


.. parsed-literal::

    Scores
    A: 4.00
    B: 3.00
    C: 4.00
    Weight: 3
    
    ranking: None


Notice that despite the scores inducing the ranking :math:`{A,C}>B`,
the ballot only knows the scores. This is to conceptually separate score
ballots from ranking ballots. If you want to convert a score ballot to a
ranking, you can use the ``score_dict_to_ranking`` function from the
``utils`` module.

.. code:: ipython3

    from votekit.utils import score_dict_to_ranking
    
    ranking=score_dict_to_ranking(score_ballot.scores)
    print(ranking)
    
    ranked_ballot = Ballot(ranking = ranking, weight = score_ballot.weight)
    print(ranked_ballot)


.. parsed-literal::

    (frozenset({'C', 'A'}), frozenset({'B'}))
    Ranking
    1.) C, A, (tie)
    2.) B, 
    Weight: 3
    


If you had an entire profile of score ballots and wanted to convert them
all to ranked, you could do so as follows.

.. code:: ipython3

    from votekit import PreferenceProfile
    score_profile = PreferenceProfile(ballots = [Ballot(scores = {"A":4, "B": 3, "C":4}, weight = 3),
                                      Ballot(scores = {"A":2, "B": 3, "C":4}, weight = 2),
                                      Ballot(scores = {"A":1, "B": 5, "C":4}, weight = 5),
                                      Ballot(scores = {"A":0, "B": 2, "C":0}, weight = 3)])
    
    print("Score profile\n",score_profile)
    
    ranked_ballots = [Ballot(ranking = score_dict_to_ranking(b.scores),
                            weight = b.weight) 
                    for b in score_profile.ballots]
    
    ranked_profile = PreferenceProfile(ballots = ranked_ballots)
    
    
    print("Ranked profile\n", ranked_profile)


.. parsed-literal::

    Score profile
     Ranking                   Scores Weight
         () (A:1.00, B:5.00, C:4.00)      5
         () (A:4.00, B:3.00, C:4.00)      3
         ()                (B:2.00,)      3
         () (A:2.00, B:3.00, C:4.00)      2
    Ranked profile
                   Ranking Scores Weight
                (B, C, A)     ()      5
    ({'C', 'A'} (Tie), B)     ()      3
                     (B,)     ()      3
                (C, B, A)     ()      2


Score ballots are flexible enough to allow any non-zero score, including
negative scores. Scores of 0 are dropped from the dictionary. Note that
not all election methods support negative scoring, but these elections
in ``VoteKit`` validate your ballots and will raise a ``TypeError`` if
an invalid score is passed.

.. code:: ipython3

    score_ballot = Ballot(scores = {"A":-1, "B": 3.14159, "C":0}, weight = 3)
    print(score_ballot)


.. parsed-literal::

    Scores
    A: -1.00
    B: 3.14
    Weight: 3
    


Rating Election
---------------

In a Rating election, to fill :math:`m` seats, voters score each
candidate independently from :math:`0-L`, where :math:`L` is some
user-specified limit. The :math:`m` winners are those with the highest
total score.

.. code:: ipython3

    from votekit.elections import Rating
    
    score_profile = PreferenceProfile(ballots = [Ballot(scores = {"A":4, "B": 3, "C":4}, weight = 3),
                                      Ballot(scores = {"A":2, "B": 3, "C":4}, weight = 2),
                                      Ballot(scores = {"A":1, "B": 5, "C":4}, weight = 5),
                                      Ballot(scores = {"A":0, "B": 2, "C":0}, weight = 3)])
    
    # elect 1 seat, each voter can rate candidates up to 5 points independently
    election = Rating(score_profile, m = 1, L = 5)
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

    {'B': Fraction(46, 1), 'C': Fraction(40, 1), 'A': Fraction(21, 1)}


Now let’s see that the Rating election validates our profile before
running the election. All of these code blocks should raise
``TypeError``\ s.

.. code:: ipython3

    ranking_profile = PreferenceProfile(ballots = [Ballot(ranking= [{"A"}, {"B"}, {"C"}])])
    
    # should raise a TypeError since this profile has no scores
    election = Rating(ranking_profile, m = 1, L = 5)


::


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    Cell In[11], line 4
          1 ranking_profile = PreferenceProfile(ballots = [Ballot(ranking= [{"A"}, {"B"}, {"C"}])])
          3 # should raise a TypeError since this profile has no scores
    ----> 4 election = Rating(ranking_profile, m = 1, L = 5)


    File ~/PycharmProjects/VoteKit/src/votekit/elections/election_types/scores/rating.py:153, in Rating.__init__(self, profile, m, L, tiebreak)
        146 def __init__(
        147     self,
        148     profile: PreferenceProfile,
       (...)
        151     tiebreak: Optional[str] = None,
        152 ):
    --> 153     super().__init__(profile, m=m, L=L, tiebreak=tiebreak)


    File ~/PycharmProjects/VoteKit/src/votekit/elections/election_types/scores/rating.py:46, in GeneralRating.__init__(self, profile, m, L, k, tiebreak)
         44 self.k = k
         45 self.tiebreak = tiebreak
    ---> 46 self._validate_profile(profile)
         47 super().__init__(
         48     profile, score_function=score_profile_from_ballot_scores, sort_high_low=True
         49 )


    File ~/PycharmProjects/VoteKit/src/votekit/elections/election_types/scores/rating.py:63, in GeneralRating._validate_profile(self, profile)
         61 for b in profile.ballots:
         62     if not b.scores:
    ---> 63         raise TypeError("All ballots must have score dictionary.")
         64     elif any(score > self.L for score in b.scores.values()):
         65         raise TypeError(
         66             f"Ballot {b} violates score limit {self.L} per candidate."
         67         )


    TypeError: All ballots must have score dictionary.


.. code:: ipython3

    negative_profile = PreferenceProfile(ballots = [Ballot(scores = {"A":-1, "B": 3.14159, "C":0})])
    
    # should raise a TypeError since this profile has negative score
    election = Rating(negative_profile, m = 1, L = 5)


::


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    Cell In[13], line 4
          1 negative_profile = PreferenceProfile(ballots = [Ballot(scores = {"A":-1, "B": 3.14159, "C":0})])
          3 # should raise a TypeError since this profile has negative
    ----> 4 election = Rating(negative_profile, m = 1, L = 5)


    File ~/PycharmProjects/VoteKit/src/votekit/elections/election_types/scores/rating.py:153, in Rating.__init__(self, profile, m, L, tiebreak)
        146 def __init__(
        147     self,
        148     profile: PreferenceProfile,
       (...)
        151     tiebreak: Optional[str] = None,
        152 ):
    --> 153     super().__init__(profile, m=m, L=L, tiebreak=tiebreak)


    File ~/PycharmProjects/VoteKit/src/votekit/elections/election_types/scores/rating.py:46, in GeneralRating.__init__(self, profile, m, L, k, tiebreak)
         44 self.k = k
         45 self.tiebreak = tiebreak
    ---> 46 self._validate_profile(profile)
         47 super().__init__(
         48     profile, score_function=score_profile_from_ballot_scores, sort_high_low=True
         49 )


    File ~/PycharmProjects/VoteKit/src/votekit/elections/election_types/scores/rating.py:69, in GeneralRating._validate_profile(self, profile)
         65     raise TypeError(
         66         f"Ballot {b} violates score limit {self.L} per candidate."
         67     )
         68 elif any(score < 0 for score in b.scores.values()):
    ---> 69     raise TypeError(f"Ballot {b} must have non-negative scores.")
         71 if self.k:
         72     if sum(b.scores.values()) > self.k:


    TypeError: Ballot Scores
    A: -1.00
    B: 3.14
    Weight: 1
     must have non-negative scores.


.. code:: ipython3

    over_L_profile = PreferenceProfile(ballots = [Ballot(scores = {"A":0, "B": 10, "C":1})])
    
    # should raise a TypeError since this profile has score over 5
    election = Rating(over_L_profile, m = 1, L = 5)


::


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    Cell In[14], line 4
          1 over_L_profile = PreferenceProfile(ballots = [Ballot(scores = {"A":0, "B": 10, "C":1})])
          3 # should raise a TypeError since this profile has score over 5
    ----> 4 election = Rating(over_L_profile, m = 1, L = 5)


    File ~/PycharmProjects/VoteKit/src/votekit/elections/election_types/scores/rating.py:153, in Rating.__init__(self, profile, m, L, tiebreak)
        146 def __init__(
        147     self,
        148     profile: PreferenceProfile,
       (...)
        151     tiebreak: Optional[str] = None,
        152 ):
    --> 153     super().__init__(profile, m=m, L=L, tiebreak=tiebreak)


    File ~/PycharmProjects/VoteKit/src/votekit/elections/election_types/scores/rating.py:46, in GeneralRating.__init__(self, profile, m, L, k, tiebreak)
         44 self.k = k
         45 self.tiebreak = tiebreak
    ---> 46 self._validate_profile(profile)
         47 super().__init__(
         48     profile, score_function=score_profile_from_ballot_scores, sort_high_low=True
         49 )


    File ~/PycharmProjects/VoteKit/src/votekit/elections/election_types/scores/rating.py:65, in GeneralRating._validate_profile(self, profile)
         63     raise TypeError("All ballots must have score dictionary.")
         64 elif any(score > self.L for score in b.scores.values()):
    ---> 65     raise TypeError(
         66         f"Ballot {b} violates score limit {self.L} per candidate."
         67     )
         68 elif any(score < 0 for score in b.scores.values()):
         69     raise TypeError(f"Ballot {b} must have non-negative scores.")


    TypeError: Ballot Scores
    B: 10.00
    C: 1.00
    Weight: 1
     violates score limit 5 per candidate.


Cumulative election
-------------------

In a Cumulative election, voters can score each candidate as in a Rating
election, but have a total budget of :math:`m` points, where
:math:`m` is the number of seats to be filled. This means candidates
cannot be scored independently, the total must sum to no more than
:math:`m`.

Winners are those with highest total score. Giving a candidate multiple
points is known as “plumping” the vote.

.. code:: ipython3

    from votekit.elections import Cumulative
    
    score_profile = PreferenceProfile(ballots = [Ballot(scores = {"A":2, "B": 0, "C":0}, weight = 3),
                                      Ballot(scores = {"A":1, "B": 1, "C":0}, weight = 2),
                                      Ballot(scores = {"A":0, "B": 0, "C":2}, weight = 5),
                                      Ballot(scores = {"A":0, "B": 2, "C":0}, weight = 4)])
    
    # elect 2 seat, each voter can rate candidates up to 2 points total
    election = Cumulative(score_profile, m = 2)
    print(election)
    print(election.get_ranking())
    print(election.election_states[0].scores)


.. parsed-literal::

          Status  Round
    B    Elected      1
    C    Elected      1
    A  Remaining      1
    (frozenset({'B', 'C'}), frozenset({'A'}))
    {'B': Fraction(10, 1), 'C': Fraction(10, 1), 'A': Fraction(8, 1)}


Here, B and C tied for 10 points and are thus elected in the same set.

Again, the Cumulative class does validation for us.

.. code:: ipython3

    over_m_profile = PreferenceProfile(ballots = [Ballot(scores = {"A":0, "B": 2, "C":1})])
    
    # should raise a TypeError since this profile has total score over 2
    election = Cumulative(over_m_profile, m = 2)


::


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    Cell In[22], line 4
          1 over_m_profile = PreferenceProfile(ballots = [Ballot(scores = {"A":0, "B": 2, "C":1})])
          3 # should raise a TypeError since this profile has total score over 2
    ----> 4 election = Cumulative(over_m_profile, m = 2)


    File ~/PycharmProjects/VoteKit/src/votekit/elections/election_types/scores/rating.py:197, in Cumulative.__init__(self, profile, m, tiebreak)
        194 def __init__(
        195     self, profile: PreferenceProfile, m: int = 1, tiebreak: Optional[str] = None
        196 ):
    --> 197     super().__init__(profile, m=m, k=m, tiebreak=tiebreak)


    File ~/PycharmProjects/VoteKit/src/votekit/elections/election_types/scores/rating.py:179, in Limited.__init__(self, profile, m, k, tiebreak)
        177     raise ValueError("k must be less than or equal to m.")
        178 # if budget is k, limit per candidate is k
    --> 179 super().__init__(profile, m=m, L=k, k=k, tiebreak=tiebreak)


    File ~/PycharmProjects/VoteKit/src/votekit/elections/election_types/scores/rating.py:46, in GeneralRating.__init__(self, profile, m, L, k, tiebreak)
         44 self.k = k
         45 self.tiebreak = tiebreak
    ---> 46 self._validate_profile(profile)
         47 super().__init__(
         48     profile, score_function=score_profile_from_ballot_scores, sort_high_low=True
         49 )


    File ~/PycharmProjects/VoteKit/src/votekit/elections/election_types/scores/rating.py:73, in GeneralRating._validate_profile(self, profile)
         71 if self.k:
         72     if sum(b.scores.values()) > self.k:
    ---> 73         raise TypeError(f"Ballot {b} violates total score budget {self.k}.")


    TypeError: Ballot Scores
    B: 2.00
    C: 1.00
    Weight: 1
     violates total score budget 2.


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
    bloc_voter_prop = {"all_voters": 1}
    slate_to_candidates= {"all_voters": ["A", "B", "C"]}
    
    # the preference interval (80,15,5)
    pref_intervals_by_bloc = {"all_voters":  
                              {"all_voters": PreferenceInterval({"A": .80,  "B": .15,  "C": .05})}
                              }
    
    cohesion_parameters = {"all_voters": {"all_voters": 1}}
    
    # the num_votes parameter says how many total points the voter is given
    # for a cumulative election, this is m, the number of seats
    # in a limited election, this could be less than m
    cumu = bg.name_Cumulative(pref_intervals_by_bloc = pref_intervals_by_bloc,
                         bloc_voter_prop = bloc_voter_prop,
                         slate_to_candidates = slate_to_candidates,
                         cohesion_parameters=cohesion_parameters,
                          num_votes=m)
    
    profile = cumu.generate_profile(number_of_ballots = 100)
    print(profile)


.. parsed-literal::

    Ranking           Scores Weight
         ()        (A:2.00,)     63
         () (A:1.00, B:1.00)     28
         () (A:1.00, C:1.00)      5
         () (B:1.00, C:1.00)      2
         ()        (B:2.00,)      2


Verify that the ballots make sense given the interval. A should receive
the most votes.

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

