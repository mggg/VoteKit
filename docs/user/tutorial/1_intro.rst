Getting started with VoteKit
============================

VoteKit is a Python package developed by MGGG in order to support
computational social choice researchers as well as practitioners and
advocates interested in so-called “alternative” systems of election.
This first section will help us hit the ground running by introducing
some of our key vocabulary and introducing some Python syntax along the
way. By the end of the section, you should know about ballots,
preference profiles, preference intervals, blocs/slates, and election
systems.

Ballots
-------

The first order of business is ballots. In the context of ranked choice
voting, a ballot records a voter’s preferences as a linear ordering of
the candidates. If a voter casts the ballot :math:`A>B>C`, this means
they prefer candidate :math:`A` to :math:`B` and candidate :math:`B` to
:math:`C`. (We often use the :math:`>` symbol to indicate higher
preference.) Let’s see how ballots are stored in VoteKit. First we
import the necessary modules.

.. code:: ipython3

    from votekit.ballot import Ballot
    from fractions import Fraction
    
    ballot = Ballot(ranking = [{"A"}, {"B"}, {"C"}], weight = Fraction(3,2))
    print(ballot)


.. parsed-literal::

    Ballot
    1.) A, 
    2.) B, 
    3.) C, 
    Weight: 3/2
    


Here, we have created one ballot. The ballot stored the ranking
:math:`A>B>C`. The weight attribute indicates “how many” of this ballot
were cast. It defaults to 1, but we have put in 3/2. A fractional weight
will be useful in single transferable vote (STV) elections!

**Try it yourself**
~~~~~~~~~~~~~~~~~~~

   Create three new ballots, one with weight 47, one with weight 22/7,
   and one with your own weight idea.

Let’s dig a bit more into how the ranking is stored. It is a list of
sets, where the first set in the list indicates which candidates were
ranked first, the second set is who was ranked second, etc. In the first
example, we stored a **full linear ranking**. There was only one
candidate listed in each position, and every candidate was listed.

Full linear rankings are not the only possible ballots. Real-world
voters frequently list multiple candidates in the same position (even if
that is against the rules of the election). As far as we know, this is
always treated by localities running ranked choice elections as a voter
error, called an **overvote**.

Voters also leave some candidates out. In an extreme case, when a voter
only lists one candidate, we call this a **bullet vote**. These are
fairly common in ranked elections. A position with no candidates listed
is sometimes called an **undervote**.

We might prefer for localities running ranked choice elections to be
smart about the voter intent to communicate a tied preference – and we
can do that in VoteKit. But we’ll get to running elections later.

.. code:: ipython3

    ballot = Ballot(ranking = [{"A", "D"}, {"B", "B", "B"}, {"C", "E", "F"}])
    print("A ballot with overvotes:", ballot)


.. parsed-literal::

    A ballot with overvotes: Ballot
    1.) D, A, (tie)
    2.) B, 
    3.) E, C, F, (tie)
    Weight: 1
    


This ballot says that candidates :math:`D` and :math:`A` were ranked
first, :math:`B` second, and :math:`E,C,F` all in third.

.. code:: ipython3

    ballot = Ballot(ranking = [{"B"}])
    print("A bullet vote:")
    print(ballot)


.. parsed-literal::

    A bullet vote:
    Ballot
    1.) B, 
    Weight: 1
    


This ballot is a bullet vote; only candidate :math:`B` is listed in
first.

**Automatic cleaning vs specified cleaning**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

What we really mean to illustrate above is that the Ballot class has no
understanding of the rules of your election. It is flexible enough to
allow all sorts of rankings, even ones that are not valid.

Since the ranking is a list of *sets*, the only default cleaning that
occurs in ballots in VoteKit is that the candidates listed in a
particular position will be deduplicated. In the code above, the first
ballot should only print with one candidate named “B” in position two.

There are many other kinds of cleaning functions, but you have to choose
to apply those yourself. This is really crucial to know; lots of
elections will behave strangely if you do not have the correct ballot
types as input, but it is up to you to clean them to the level needed
for your method of election.

Preference Profiles
-------------------

When we want to aggregate a collection of ballots cast by voters, we use
the ``PreferenceProfile`` object. It stores all of the ballots, allows
us to visualize them, and comes with some handy features.

First we display the simple profile, which just repeats the weights as
they were inputted.

.. code:: ipython3

    from votekit.pref_profile import PreferenceProfile
    
    candidates = ["A", "B", "C"]
    
    # let's assume that the ballots come from voters, 
    # so they all have integer weight for now
    ballots = [Ballot(ranking = [{"A"}, {"B"}, {"C"}],weight=3),
               Ballot(ranking = [{"B"}, {"A"}, {"C"}]),
               Ballot(ranking = [{"C"}, {"B"}, {"A"}]),
               Ballot(ranking = [{"A"}, {"B"}, {"C"}]),
               Ballot(ranking = [{"A"}, {"B"}, {"C"}]),
               Ballot(ranking = [{"B"}, {"A"}, {"C"}])]
    
    # we give the profile a list of ballots and a list of candidates
    profile = PreferenceProfile(ballots = ballots, candidates = candidates)
    
    print(profile)


.. parsed-literal::

      Ballots Weight
    (A, B, C)      3
    (B, A, C)      1
    (C, B, A)      1
    (A, B, C)      1
    (A, B, C)      1
    (B, A, C)      1


The ``PreferenceProfile`` class takes a list of ``Ballot`` objects and a
list of candidates. The candidate names must be distinct, and it will
raise an error if not. Providing the list of candidates is actually
optional, and it has no impact on the Profile object. However, later
when we move on to ballot generation, the list of candidates will be
important.

Notice that printing the profile did not automatically combine like
ballots into a single line. But there’s an easy way to get the condensed
profile, as follows.

.. code:: ipython3

    condensed_profile  = profile.condense_ballots()
    print(condensed_profile)


.. parsed-literal::

      Ballots Weight
    (A, B, C)      5
    (B, A, C)      2
    (C, B, A)      1


In these examples, the profiles are very short, so we can print the
entire profile. If there were more ballots (either uncondensed or
different rankings), we’d see the message “PreferenceProfile too long,
only showing 15 out of XX rows.”

.. code:: ipython3

    ballots = [Ballot(ranking = [{"A"}, {"B"}, {"C"}]),
               Ballot(ranking = [{"B"}, {"A"}, {"C"}]),
               Ballot(ranking = [{"C"}, {"B"}, {"A"}]),
               Ballot(ranking = [{"A"}]),
               Ballot(ranking = [{"A"}, {"B"}, {"C"}]),
               Ballot(ranking = [{"B"}, {"A"}])]
    
    profile = PreferenceProfile(ballots = ballots*6,
                                candidates = candidates)
    
    print(profile)


.. parsed-literal::

    PreferenceProfile too long, only showing 15 out of 36 rows.
      Ballots Weight
    (A, B, C)      1
    (B, A, C)      1
    (C, B, A)      1
         (A,)      1
    (A, B, C)      1
       (B, A)      1
    (A, B, C)      1
    (B, A, C)      1
    (C, B, A)      1
         (A,)      1
    (A, B, C)      1
       (B, A)      1
    (A, B, C)      1
    (B, A, C)      1
    (C, B, A)      1


To see more of the ballots, we can use the ``head`` and ``tail`` methods
in the ``PreferenceProfile`` class. These display a user-specified
number of ballots. By default, it lists them in the order inputted. We
can ask ``head`` and ``tail`` to display them in order by weight by
using the ``sort_by_weight`` parameter and setting it to ``True``.

.. code:: ipython3

    # this will print the top 8 in order of input
    print(profile.head(8))
    print()
    
    # and the bottom 8
    print(profile.tail(8))
    print()
    
    # and the entry indexed 10, which includes the percent of the profile 
    # this ballot accounts for
    print(profile.df.iloc[10])
    print()
    
    # condense and sort by by weight
    condensed_profile = profile.condense_ballots()
    print(condensed_profile.head(8,sort_by_weight=True))



.. parsed-literal::

         Ballots Weight
    0  (A, B, C)      1
    1  (B, A, C)      1
    2  (C, B, A)      1
    3       (A,)      1
    4  (A, B, C)      1
    5     (B, A)      1
    6  (A, B, C)      1
    7  (B, A, C)      1
    
          Ballots Weight
    35  (A, B, C)      1
    34  (C, B, A)      1
    33       (A,)      1
    32  (A, B, C)      1
    31     (B, A)      1
    30  (A, B, C)      1
    29  (B, A, C)      1
    28  (B, A, C)      1
    
    Ballots    (A, B, C)
    Weight             1
    Percent        2.78%
    Name: 10, dtype: object
    
         Ballots Weight
    0  (A, B, C)     12
    1  (B, A, C)      6
    2  (C, B, A)      6
    3       (A,)      6
    4     (B, A)      6


A few other useful methods are listed here. Use ``profile.METHODNAME()``
for each one.

-  ``get_candidates()`` returns the list of candidates who received any
   votes.

-  ``get_ballots()`` returns the list of ballots (useful if you want to
   extract the ballots as a list to write custom code, say).

-  ``num_ballots()`` returns the number of ballots, which is the sum of
   the weights.

-  ``to_dict(standardize = False)`` returns the profile as dictionary
   whose keys are the rankings that received votes and whose values are
   the weights (condensed). Comes with an optional ``standardize``
   argument which divides the weights by the total weight.

-  ``to_csv(fpath = "name_of_file.csv")`` saves the profile as a csv
   (useful if you want to replicate runs of an experiment).

**Try it yourself**
~~~~~~~~~~~~~~~~~~~

   Try using all five of the above methods, with or without condensing
   the ballots. Try switching the ``standardize`` parameter in
   ``to_dict`` from False to True, and change the ``fpath`` parameter in
   ``to_csv`` to a file name that makes sense.

Preference Intervals
--------------------

There are a few ways to input ballots into VoteKit. You can type them
all by hand as we did above, you can read them in from real-world vote
records, or you can generate them within VoteKit. While we will dive a
lot deeper into reading and generating in future sections, it is
worthwhile to introduce some of the vocabulary surrounding generative
models here.

Most of our generative models rely on a **preference interval**. A
preference interval stores information about the relative strengths of a
voter’s priorities for candidates. We visualize this, unsurprisingly, as
an interval. We take the interval :math:`[0,1]` and divide it into
pieces, where we’ve taken all the preference weights and scaled so they
add to 1.

For example,

::

   {"A": 0.7, "B": 0.2, "C": 0.1}

is a dictionary that represents an ordered preference interval where A
is preferred to B by a ratio of 7/2, etc.

Later, the ballot generator models will pull from these preferences to
create a ballot for each voter.

It should be remarked that there is a difference, at least to VoteKit,
between the intervals

::

   {"A": 0.7, "B": 0.3, "C": 0} and
   {"A": 0.7, "B": 0.3}

While both say there is no preference for candidate C, if the latter
interval is fed into VoteKit, that third candidate will never appear on
a generated ballot. If we feed it the former interval, the third
candidate will appear at the bottom of the ballot.

.. figure:: ../../_static/assets/preference_interval.png
   :alt: png



One of the generative models is called the **slate-Plackett-Luce
model**, or s-PL. In s-PL, voters fill in their ballot from the top
position to the bottom by choosing from the available candidates in
proportion to their preference weights. We call this the impulsive voter
model.

You can read more about s-PL in our social choice documentation, but for
now let’s use it to explore how intervals work. We will assume there is
only one bloc of voters. This makes the syntax look a little strange,
but bear with us.

.. code:: ipython3

    import votekit.ballot_generator as bg
    from votekit import PreferenceInterval
    
    # the sPL model assumes there are blocs of voters, 
    # but we can just say that there is only one bloc
    bloc_voter_prop = {"all_voters": 1}
    slate_to_candidates= {"all_voters": ["A", "B", "C"]}
    
    # the preference interval (80,15,5)
    pref_intervals_by_bloc = {"all_voters":  
                              {"all_voters": PreferenceInterval({"A": .80,  "B":.15,  "C":.05})}
                              }
    
    # the sPL model needs an estimate of cohesion between blocs, 
    # but there is only one bloc here
    cohesion_parameters = {"all_voters": {"all_voters": 1}}
    
    pl = bg.slate_PlackettLuce(pref_intervals_by_bloc = pref_intervals_by_bloc,
                         bloc_voter_prop = bloc_voter_prop,
                         slate_to_candidates = slate_to_candidates,
                         cohesion_parameters=cohesion_parameters)
    
    profile = pl.generate_profile(number_of_ballots = 100)
    print(profile)


.. parsed-literal::

      Ballots Weight
    (A, B, C)     61
    (A, C, B)     17
    (B, A, C)     17
    (C, A, B)      4
    (B, C, A)      1


Re-run the above block several times to see that the elections will come
out different! The s-PL model is random, meaning we won’t always get the
same profile when we run ``generate_profile`` (although we are planning
to implement an explicit ``random seed`` option so that you can
replicate runs). You probably won’t get the same output as what is
stored in this tutorial either. That’s okay! Check that most ballots
rank :math:`A` first, which is expected because they had the largest
portion of the preference interval. Likewise, :math:`C` is least
popular.

Blocs
-----

A **bloc** of voters is a group of voters who have similar voting
behavior, generally preferring their **slate** of candidates to the
slates associated to other blocs. In VoteKit, we model this by assuming
voters within a bloc have the same preference interval. Let’s look at an
example where there are two blocs called Alpha and Xenon, each with a
two-candidate slate (:math:`A,B` and :math:`X,Y`, respectively).

By introducing blocs, we also need to discuss cohesion parameters. In
realistic polarized elections, we might be able to identify two groups
with different voting tendencies, but real voting blocs are not
perfectly monolithic—some voters will stick with their slate, but many
others might have a tendency to “cross over” to the other slate
sometimes in constructing their ballot.

The precise meaning of these vary by model, but broadly speaking,
**cohesion parameters** measure the strength with which voters stick to
their bloc.

.. code:: ipython3

    slate_to_candidates= {"Alpha": ["A", "B"],
                          "Xenon": ["X", "Y"]}
    
    # note that we include candidates with 0 support, 
    # and that our preference intervals will automatically rescale to sum to 1
    
    pref_intervals_by_bloc = {"Alpha": {"Alpha": PreferenceInterval({"A": .8, "B":.15}),
                                        "Xenon": PreferenceInterval({"X":0, "Y": .05})},
    
                             "Xenon": {"Alpha": PreferenceInterval({"A": .05, "B":.05}),
                                       "Xenon": PreferenceInterval({"X":.45, "Y": .45})}}
    
    
    bloc_voter_prop = {"Alpha": .8, "Xenon": .2}
    
    # assume that each bloc is 90% cohesive
    # we'll discuss exactly what that means later
    cohesion_parameters = {"Alpha": {"Alpha": .9, "Xenon": .1},
                           "Xenon": {"Xenon": .9, "Alpha": .1}}
    
    pl = bg.slate_PlackettLuce(pref_intervals_by_bloc = pref_intervals_by_bloc,
                         bloc_voter_prop = bloc_voter_prop,
                         slate_to_candidates = slate_to_candidates,
                         cohesion_parameters=cohesion_parameters)
    
    # the by_bloc parameter allows us to see which ballots came from which blocs of voters
    profile_dict, agg_profile = pl.generate_profile(number_of_ballots = 10000, by_bloc=True)
    print("The ballots from Alpha voters\n", profile_dict["Alpha"])
    
    print("The ballots from Xenon voters\n", profile_dict["Xenon"])
    
    print("Aggregated ballots\n", agg_profile)


.. parsed-literal::

    The ballots from Alpha voters
          Ballots Weight
    (A, B, Y, X)   5527
    (B, A, Y, X)    967
    (Y, A, B, X)    673
    (A, Y, B, X)    605
    (Y, B, A, X)    123
    (B, Y, A, X)    105
    The ballots from Xenon voters
     PreferenceProfile too long, only showing 15 out of 24 rows.
         Ballots Weight
    (Y, X, A, B)    415
    (X, Y, B, A)    405
    (Y, X, B, A)    401
    (X, Y, A, B)    397
    (B, Y, X, A)     50
    (A, X, Y, B)     45
    (B, X, Y, A)     44
    (Y, B, X, A)     41
    (X, A, Y, B)     40
    (Y, A, X, B)     36
    (X, B, Y, A)     35
    (A, Y, X, B)     33
    (A, B, X, Y)      9
    (B, Y, A, X)      9
    (A, X, B, Y)      7
    Aggregated ballots
     PreferenceProfile too long, only showing 15 out of 30 rows.
         Ballots Weight
    (A, B, Y, X)   5527
    (B, A, Y, X)    967
    (Y, A, B, X)    673
    (A, Y, B, X)    605
    (Y, X, A, B)    415
    (X, Y, B, A)    405
    (Y, X, B, A)    401
    (X, Y, A, B)    397
    (Y, B, A, X)    123
    (B, Y, A, X)    105
    (B, Y, X, A)     50
    (A, X, Y, B)     45
    (B, X, Y, A)     44
    (Y, B, X, A)     41
    (X, A, Y, B)     40


Scan this to be sure it is reasonable, recalling that our intervals say
that the Alpha voters prefer :math:`A` to :math:`B`, while :math:`X` has
no support in that bloc. Xenon voters like :math:`X` and :math:`Y`
equally, and then like :math:`A` and :math:`B` equally (although much
less than their own slate). There should be a lot more Alpha-style
voters than Xenon-style voters.

Elections
---------

Finally, we are ready to run an election. It is important to distinguish
between *preference profiles*, which are a collection of ballots, and
*elections*, which are the method by which those ballots are converted
to an outcome (candidates elected to seats). We will explore all sorts
of election types in later notebooks. For now, let’s use a plurality
election on a small set of ballots so we can verify that it behaves as
it should.

.. code:: ipython3

    from votekit.elections import Plurality
    
    ballots = [Ballot(ranking = [{"A"}, {"B"}, {"C"}]),
               Ballot(ranking = [{"B"}, {"A"}, {"C"}]),
               Ballot(ranking = [{"C"}, {"B"}, {"A"}]),
               Ballot(ranking = [{"A"}, {"B"}, {"C"}]),
               Ballot(ranking = [{"A"}, {"B"}, {"C"}]),
               Ballot(ranking = [{"B"}, {"A"}, {"C"}])]
    
    profile = PreferenceProfile(ballots = ballots*6,
                                candidates = candidates)
    
    profile = profile.condense_ballots()
    
    print(profile)
    
    election = Plurality(profile = profile,
                         seats = 1)
    
    print(election.run_election())


.. parsed-literal::

      Ballots Weight
    (A, B, C)     18
    (B, A, C)     12
    (C, B, A)      6
    Current Round: 1
    Candidate     Status  Round
            A    Elected      1
            B Eliminated      1
            C Eliminated      1


If everything worked as intended, you should see that :math:`A` was
elected, while :math:`B,C` were eliminated. There is only one round, as
plurality elections are single step.

You can also run a plurality election with more seats than one; it just
takes the :math:`m` candidates with the most first-place support as
winners.

For advanced users: if several candidates had the same level of
first-place support, the default tiebreaker in VoteKit is just uniformly
random. Please note: when ``run_election`` is run for the first time,
the result is cached. Any subsequent call will use the same will return
the same result, randomness and all. If you want to re-run with
different randomness, be sure the call to ``run_election`` is in the
same code block that the election is defined.

Conclusion
~~~~~~~~~~

The goal of this section was to introduce the vocabulary of VoteKit and
ranked choice voting. You should now know about ballots, preference
profiles, preference intervals, blocs/slates, and the distinction
between profiles and elections.

Extra Prompts
^^^^^^^^^^^^^

If you have finished this section and are looking to extend your
understanding, try the following prompts:

-  Write your own profile with four candidates named Trump, Rubio, Cruz,
   and Kasich, a preference interval of your choice, and with the bloc
   name set to “Repubs2016”. Generate 1000 ballots. Are they distributed
   how they should be given your preference interval?
-  Create a preference profile where candidates :math:`B,C` should be
   elected under a 2-seat plurality election. Run the election and
   confirm!
