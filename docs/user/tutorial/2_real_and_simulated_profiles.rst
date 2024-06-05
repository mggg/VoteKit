Real and Simulated Profiles
===========================

In the previous section, we discussed the ``Ballot`` class. It was very
flexible, allowing for many possible rankings (beyond full linear
rankings) to be stored. By the end of this section, you should be able
to read and clean ballots from real-world voting records, generate
ballots using a variety of models, and understand the candidate simplex.

Real-World Data
---------------

We will use the 2013 Minneapolis mayoral election as our first example.
This election had 35 candidates running for one seat, and used a
single-winner IRV election to choose the winner. Voters were allowed to
rank their top three candidates.

Let’s load in the **cast vote record** (CVR) from the election, which we
have stored in the VoteKit GitHub repository. Please download the file
and place it in your working directory (the same folder as your code).
The file can be found
`here <https://github.com/mggg/VoteKit/blob/main/src/votekit/data/mn_2013_cast_vote_record.csv>`__.

First we load the appropriate modules.

.. code:: ipython3

    from votekit.cvr_loaders import load_csv
    from votekit.elections import IRV
    from votekit.cleaning import remove_noncands

Next we’ll use the ``load_csv`` function to load the data. The data
should be a csv file, where each row is a ballot, and there is a column
for every position—i.e., a first-place vote column, a second-place vote
column, and so on.

The ``load_csv`` function has some optional parameters; you can specify
which columns of the csv contain ranking data (all of our columns did so
no need to specify), whether there is a weight column, some choice of
end-of-line delimiters (besides the standard, which is a carriage
return), and a voter ID column. It will return a ``PreferenceProfile``
object.

.. code:: ipython3

    minneapolis_profile = load_csv("mn_2013_cast_vote_record.csv")
    print(minneapolis_profile.head(6))


.. parsed-literal::

      Ballots                                          Weight
    0              (MARK ANDREW, undervote, undervote)  3864 
    1         (BETSY HODGES, MARK ANDREW, DON SAMUELS)  3309 
    2         (BETSY HODGES, DON SAMUELS, MARK ANDREW)  3031 
    3         (MARK ANDREW, BETSY HODGES, DON SAMUELS)  2502 
    4             (BETSY HODGES, undervote, undervote)  2212 
    5  (BETSY HODGES, DON SAMUELS, JACKIE CHERRYHOMES)  1831 


Note that the ``load_csv`` function automatically condenses the profile.

Let’s explore the profile using some of the tools we learned in the
previous notebook.

.. code:: ipython3

    print("The list of candidates is")
    print(minneapolis_profile.get_candidates())
    
    print(f"There are {len(minneapolis_profile.get_candidates())} candidates.")


.. parsed-literal::

    The list of candidates is
    ['JOHN LESLIE HARTWIG', 'ALICIA K. BENNETT', 'ABDUL M RAHAMAN "THE ROCK"', 'CAPTAIN JACK SPARROW', 'STEPHANIE WOODRUFF', 'JAMES EVERETT', 'JAMES "JIMMY" L. STROUD, JR.', 'DOUG MANN', 'CHRISTOPHER CLARK', 'TROY BENJEGERDES', 'JACKIE CHERRYHOMES', 'DON SAMUELS', 'KURTIS W. HANNA', 'overvote', 'MARK ANDREW', 'OLE SAVIOR', 'TONY LANE', 'JAYMIE KELLY', 'MIKE GOULD', 'CHRISTOPHER ROBIN ZIMMERMAN', 'GREGG A. IVERSON', 'DAN COHEN', 'CYD GORMAN', 'UWI', 'BILL KAHN', 'RAHN V. WORKCUFF', 'MERRILL ANDERSON', 'CAM WINTON', 'EDMUND BERNARD BRUYERE', 'BETSY HODGES', 'undervote', 'BOB FINE', 'JOHN CHARLES WILSON', 'JEFFREY ALAN WAGNER', 'JOSHUA REA', 'MARK V ANDERSON', 'NEAL BAXTER', 'BOB "AGAIN" CARNEY JR']
    There are 38 candidates.


Woah, that’s a little funky! There are candidates called ‘undervote’,
‘overvote’, and ‘UWI’. This cast vote record was already cleaned by the
City of Minneapolis, and they chose this way of parsing the ballots:
‘undervote’ indicates that the voter left a position unfilled, such as
by having no candidate listed in second place. The ‘overvote’ notation
arises when a voter puts two candidates in one position, like by putting
Hodges and Samuels both in first place. Unfortunately this way of
storing the profile means we have lost any knowledge of the voter intent
(which was probably to indicate equal preference). ‘UWI’ stands for
unregistered write-in.

This reminds us that it is really important to think carefully about how
we want to handle cleaning ballots, as some storage methods are
efficient but lossy. For now, let’s assume that we want to further
condense the ballots, discarding ‘undervote’, ‘overvote’, and ‘UWI’ as
candidates. The function ``remove_noncands`` will do this for us once we
specify which (non)candidates to remove. If a ballot was “A B
undervote”, it will become “A B”. If a ballot was “A UWI B” it will now
be “A B” as well. Many other cleaning options are reasonable.

.. code:: ipython3

    print("There were",len(minneapolis_profile.get_candidates()),"candidates\n")
    
    clean_profile = remove_noncands(minneapolis_profile, ["undervote", "overvote", "UWI"])
    print(clean_profile.get_candidates())
    
    print("\nThere are now",len(clean_profile.get_candidates()),"candidates")
    
    print(clean_profile.head(6, percents=True))


.. parsed-literal::

    There were 38 candidates
    
    ['NEAL BAXTER', 'JAYMIE KELLY', 'MIKE GOULD', 'CHRISTOPHER ROBIN ZIMMERMAN', 'GREGG A. IVERSON', 'DAN COHEN', 'JOHN LESLIE HARTWIG', 'ALICIA K. BENNETT', 'CYD GORMAN', 'BILL KAHN', 'RAHN V. WORKCUFF', 'MERRILL ANDERSON', 'CAPTAIN JACK SPARROW', 'CAM WINTON', 'STEPHANIE WOODRUFF', 'EDMUND BERNARD BRUYERE', 'JAMES EVERETT', 'BETSY HODGES', 'JAMES "JIMMY" L. STROUD, JR.', 'DOUG MANN', 'CHRISTOPHER CLARK', 'TROY BENJEGERDES', 'JACKIE CHERRYHOMES', 'BOB FINE', 'JOHN CHARLES WILSON', 'DON SAMUELS', 'JEFFREY ALAN WAGNER', 'KURTIS W. HANNA', 'JOSHUA REA', 'MARK ANDREW', 'OLE SAVIOR', 'MARK V ANDERSON', 'ABDUL M RAHAMAN "THE ROCK"', 'TONY LANE', 'BOB "AGAIN" CARNEY JR']
    
    There are now 35 candidates
      Ballots                                          Weight Percent
    0                                   (MARK ANDREW,)  3864   4.87% 
    1         (BETSY HODGES, MARK ANDREW, DON SAMUELS)  3309   4.17% 
    2         (BETSY HODGES, DON SAMUELS, MARK ANDREW)  3031   3.82% 
    3         (MARK ANDREW, BETSY HODGES, DON SAMUELS)  2502   3.15% 
    4                                  (BETSY HODGES,)  2212   2.79% 
    5  (BETSY HODGES, DON SAMUELS, JACKIE CHERRYHOMES)  1831   2.31% 


Things look a bit cleaner; all three of the non-candidate strings have
been removed. Note that the order of candidates is not very meaningful;
it’s just the order in which the names occurred in the input data. When
listing by weight, note how the top ballot changed from (Mark Andrew,
undervote, undervote) to just a bullet vote for Mark Andrew, which
occurred on almost 5 percent of ballots! Briefly, let’s run the same
kind of election type that was conducted in 2013 to verify we get the
same outcome as the city announced. The city used IRV elections (which
are equivalent to STV for one seat). Let’s check it out.

.. code:: ipython3

    # an IRV election for one seat
    minn_election = IRV(profile = clean_profile)
    minn_election.run_election()


.. parsed-literal::

    Current Round: 35




.. parsed-literal::

                       Candidate     Status  Round
                    BETSY HODGES    Elected     35
                     MARK ANDREW Eliminated     34
                     DON SAMUELS Eliminated     33
                      CAM WINTON Eliminated     32
              JACKIE CHERRYHOMES Eliminated     31
                        BOB FINE Eliminated     30
                       DAN COHEN Eliminated     29
              STEPHANIE WOODRUFF Eliminated     28
                 MARK V ANDERSON Eliminated     27
                       DOUG MANN Eliminated     26
                      OLE SAVIOR Eliminated     25
                   JAMES EVERETT Eliminated     24
               ALICIA K. BENNETT Eliminated     23
      ABDUL M RAHAMAN "THE ROCK" Eliminated     22
            CAPTAIN JACK SPARROW Eliminated     21
               CHRISTOPHER CLARK Eliminated     20
                       TONY LANE Eliminated     19
                    JAYMIE KELLY Eliminated     18
                      MIKE GOULD Eliminated     17
                 KURTIS W. HANNA Eliminated     16
     CHRISTOPHER ROBIN ZIMMERMAN Eliminated     15
             JEFFREY ALAN WAGNER Eliminated     14
                     NEAL BAXTER Eliminated     13
                TROY BENJEGERDES Eliminated     12
                GREGG A. IVERSON Eliminated     11
                MERRILL ANDERSON Eliminated     10
                      JOSHUA REA Eliminated      9
                       BILL KAHN Eliminated      8
             JOHN LESLIE HARTWIG Eliminated      7
          EDMUND BERNARD BRUYERE Eliminated      6
    JAMES "JIMMY" L. STROUD, JR. Eliminated      5
                RAHN V. WORKCUFF Eliminated      4
           BOB "AGAIN" CARNEY JR Eliminated      3
                      CYD GORMAN Eliminated      2
             JOHN CHARLES WILSON Eliminated      1



If you’re so moved, take a moment to `go
verify <https://en.wikipedia.org/wiki/2013_Minneapolis_mayoral_election>`__
that we got the same order of elimination and the same winning candidate
as in the official election.

Well that was simple! One takeaway: cleaning your data is a crucial
step, and how you clean your data depends on your own context. This is
why VoteKit provides helper functions to clean ballots, but it does not
automatically apply them.

Simulated voting with ballot generators
---------------------------------------

If we want to get a large sample of ballots without using real-world
data, we can use a variety of ballot generators included in VoteKit.

Bradley-Terry
~~~~~~~~~~~~~

The slate-Bradley-Terry model (s-BT) uses the same set of input
parameters as s-PL: ``slate_to_candidates``, ``bloc_voter_prop``,
``cohesion_parameters``, and ``pref_intervals_by_bloc``. We call s-BT
the deliberative voter model because part of the generation process
involves making all pairwise comparisons between candidates on the
ballot. A more detailed discussion can be found in our `social choice
documentation <../../social_choice_docs/scr.html#slate-bradley-terry>`__.

.. code:: ipython3

    import votekit.ballot_generator as bg
    from votekit import PreferenceInterval
    
    slate_to_candidates= {"Alpha": ["A", "B"],
                          "Xenon": ["X", "Y"]}
    
    # note that we include candidates with 0 support, and that our preference intervals
    # will automatically rescale to sum to 1
    
    pref_intervals_by_bloc = {"Alpha": {"Alpha": PreferenceInterval({"A": .8, "B":.15}),
                                        "Xenon": PreferenceInterval({"X":0, "Y": .05})},
    
                             "Xenon": {"Alpha": PreferenceInterval({"A": .05, "B":.05}),
                                       "Xenon": PreferenceInterval({"X":.45, "Y": .45})}}
    
    
    bloc_voter_prop = {"Alpha": .8, "Xenon": .2}
    
    # assume that each bloc is 90% cohesive
    cohesion_parameters = {"Alpha": {"Alpha": .9, "Xenon": .1},
                           "Xenon": {"Xenon": .9, "Alpha": .1}}
    
    bt = bg.slate_BradleyTerry(pref_intervals_by_bloc = pref_intervals_by_bloc,
                         bloc_voter_prop = bloc_voter_prop,
                         slate_to_candidates = slate_to_candidates,
                         cohesion_parameters=cohesion_parameters)
    
    profile = bt.generate_profile(number_of_ballots = 100)
    print(profile)


.. parsed-literal::

         Ballots Weight
    (A, B, Y, X)     57
    (B, A, Y, X)     16
    (Y, X, B, A)      9
    (X, Y, B, A)      5
    (A, Y, B, X)      4
    (X, Y, A, B)      3
    (Y, X, A, B)      3
    (Y, A, B, X)      2
    (B, Y, A, X)      1


.. admonition:: A note on s-BT 
    :class: note 

    The probability distribution that s-BT samples from is too cumbersome to compute for 
    more than 11 candidates. We have implemented a Markov chain Monte Carlo (MCMC)
    sampling method to account for this. Simply set
    ``deterministic = False`` in the ``generate_profile`` method to use the
    MCMC code. The sample size should be increased to ensure mixing of the chain.

.. code:: ipython3

    mcmc_profile = bt.generate_profile(number_of_ballots = 10000, deterministic=False)
    print(profile)


.. parsed-literal::

         Ballots Weight
    (A, B, Y, X)     57
    (B, A, Y, X)     16
    (Y, X, B, A)      9
    (X, Y, B, A)      5
    (A, Y, B, X)      4
    (X, Y, A, B)      3
    (Y, X, A, B)      3
    (Y, A, B, X)      2
    (B, Y, A, X)      1


Generating Preference Intervals from Hyperparameters
----------------------------------------------------

Now that we have seen a few ballot generators, we can introduce the
candidate simplex and the Dirichlet distribution.

We saw that you can initialize the Plackett-Luce model and the
Bradley-Terry model from a preference interval (or multiple ones if you
have different voting blocs). Recall, a preference interval stores a
voter’s preference for candidates as a vector of non-negative values
that sum to 1. Other models that rely on preference intervals include
the Alternating Crossover model (AC) and the Cambridge Sampler (CS).
There is a nice geometric representation of preference intervals via the
candidate simplex.

Candidate Simplex
~~~~~~~~~~~~~~~~~

Informally, the candidate simplex is a geometric representation of the
space of preference intervals. With two candidates, it is an interval;
with three candidates, it is a triangle; with four, a tetrahedron; and
so on getting harder to visualize as the dimension goes up.

This will be easiest to visualize with three candidates :math:`A,B,C`.
Then there is a one-to-one correspondence between positions in the
triangle and what are called **convex combinations** of the extreme
points. For instance, :math:`.8A+.15B+.05C` is a weighted average of
those points giving 80% of the weight to :math:`A`, 15% to :math:`B`,
and 5% to :math:`C`. The result is a point that is closest to :math:`A`,
as seen in the picture.

Those coefficients, which sum to 1, become the lengths of the
candidate’s sub-intervals. So this lets us see the simplex as the space
of all preference intervals.

.. figure:: ../../_static/assets/candidate_simplex.png
   :alt: png



Dirichlet Distribution
~~~~~~~~~~~~~~~~~~~~~~

**Dirichlet distributions** are a one-parameter family of probability
distributions on the simplex—this is used here to choose a preference
interval at random. We parameterize it with a value
:math:`\alpha \in (0,\infty)`. As :math:`\alpha\to \infty`, the support
of the distribution moves to the center of the simplex. This means we
are more likely to sample preference intervals that have roughly equal
support for all candidates, which will translate to all orderings being
equally likely. As :math:`\alpha\to 0`, the mass moves to the vertices.
This means we are more likely to choose a preference interval that has
strong support for a single candidate. In between is :math:`\alpha=1`,
where any region of the simplex is weighted in proportion to its area.
We think of this as the “all bets are off” setting – you might choose a
balanced preference, a concentrated preference, or something in between.

The value :math:`\alpha` is never allowed to equal 0 or :math:`\infty`
in Python, so VoteKit changes these to a very small number
(:math:`10^{-10}`) and a very large number :math:`(10^{20})`. We don’t
recommend using values that extreme. In previous studies, MGGG members
have taken :math:`\alpha = 1/2` to be “small” and :math:`\alpha = 2` to
be “big.”

.. figure:: ../../_static/assets/dirichlet_distribution.png
   :alt: png



It is easy to sample a ``PreferenceInterval`` from the Dirichlet
distribution. Rerun the code below several times to get a feel for how
these change with randomness.

.. code:: ipython3

    strong_pref_interval = PreferenceInterval.from_dirichlet(candidates=["A", "B", "C"], 
                                                             alpha=.1)
    print("Strong preference for one candidate", strong_pref_interval)
    
    abo_pref_interval = PreferenceInterval.from_dirichlet(candidates=["A", "B", "C"], 
                                                          alpha=1)
    print("All bets are off preference", abo_pref_interval)
    
    unif_pref_interval = PreferenceInterval.from_dirichlet(candidates=["A", "B", "C"], 
                                                           alpha=10)
    print("Uniform preference for all candidates", unif_pref_interval)


.. parsed-literal::

    Strong preference for one candidate {'A': 0.9407, 'B': 0.0592, 'C': 0.0001}
    All bets are off preference {'A': 0.5257, 'B': 0.0162, 'C': 0.4582}
    Uniform preference for all candidates {'A': 0.3208, 'B': 0.3846, 'C': 0.2946}


Let’s initialize the s-PL model from the Dirichlet distribution, using
that to build a preference interval rather than specifying the interval.
Each bloc will need two Dirichlet alpha values; one to describe their
own preference interval, and another to describe their preference for
the opposing candidates.

.. code:: ipython3

    bloc_voter_prop = {"X": .8, "Y": .2}
    
    # the values of .9 indicate that these blocs are highly polarized;
    # they prefer their own candidates much more than the opposing slate
    cohesion_parameters = {"X": {"X":.9, "Y":.1},
                            "Y": {"Y":.9, "X":.1}}
    
    alphas = {"X": {"X":2, "Y":1},
                        "Y": {"X":1, "Y":.5}}
    
    slate_to_candidates = {"X": ["X1", "X2"],
                            "Y": ["Y1", "Y2"]}
    
    # the from_params method allows us to sample from 
    # the Dirichlet distribution for our intervals
    pl = bg.slate_PlackettLuce.from_params(slate_to_candidates=slate_to_candidates,
              bloc_voter_prop=bloc_voter_prop,
              cohesion_parameters=cohesion_parameters,
              alphas=alphas)
    
    print("Preference interval for X bloc and X candidates")
    print(pl.pref_intervals_by_bloc["X"]["X"])
    print()
    print("Preference interval for X bloc and Y candidates")
    print(pl.pref_intervals_by_bloc["X"]["Y"])
    
    print()
    profile_dict, agg_profile = pl.generate_profile(number_of_ballots = 100, by_bloc = True)
    print(profile_dict["X"])


.. parsed-literal::

    Preference interval for X bloc and X candidates
    {'X1': 0.3711, 'X2': 0.6289}
    
    Preference interval for X bloc and Y candidates
    {'Y1': 0.1051, 'Y2': 0.8949}
    
             Ballots Weight
    (X2, X1, Y2, Y1)     39
    (X1, X2, Y2, Y1)     17
    (X2, X1, Y1, Y2)      4
    (Y2, X1, X2, Y1)      4
    (Y2, X2, X1, Y1)      4
    (X2, Y2, X1, Y1)      3
    (X1, Y2, X2, Y1)      3
    (X1, X2, Y1, Y2)      2
    (Y1, X2, Y2, X1)      1
    (Y2, X2, Y1, X1)      1
    (Y2, Y1, X2, X1)      1
    (X2, Y2, Y1, X1)      1


Let’s confirm that the intervals and ballots look reasonable. We have
:math:`\alpha_{XX} = 2` and :math:`\alpha_{XY} = 1`. This means that the
:math:`X` voters tend to be relatively indifferent among their own
candidates, but might adopt any candidate strength behavior for the
:math:`Y` slate.

**Try it yourself**
~~~~~~~~~~~~~~~~~~~

   Change the code above to check that the preference intervals and
   ballots for the :math:`Y` bloc look reasonable.

Cambridge Sampler
-----------------

We introduce one more method of generating ballots: the **Cambridge
Sampler** (CS). CS generates ranked ballots using historical election
data from Cambridge, MA (which has been continuously conducting ranked
choice elections since 1941). It is the only ballot generator we will
see today that is capable of producing incomplete ballots, including
bullet votes.

By default, CS uses five elections (2009-2017, odd years); with the help
of local organizers, we coded the candidates as White (W) or People of
Color (POC, or C for short). This is not necessarily the biggest factor
predicting people’s vote in Cambridge – housing policy is the biggie –
but it’s a good place to find realistic rankings, with candidates of two
types.

You also have the option of providing CS with your own historical
election data from which to generate ballots instead of using Cambridge
data.

.. code:: ipython3

    bloc_voter_prop = {"W": .8, "C": .2}
    
    # the values of .9 indicate that these blocs are highly polarized;
    # they prefer their own candidates much more than the opposing slate
    cohesion_parameters = {"W": {"W":.9, "C":.1},
                            "C": {"C":.9, "W":.1}}
    
    alphas = {"W": {"W":2, "C":1},
              "C": {"W":1, "C":.5}}
    
    slate_to_candidates = {"W": ["W1", "W2", "W3"],
                            "C": ["C1", "C2"]}
    
    cs = bg.CambridgeSampler.from_params(slate_to_candidates=slate_to_candidates,
              bloc_voter_prop=bloc_voter_prop,
              cohesion_parameters=cohesion_parameters,
              alphas=alphas)
    
    
    profile = cs.generate_profile(number_of_ballots= 1000)
    print(profile)


.. parsed-literal::

    PreferenceProfile too long, only showing 15 out of 277 rows.
                 Ballots Weight
                   (W2,)     24
            (W2, W1, W3)     21
            (W1, W2, W3)     18
    (W3, W2, W1, C1, C2)     17
                   (C1,)     17
    (W2, W3, W1, C1, C2)     15
                   (W1,)     15
                (C1, C2)     14
            (W2, W3, W1)     14
                   (W3,)     13
    (W2, C1, C2, W3, W1)     13
            (W2, C1, C2)     12
    (W2, W1, W3, C1, C2)     12
        (W2, W3, W1, C1)     12
        (W2, W1, W3, C1)     12


Note: the ballot type (as in, Ws and Cs) is strictly drawn from the
historical frequencies. The candidate IDs (as in W1 and W2 among the W
slate) are filled in by sampling without replacement from the preference
interval that you either provided or made from Dirichlet alphas. That is
the only role of the preference interval.

Conclusion
----------

There are many other models of ballot generation in VoteKit, both for
ranked choice ballots and points based ballots (think cumulative or
approval voting). See the `ballot
generator <../../api#Ballot_Generators>`__ section of the VoteKit
documentation for more.
