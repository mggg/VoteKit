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
`here <https://github.com/mggg/VoteKit/blob/main/notebooks/mn_2013_cast_vote_record.csv>`__.

First we load the appropriate modules.

.. code:: ipython3

    from votekit.cvr_loaders import load_csv
    from votekit.elections import IRV
    from votekit.cleaning import remove_and_condense

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
    print(minneapolis_profile)


.. parsed-literal::

    Profile contains rankings: True
    Maximum ranking length: 3
    Profile contains scores: False
    Candidates: ('ABDUL M RAHAMAN "THE ROCK"', 'DAN COHEN', 'JAMES EVERETT', 'MARK V ANDERSON', 'TROY BENJEGERDES', 'undervote', 'ALICIA K. BENNETT', 'BETSY HODGES', 'MARK ANDREW', 'MIKE GOULD', 'BILL KAHN', 'BOB FINE', 'CAM WINTON', 'DON SAMUELS', 'JACKIE CHERRYHOMES', 'JEFFREY ALAN WAGNER', 'JOHN LESLIE HARTWIG', 'KURTIS W. HANNA', 'JOSHUA REA', 'MERRILL ANDERSON', 'NEAL BAXTER', 'STEPHANIE WOODRUFF', 'UWI', 'BOB "AGAIN" CARNEY JR', 'TONY LANE', 'CAPTAIN JACK SPARROW', 'GREGG A. IVERSON', 'JAMES "JIMMY" L. STROUD, JR.', 'JAYMIE KELLY', 'CYD GORMAN', 'EDMUND BERNARD BRUYERE', 'DOUG MANN', 'CHRISTOPHER ROBIN ZIMMERMAN', 'RAHN V. WORKCUFF', 'JOHN CHARLES WILSON', 'OLE SAVIOR', 'overvote', 'CHRISTOPHER CLARK')
    Candidates who received votes: ('ABDUL M RAHAMAN "THE ROCK"', 'DAN COHEN', 'JAMES EVERETT', 'MARK V ANDERSON', 'TROY BENJEGERDES', 'undervote', 'ALICIA K. BENNETT', 'BETSY HODGES', 'MARK ANDREW', 'MIKE GOULD', 'BILL KAHN', 'BOB FINE', 'CAM WINTON', 'DON SAMUELS', 'JACKIE CHERRYHOMES', 'JEFFREY ALAN WAGNER', 'JOHN LESLIE HARTWIG', 'KURTIS W. HANNA', 'JOSHUA REA', 'MERRILL ANDERSON', 'NEAL BAXTER', 'STEPHANIE WOODRUFF', 'UWI', 'BOB "AGAIN" CARNEY JR', 'TONY LANE', 'CAPTAIN JACK SPARROW', 'GREGG A. IVERSON', 'JAMES "JIMMY" L. STROUD, JR.', 'JAYMIE KELLY', 'CYD GORMAN', 'EDMUND BERNARD BRUYERE', 'DOUG MANN', 'CHRISTOPHER ROBIN ZIMMERMAN', 'RAHN V. WORKCUFF', 'JOHN CHARLES WILSON', 'OLE SAVIOR', 'overvote', 'CHRISTOPHER CLARK')
    Total number of Ballot objects: 7084
    Total weight of Ballot objects: 80101.0
    


Note that the ``load_csv`` function automatically condenses the profile.

Let’s explore the profile using some of the tools we learned in the
previous notebook.

.. code:: ipython3

    print("The list of candidates is")
    
    for candidate in sorted(minneapolis_profile.candidates):
        print(f"\t{candidate}")
    
    print(f"There are {len(minneapolis_profile.candidates)} candidates.")


.. parsed-literal::

    The list of candidates is
    	ABDUL M RAHAMAN "THE ROCK"
    	ALICIA K. BENNETT
    	BETSY HODGES
    	BILL KAHN
    	BOB "AGAIN" CARNEY JR
    	BOB FINE
    	CAM WINTON
    	CAPTAIN JACK SPARROW
    	CHRISTOPHER CLARK
    	CHRISTOPHER ROBIN ZIMMERMAN
    	CYD GORMAN
    	DAN COHEN
    	DON SAMUELS
    	DOUG MANN
    	EDMUND BERNARD BRUYERE
    	GREGG A. IVERSON
    	JACKIE CHERRYHOMES
    	JAMES "JIMMY" L. STROUD, JR.
    	JAMES EVERETT
    	JAYMIE KELLY
    	JEFFREY ALAN WAGNER
    	JOHN CHARLES WILSON
    	JOHN LESLIE HARTWIG
    	JOSHUA REA
    	KURTIS W. HANNA
    	MARK ANDREW
    	MARK V ANDERSON
    	MERRILL ANDERSON
    	MIKE GOULD
    	NEAL BAXTER
    	OLE SAVIOR
    	RAHN V. WORKCUFF
    	STEPHANIE WOODRUFF
    	TONY LANE
    	TROY BENJEGERDES
    	UWI
    	overvote
    	undervote
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
candidates. The function ``remove_and_condense`` will do this for us
once we specify which (non)candidates to remove. If a ballot was “A B
undervote”, it will become “A B”. If a ballot was “A UWI B” it will now
be “A B” as well. Many other cleaning options are reasonable.

.. code:: ipython3

    print("There were", len(minneapolis_profile.candidates), "candidates\n")
    
    clean_profile = remove_and_condense(["undervote", "overvote", "UWI"], minneapolis_profile)
    print(clean_profile.candidates)
    
    print("\nThere are now", len(clean_profile.candidates), "candidates")
    
    print(clean_profile)


.. parsed-literal::

    There were 38 candidates
    
    ('JACKIE CHERRYHOMES', 'TROY BENJEGERDES', 'EDMUND BERNARD BRUYERE', 'DON SAMUELS', 'CAM WINTON', 'DOUG MANN', 'STEPHANIE WOODRUFF', 'JOHN CHARLES WILSON', 'DAN COHEN', 'JOSHUA REA', 'TONY LANE', 'BOB "AGAIN" CARNEY JR', 'NEAL BAXTER', 'ALICIA K. BENNETT', 'BETSY HODGES', 'CAPTAIN JACK SPARROW', 'JOHN LESLIE HARTWIG', 'JAYMIE KELLY', 'CHRISTOPHER ROBIN ZIMMERMAN', 'MERRILL ANDERSON', 'JAMES "JIMMY" L. STROUD, JR.', 'BILL KAHN', 'KURTIS W. HANNA', 'RAHN V. WORKCUFF', 'CYD GORMAN', 'JEFFREY ALAN WAGNER', 'GREGG A. IVERSON', 'MARK V ANDERSON', 'MIKE GOULD', 'ABDUL M RAHAMAN "THE ROCK"', 'CHRISTOPHER CLARK', 'OLE SAVIOR', 'JAMES EVERETT', 'MARK ANDREW', 'BOB FINE')
    
    There are now 35 candidates
    Profile has been cleaned
    Profile contains rankings: True
    Maximum ranking length: 3
    Profile contains scores: False
    Candidates: ('JACKIE CHERRYHOMES', 'TROY BENJEGERDES', 'EDMUND BERNARD BRUYERE', 'DON SAMUELS', 'CAM WINTON', 'DOUG MANN', 'STEPHANIE WOODRUFF', 'JOHN CHARLES WILSON', 'DAN COHEN', 'JOSHUA REA', 'TONY LANE', 'BOB "AGAIN" CARNEY JR', 'NEAL BAXTER', 'ALICIA K. BENNETT', 'BETSY HODGES', 'CAPTAIN JACK SPARROW', 'JOHN LESLIE HARTWIG', 'JAYMIE KELLY', 'CHRISTOPHER ROBIN ZIMMERMAN', 'MERRILL ANDERSON', 'JAMES "JIMMY" L. STROUD, JR.', 'BILL KAHN', 'KURTIS W. HANNA', 'RAHN V. WORKCUFF', 'CYD GORMAN', 'JEFFREY ALAN WAGNER', 'GREGG A. IVERSON', 'MARK V ANDERSON', 'MIKE GOULD', 'ABDUL M RAHAMAN "THE ROCK"', 'CHRISTOPHER CLARK', 'OLE SAVIOR', 'JAMES EVERETT', 'MARK ANDREW', 'BOB FINE')
    Candidates who received votes: ('ABDUL M RAHAMAN "THE ROCK"', 'DAN COHEN', 'JAMES EVERETT', 'MARK V ANDERSON', 'TROY BENJEGERDES', 'ALICIA K. BENNETT', 'BETSY HODGES', 'MARK ANDREW', 'MIKE GOULD', 'BILL KAHN', 'BOB FINE', 'CAM WINTON', 'DON SAMUELS', 'JACKIE CHERRYHOMES', 'JEFFREY ALAN WAGNER', 'JOHN LESLIE HARTWIG', 'KURTIS W. HANNA', 'JOSHUA REA', 'MERRILL ANDERSON', 'NEAL BAXTER', 'STEPHANIE WOODRUFF', 'BOB "AGAIN" CARNEY JR', 'TONY LANE', 'CAPTAIN JACK SPARROW', 'GREGG A. IVERSON', 'JAMES "JIMMY" L. STROUD, JR.', 'JAYMIE KELLY', 'CYD GORMAN', 'EDMUND BERNARD BRUYERE', 'DOUG MANN', 'CHRISTOPHER ROBIN ZIMMERMAN', 'RAHN V. WORKCUFF', 'JOHN CHARLES WILSON', 'OLE SAVIOR', 'CHRISTOPHER CLARK')
    Total number of Ballot objects: 7073
    Total weight of Ballot objects: 79378.0
    


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
    minn_election = IRV(profile=clean_profile)
    print(minn_election)


.. parsed-literal::

                                      Status  Round
    BETSY HODGES                     Elected     35
    MARK ANDREW                   Eliminated     34
    DON SAMUELS                   Eliminated     33
    CAM WINTON                    Eliminated     32
    JACKIE CHERRYHOMES            Eliminated     31
    BOB FINE                      Eliminated     30
    DAN COHEN                     Eliminated     29
    STEPHANIE WOODRUFF            Eliminated     28
    MARK V ANDERSON               Eliminated     27
    DOUG MANN                     Eliminated     26
    OLE SAVIOR                    Eliminated     25
    JAMES EVERETT                 Eliminated     24
    ALICIA K. BENNETT             Eliminated     23
    ABDUL M RAHAMAN "THE ROCK"    Eliminated     22
    CAPTAIN JACK SPARROW          Eliminated     21
    CHRISTOPHER CLARK             Eliminated     20
    TONY LANE                     Eliminated     19
    JAYMIE KELLY                  Eliminated     18
    MIKE GOULD                    Eliminated     17
    KURTIS W. HANNA               Eliminated     16
    CHRISTOPHER ROBIN ZIMMERMAN   Eliminated     15
    JEFFREY ALAN WAGNER           Eliminated     14
    NEAL BAXTER                   Eliminated     13
    TROY BENJEGERDES              Eliminated     12
    GREGG A. IVERSON              Eliminated     11
    MERRILL ANDERSON              Eliminated     10
    JOSHUA REA                    Eliminated      9
    BILL KAHN                     Eliminated      8
    JOHN LESLIE HARTWIG           Eliminated      7
    EDMUND BERNARD BRUYERE        Eliminated      6
    JAMES "JIMMY" L. STROUD, JR.  Eliminated      5
    RAHN V. WORKCUFF              Eliminated      4
    BOB "AGAIN" CARNEY JR         Eliminated      3
    CYD GORMAN                    Eliminated      2
    JOHN CHARLES WILSON           Eliminated      1


.. parsed-literal::

    /Users/cdonnay/Documents/GitHub/MGGG/VoteKit/src/votekit/pref_profile/pref_profile.py:1109: UserWarning: Profile does not contain rankings but max_ranking_length=3. Setting max_ranking_length to 0.
      warnings.warn(


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
    
    slate_to_candidates = {"Alpha": ["A", "B"], "Xenon": ["X", "Y"]}
    
    # note that we include candidates with 0 support, and that our preference intervals
    # will automatically rescale to sum to 1
    
    pref_intervals_by_bloc = {
        "Alpha": {
            "Alpha": PreferenceInterval({"A": 0.8, "B": 0.15}),
            "Xenon": PreferenceInterval({"X": 0, "Y": 0.05}),
        },
        "Xenon": {
            "Alpha": PreferenceInterval({"A": 0.05, "B": 0.05}),
            "Xenon": PreferenceInterval({"X": 0.45, "Y": 0.45}),
        },
    }
    
    
    bloc_voter_prop = {"Alpha": 0.8, "Xenon": 0.2}
    
    # assume that each bloc is 90% cohesive
    cohesion_parameters = {
        "Alpha": {"Alpha": 0.9, "Xenon": 0.1},
        "Xenon": {"Xenon": 0.9, "Alpha": 0.1},
    }
    
    bt = bg.slate_BradleyTerry(
        pref_intervals_by_bloc=pref_intervals_by_bloc,
        bloc_voter_prop=bloc_voter_prop,
        slate_to_candidates=slate_to_candidates,
        cohesion_parameters=cohesion_parameters,
    )
    
    profile = bt.generate_profile(number_of_ballots=100)
    print(profile.df)


.. parsed-literal::

                 Ranking_1 Ranking_2 Ranking_3 Ranking_4 Voter Set  Weight
    Ballot Index                                                          
    0                  (A)       (B)       (Y)       (X)        {}    69.0
    1                  (A)       (Y)       (B)       (X)        {}     4.0
    2                  (B)       (A)       (Y)       (X)        {}     7.0
    3                  (Y)       (X)       (A)       (B)        {}     7.0
    4                  (Y)       (X)       (B)       (A)        {}     4.0
    5                  (X)       (Y)       (A)       (B)        {}     5.0
    6                  (X)       (Y)       (B)       (A)        {}     4.0


.. admonition:: A note on s-BT :class: note The probability distribution
that s-BT samples from is too cumbersome to compute for more than 12
candidates. We have implemented a Markov chain Monte Carlo (MCMC)
sampling method to account for this. Simply set
``deterministic = False`` in the ``generate_profile`` method to use the
MCMC code. The sample size should be increased to ensure mixing of the
chain.

.. code:: ipython3

    mcmc_profile = bt.generate_profile(number_of_ballots=10000, deterministic=False)
    print(profile.df)


.. parsed-literal::

                 Ranking_1 Ranking_2 Ranking_3 Ranking_4 Voter Set  Weight
    Ballot Index                                                          
    0                  (A)       (B)       (Y)       (X)        {}    69.0
    1                  (A)       (Y)       (B)       (X)        {}     4.0
    2                  (B)       (A)       (Y)       (X)        {}     7.0
    3                  (Y)       (X)       (A)       (B)        {}     7.0
    4                  (Y)       (X)       (B)       (A)        {}     4.0
    5                  (X)       (Y)       (A)       (B)        {}     5.0
    6                  (X)       (Y)       (B)       (A)        {}     4.0


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

    strong_pref_interval = PreferenceInterval.from_dirichlet(
        candidates=["A", "B", "C"], alpha=0.1
    )
    print("Strong preference for one candidate", strong_pref_interval)
    
    abo_pref_interval = PreferenceInterval.from_dirichlet(
        candidates=["A", "B", "C"], alpha=1
    )
    print("All bets are off preference", abo_pref_interval)
    
    unif_pref_interval = PreferenceInterval.from_dirichlet(
        candidates=["A", "B", "C"], alpha=10
    )
    print("Uniform preference for all candidates", unif_pref_interval)


.. parsed-literal::

    Strong preference for one candidate {'A': 0.9997, 'B': 0.0003, 'C': 0.0}
    All bets are off preference {'A': 0.4123, 'B': 0.008, 'C': 0.5797}
    Uniform preference for all candidates {'A': 0.3021, 'B': 0.3939, 'C': 0.3039}


Let’s initialize the s-PL model from the Dirichlet distribution, using
that to build a preference interval rather than specifying the interval.
Each bloc will need two Dirichlet alpha values; one to describe their
own preference interval, and another to describe their preference for
the opposing candidates.

.. code:: ipython3

    bloc_voter_prop = {"X": 0.8, "Y": 0.2}
    
    # the values of .9 indicate that these blocs are highly polarized;
    # they prefer their own candidates much more than the opposing slate
    cohesion_parameters = {"X": {"X": 0.9, "Y": 0.1}, "Y": {"Y": 0.9, "X": 0.1}}
    
    alphas = {"X": {"X": 2, "Y": 1}, "Y": {"X": 1, "Y": 0.5}}
    
    slate_to_candidates = {"X": ["X1", "X2"], "Y": ["Y1", "Y2"]}
    
    # the from_params method allows us to sample from
    # the Dirichlet distribution for our intervals
    pl = bg.slate_PlackettLuce.from_params(
        slate_to_candidates=slate_to_candidates,
        bloc_voter_prop=bloc_voter_prop,
        cohesion_parameters=cohesion_parameters,
        alphas=alphas,
    )
    
    print("Preference interval for X bloc and X candidates")
    print(pl.pref_intervals_by_bloc["X"]["X"])
    print()
    print("Preference interval for X bloc and Y candidates")
    print(pl.pref_intervals_by_bloc["X"]["Y"])
    
    print()
    profile_dict, agg_profile = pl.generate_profile(number_of_ballots=100, by_bloc=True)
    print(profile_dict["X"].df)


.. parsed-literal::

    Preference interval for X bloc and X candidates
    {'X1': 0.4421, 'X2': 0.5579}
    
    Preference interval for X bloc and Y candidates
    {'Y1': 0.4563, 'Y2': 0.5437}
    
                 Ranking_1 Ranking_2 Ranking_3 Ranking_4  Weight Voter Set
    Ballot Index                                                          
    0                 (X2)      (X1)      (Y2)      (Y1)    20.0        {}
    1                 (X2)      (X1)      (Y1)      (Y2)    14.0        {}
    2                 (X2)      (Y2)      (X1)      (Y1)     1.0        {}
    3                 (X2)      (Y1)      (X1)      (Y2)     2.0        {}
    4                 (X1)      (X2)      (Y2)      (Y1)    22.0        {}
    5                 (X1)      (X2)      (Y1)      (Y2)    10.0        {}
    6                 (X1)      (Y2)      (X2)      (Y1)     2.0        {}
    7                 (X1)      (Y1)      (Y2)      (X2)     1.0        {}
    8                 (X1)      (Y1)      (X2)      (Y2)     2.0        {}
    9                 (Y1)      (X2)      (X1)      (Y2)     1.0        {}
    10                (Y2)      (X1)      (Y1)      (X2)     1.0        {}
    11                (Y2)      (X1)      (X2)      (Y1)     1.0        {}
    12                (Y2)      (X2)      (X1)      (Y1)     2.0        {}
    13                (Y2)      (Y1)      (X2)      (X1)     1.0        {}


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

    bloc_voter_prop = {"W": 0.8, "C": 0.2}
    
    # the values of .9 indicate that these blocs are highly polarized;
    # they prefer their own candidates much more than the opposing slate
    cohesion_parameters = {"W": {"W": 0.9, "C": 0.1}, "C": {"C": 0.9, "W": 0.1}}
    
    alphas = {"W": {"W": 2, "C": 1}, "C": {"W": 1, "C": 0.5}}
    
    slate_to_candidates = {"W": ["W1", "W2", "W3"], "C": ["C1", "C2"]}
    
    cs = bg.CambridgeSampler.from_params(
        slate_to_candidates=slate_to_candidates,
        bloc_voter_prop=bloc_voter_prop,
        cohesion_parameters=cohesion_parameters,
        alphas=alphas,
    )
    
    
    profile = cs.generate_profile(number_of_ballots=1000)
    print(profile.df.to_string())


.. parsed-literal::

                 Ranking_1 Ranking_2 Ranking_3 Ranking_4 Ranking_5 Voter Set  Weight
    Ballot Index                                                                    
    0                 (W1)      (W2)       (~)       (~)       (~)        {}     8.0
    1                 (W1)      (W2)      (W3)       (~)       (~)        {}    23.0
    2                 (W1)      (W2)      (W3)      (C2)       (~)        {}    13.0
    3                 (W1)      (W2)      (W3)      (C2)      (C1)        {}    19.0
    4                 (W1)      (W2)      (W3)      (C1)       (~)        {}     8.0
    5                 (W1)      (W2)      (W3)      (C1)      (C2)        {}     8.0
    6                 (W1)      (W2)      (C2)       (~)       (~)        {}     8.0
    7                 (W1)      (W2)      (C2)      (C1)       (~)        {}     2.0
    8                 (W1)      (W2)      (C2)      (C1)      (W3)        {}     5.0
    9                 (W1)      (W2)      (C2)      (W3)       (~)        {}     3.0
    10                (W1)      (W2)      (C2)      (W3)      (C1)        {}     9.0
    11                (W1)      (W2)      (C1)       (~)       (~)        {}     1.0
    12                (W1)      (W2)      (C1)      (C2)      (W3)        {}     3.0
    13                (W1)      (W2)      (C1)      (W3)      (C2)        {}     2.0
    14                (W1)      (C1)       (~)       (~)       (~)        {}     1.0
    15                (W1)      (C1)      (W3)       (~)       (~)        {}     1.0
    16                (W1)      (C1)      (W3)      (C2)      (W2)        {}     1.0
    17                (W1)      (C1)      (C2)       (~)       (~)        {}     4.0
    18                (W1)      (C1)      (C2)      (W2)       (~)        {}     2.0
    19                (W1)      (C1)      (C2)      (W2)      (W3)        {}     1.0
    20                (W1)      (C1)      (C2)      (W3)      (W2)        {}     4.0
    21                (W1)      (C1)      (W2)       (~)       (~)        {}     2.0
    22                (W1)      (C1)      (W2)      (C2)      (W3)        {}     2.0
    23                (W1)      (C1)      (W2)      (W3)       (~)        {}     2.0
    24                (W1)      (C1)      (W2)      (W3)      (C2)        {}     2.0
    25                (W1)      (W3)       (~)       (~)       (~)        {}     9.0
    26                (W1)      (W3)      (C2)       (~)       (~)        {}     4.0
    27                (W1)      (W3)      (C2)      (W2)       (~)        {}     5.0
    28                (W1)      (W3)      (C2)      (W2)      (C1)        {}     4.0
    29                (W1)      (W3)      (C2)      (C1)       (~)        {}     2.0
    30                (W1)      (W3)      (C2)      (C1)      (W2)        {}     4.0
    31                (W1)      (W3)      (W2)       (~)       (~)        {}    12.0
    32                (W1)      (W3)      (W2)      (C2)       (~)        {}     7.0
    33                (W1)      (W3)      (W2)      (C2)      (C1)        {}    10.0
    34                (W1)      (W3)      (W2)      (C1)       (~)        {}     2.0
    35                (W1)      (W3)      (W2)      (C1)      (C2)        {}     2.0
    36                (W1)      (W3)      (C1)      (C2)       (~)        {}     2.0
    37                (W1)      (W3)      (C1)      (C2)      (W2)        {}     2.0
    38                (W1)      (W3)      (C1)      (W2)      (C2)        {}     4.0
    39                (W1)      (C2)       (~)       (~)       (~)        {}     8.0
    40                (W1)      (C2)      (W3)       (~)       (~)        {}     3.0
    41                (W1)      (C2)      (W3)      (W2)       (~)        {}     3.0
    42                (W1)      (C2)      (W3)      (W2)      (C1)        {}     7.0
    43                (W1)      (C2)      (W3)      (C1)      (W2)        {}     2.0
    44                (W1)      (C2)      (W2)       (~)       (~)        {}     9.0
    45                (W1)      (C2)      (W2)      (C1)       (~)        {}     4.0
    46                (W1)      (C2)      (W2)      (C1)      (W3)        {}     5.0
    47                (W1)      (C2)      (W2)      (W3)       (~)        {}     2.0
    48                (W1)      (C2)      (W2)      (W3)      (C1)        {}     6.0
    49                (W1)      (C2)      (C1)       (~)       (~)        {}     6.0
    50                (W1)      (C2)      (C1)      (W2)       (~)        {}     6.0
    51                (W1)      (C2)      (C1)      (W2)      (W3)        {}     8.0
    52                (W1)      (C2)      (C1)      (W3)       (~)        {}     1.0
    53                (W1)      (C2)      (C1)      (W3)      (W2)        {}     5.0
    54                (W1)       (~)       (~)       (~)       (~)        {}    21.0
    55                (W2)      (C1)       (~)       (~)       (~)        {}     1.0
    56                (W2)      (C1)      (W3)      (C2)       (~)        {}     1.0
    57                (W2)      (C1)      (C2)       (~)       (~)        {}     2.0
    58                (W2)      (C1)      (C2)      (W1)       (~)        {}     1.0
    59                (W2)      (C1)      (C2)      (W1)      (W3)        {}     4.0
    60                (W2)      (C1)      (C2)      (W3)       (~)        {}     2.0
    61                (W2)      (C1)      (C2)      (W3)      (W1)        {}     3.0
    62                (W2)      (C1)      (W1)       (~)       (~)        {}     1.0
    63                (W2)      (C1)      (W1)      (C2)      (W3)        {}     3.0
    64                (W2)      (C1)      (W1)      (W3)       (~)        {}     1.0
    65                (W2)      (C1)      (W1)      (W3)      (C2)        {}     3.0
    66                (W2)      (W1)       (~)       (~)       (~)        {}    11.0
    67                (W2)      (W1)      (W3)       (~)       (~)        {}    20.0
    68                (W2)      (W1)      (W3)      (C2)       (~)        {}    12.0
    69                (W2)      (W1)      (W3)      (C2)      (C1)        {}    18.0
    70                (W2)      (W1)      (W3)      (C1)       (~)        {}     5.0
    71                (W2)      (W1)      (W3)      (C1)      (C2)        {}     5.0
    72                (W2)      (W1)      (C2)       (~)       (~)        {}     6.0
    73                (W2)      (W1)      (C2)      (C1)       (~)        {}     5.0
    74                (W2)      (W1)      (C2)      (C1)      (W3)        {}     9.0
    75                (W2)      (W1)      (C2)      (W3)       (~)        {}     6.0
    76                (W2)      (W1)      (C2)      (W3)      (C1)        {}    13.0
    77                (W2)      (W1)      (C1)      (C2)       (~)        {}     1.0
    78                (W2)      (W1)      (C1)      (C2)      (W3)        {}     3.0
    79                (W2)      (W1)      (C1)      (W3)       (~)        {}     1.0
    80                (W2)      (W1)      (C1)      (W3)      (C2)        {}     1.0
    81                (W2)      (W3)       (~)       (~)       (~)        {}     2.0
    82                (W2)      (W3)      (C2)       (~)       (~)        {}     4.0
    83                (W2)      (W3)      (C2)      (W1)       (~)        {}     1.0
    84                (W2)      (W3)      (C2)      (W1)      (C1)        {}     4.0
    85                (W2)      (W3)      (C2)      (C1)       (~)        {}     2.0
    86                (W2)      (W3)      (C2)      (C1)      (W1)        {}     1.0
    87                (W2)      (W3)      (C1)       (~)       (~)        {}     3.0
    88                (W2)      (W3)      (C1)      (C2)       (~)        {}     1.0
    89                (W2)      (W3)      (C1)      (C2)      (W1)        {}     3.0
    90                (W2)      (W3)      (C1)      (W1)       (~)        {}     1.0
    91                (W2)      (W3)      (C1)      (W1)      (C2)        {}     2.0
    92                (W2)      (W3)      (W1)       (~)       (~)        {}    11.0
    93                (W2)      (W3)      (W1)      (C2)       (~)        {}     4.0
    94                (W2)      (W3)      (W1)      (C2)      (C1)        {}     9.0
    95                (W2)      (W3)      (W1)      (C1)       (~)        {}     2.0
    96                (W2)      (W3)      (W1)      (C1)      (C2)        {}     6.0
    97                (W2)      (C2)       (~)       (~)       (~)        {}     1.0
    98                (W2)      (C2)      (W3)      (W1)       (~)        {}     5.0
    99                (W2)      (C2)      (W3)      (W1)      (C1)        {}     5.0
    100               (W2)      (C2)      (W3)      (C1)       (~)        {}     2.0
    101               (W2)      (C2)      (W3)      (C1)      (W1)        {}     3.0
    102               (W2)      (C2)      (C1)       (~)       (~)        {}     6.0
    103               (W2)      (C2)      (C1)      (W1)       (~)        {}     2.0
    104               (W2)      (C2)      (C1)      (W1)      (W3)        {}     7.0
    105               (W2)      (C2)      (C1)      (W3)       (~)        {}     3.0
    106               (W2)      (C2)      (C1)      (W3)      (W1)        {}     8.0
    107               (W2)      (C2)      (W1)      (C1)       (~)        {}     3.0
    108               (W2)      (C2)      (W1)      (C1)      (W3)        {}     7.0
    109               (W2)      (C2)      (W1)      (W3)       (~)        {}     2.0
    110               (W2)      (C2)      (W1)      (W3)      (C1)        {}     9.0
    111               (W2)       (~)       (~)       (~)       (~)        {}    22.0
    112               (W3)      (W2)       (~)       (~)       (~)        {}     3.0
    113               (W3)      (W2)      (C2)       (~)       (~)        {}     4.0
    114               (W3)      (W2)      (C2)      (W1)       (~)        {}     2.0
    115               (W3)      (W2)      (C2)      (W1)      (C1)        {}     8.0
    116               (W3)      (W2)      (C2)      (C1)       (~)        {}     1.0
    117               (W3)      (W2)      (C2)      (C1)      (W1)        {}     2.0
    118               (W3)      (W2)      (C1)       (~)       (~)        {}     3.0
    119               (W3)      (W2)      (C1)      (C2)       (~)        {}     1.0
    120               (W3)      (W2)      (C1)      (C2)      (W1)        {}     2.0
    121               (W3)      (W2)      (C1)      (W1)      (C2)        {}     2.0
    122               (W3)      (W2)      (W1)       (~)       (~)        {}     4.0
    123               (W3)      (W2)      (W1)      (C2)       (~)        {}     2.0
    124               (W3)      (W2)      (W1)      (C2)      (C1)        {}     3.0
    125               (W3)      (W2)      (W1)      (C1)       (~)        {}     2.0
    126               (W3)      (W2)      (W1)      (C1)      (C2)        {}     3.0
    127               (W3)      (C1)       (~)       (~)       (~)        {}     2.0
    128               (W3)      (C1)      (C2)       (~)       (~)        {}     1.0
    129               (W3)      (C1)      (C2)      (W1)      (W2)        {}     3.0
    130               (W3)      (C1)      (W2)      (C2)       (~)        {}     2.0
    131               (W3)      (C1)      (W1)      (W2)       (~)        {}     3.0
    132               (W3)      (C1)      (W1)      (W2)      (C2)        {}     1.0
    133               (W3)      (W1)       (~)       (~)       (~)        {}     5.0
    134               (W3)      (W1)      (C2)       (~)       (~)        {}     3.0
    135               (W3)      (W1)      (C2)      (C1)      (W2)        {}     2.0
    136               (W3)      (W1)      (W2)       (~)       (~)        {}     7.0
    137               (W3)      (W1)      (W2)      (C2)       (~)        {}     5.0
    138               (W3)      (W1)      (W2)      (C2)      (C1)        {}    10.0
    139               (W3)      (W1)      (W2)      (C1)       (~)        {}     1.0
    140               (W3)      (W1)      (W2)      (C1)      (C2)        {}     3.0
    141               (W3)      (W1)      (C1)       (~)       (~)        {}     2.0
    142               (W3)      (W1)      (C1)      (C2)      (W2)        {}     2.0
    143               (W3)      (W1)      (C1)      (W2)       (~)        {}     3.0
    144               (W3)      (W1)      (C1)      (W2)      (C2)        {}     2.0
    145               (W3)      (C2)       (~)       (~)       (~)        {}     4.0
    146               (W3)      (C2)      (W2)       (~)       (~)        {}     1.0
    147               (W3)      (C2)      (W2)      (W1)       (~)        {}     1.0
    148               (W3)      (C2)      (W2)      (W1)      (C1)        {}     3.0
    149               (W3)      (C2)      (W2)      (C1)      (W1)        {}     1.0
    150               (W3)      (C2)      (C1)       (~)       (~)        {}     5.0
    151               (W3)      (C2)      (C1)      (W2)       (~)        {}     1.0
    152               (W3)      (C2)      (C1)      (W2)      (W1)        {}     5.0
    153               (W3)      (C2)      (C1)      (W1)       (~)        {}     1.0
    154               (W3)      (C2)      (C1)      (W1)      (W2)        {}     2.0
    155               (W3)      (C2)      (W1)       (~)       (~)        {}     2.0
    156               (W3)      (C2)      (W1)      (W2)       (~)        {}     3.0
    157               (W3)      (C2)      (W1)      (W2)      (C1)        {}     4.0
    158               (W3)      (C2)      (W1)      (C1)      (W2)        {}     2.0
    159               (W3)       (~)       (~)       (~)       (~)        {}     8.0
    160               (C2)      (W2)       (~)       (~)       (~)        {}     1.0
    161               (C2)      (W2)      (W3)      (W1)      (C1)        {}     1.0
    162               (C2)      (W2)      (W3)      (C1)      (W1)        {}     1.0
    163               (C2)      (W2)      (C1)      (W1)      (W3)        {}     2.0
    164               (C2)      (W2)      (W1)      (C1)       (~)        {}     1.0
    165               (C2)      (C1)       (~)       (~)       (~)        {}     5.0
    166               (C2)      (C1)      (W3)      (W2)      (W1)        {}     3.0
    167               (C2)      (C1)      (W3)      (W1)      (W2)        {}     3.0
    168               (C2)      (C1)      (W2)       (~)       (~)        {}     2.0
    169               (C2)      (C1)      (W2)      (W1)       (~)        {}     1.0
    170               (C2)      (C1)      (W2)      (W1)      (W3)        {}     2.0
    171               (C2)      (C1)      (W2)      (W3)      (W1)        {}     2.0
    172               (C2)      (C1)      (W1)       (~)       (~)        {}     3.0
    173               (C2)      (C1)      (W1)      (W2)       (~)        {}     1.0
    174               (C2)      (C1)      (W1)      (W2)      (W3)        {}     1.0
    175               (C2)      (W1)       (~)       (~)       (~)        {}     1.0
    176               (C2)      (W1)      (W3)      (W2)      (C1)        {}     1.0
    177               (C2)      (W1)      (W2)       (~)       (~)        {}     2.0
    178               (C2)      (W1)      (W2)      (W3)      (C1)        {}     3.0
    179               (C2)      (W1)      (C1)       (~)       (~)        {}     1.0
    180               (C2)      (W1)      (C1)      (W2)      (W3)        {}     3.0
    181               (C2)      (W1)      (C1)      (W3)      (W2)        {}     1.0
    182               (C2)      (W3)      (W2)       (~)       (~)        {}     2.0
    183               (C2)      (W3)      (W2)      (C1)       (~)        {}     1.0
    184               (C2)      (W3)      (C1)      (W2)      (W1)        {}     1.0
    185               (C2)      (W3)      (W1)       (~)       (~)        {}     1.0
    186               (C2)      (W3)      (W1)      (W2)      (C1)        {}     1.0
    187               (C2)       (~)       (~)       (~)       (~)        {}     9.0
    188               (C1)      (W2)      (C2)      (W3)      (W1)        {}     2.0
    189               (C1)      (W2)      (W1)      (W3)      (C2)        {}     1.0
    190               (C1)      (W1)      (W3)      (C2)       (~)        {}     1.0
    191               (C1)      (W1)      (C2)      (W2)       (~)        {}     1.0
    192               (C1)      (W1)      (C2)      (W3)       (~)        {}     1.0
    193               (C1)      (W1)      (C2)      (W3)      (W2)        {}     1.0
    194               (C1)      (C2)       (~)       (~)       (~)        {}     2.0
    195               (C1)      (C2)      (W3)      (W2)      (W1)        {}     1.0
    196               (C1)      (C2)      (W2)       (~)       (~)        {}     2.0
    197               (C1)      (C2)      (W2)      (W1)      (W3)        {}     2.0
    198               (C1)      (C2)      (W1)      (W2)       (~)        {}     1.0
    199               (C1)      (C2)      (W1)      (W2)      (W3)        {}     4.0
    200               (C1)      (C2)      (W1)      (W3)       (~)        {}     2.0
    201               (C1)       (~)       (~)       (~)       (~)        {}     3.0
    202               (C2)      (C1)      (W3)      (W2)      (W1)        {}    13.0
    203               (C2)      (C1)      (W3)      (W2)       (~)        {}     3.0
    204               (C2)      (C1)      (W3)       (~)       (~)        {}    11.0
    205               (C2)      (C1)      (W3)      (W1)       (~)        {}     2.0
    206               (C2)      (C1)      (W3)      (W1)      (W2)        {}     3.0
    207               (C2)      (C1)      (W1)       (~)       (~)        {}     2.0
    208               (C2)      (C1)      (W1)      (W3)      (W2)        {}     1.0
    209               (C2)      (C1)      (W2)       (~)       (~)        {}     1.0
    210               (C2)      (C1)       (~)       (~)       (~)        {}    14.0
    211               (C2)      (W3)      (C1)      (W2)      (W1)        {}     8.0
    212               (C2)      (W3)      (C1)      (W2)       (~)        {}     1.0
    213               (C2)      (W3)      (C1)       (~)       (~)        {}     6.0
    214               (C2)      (W3)      (C1)      (W1)      (W2)        {}     2.0
    215               (C2)      (W3)      (W1)      (W2)      (C1)        {}     3.0
    216               (C2)      (W3)      (W1)       (~)       (~)        {}     3.0
    217               (C2)      (W3)      (W1)      (C1)      (W2)        {}     4.0
    218               (C2)      (W3)      (W2)       (~)       (~)        {}     2.0
    219               (C2)      (W3)      (W2)      (W1)       (~)        {}     2.0
    220               (C2)      (W3)      (W2)      (W1)      (C1)        {}     9.0
    221               (C2)      (W3)      (W2)      (C1)      (W1)        {}     1.0
    222               (C2)      (W3)       (~)       (~)       (~)        {}     3.0
    223               (C2)      (W2)      (W3)      (W1)       (~)        {}     2.0
    224               (C2)      (W2)      (W3)      (C1)      (W1)        {}     1.0
    225               (C2)      (W2)      (C1)       (~)       (~)        {}     1.0
    226               (C2)       (~)       (~)       (~)       (~)        {}     8.0
    227               (C2)      (W1)      (W3)      (C1)      (W2)        {}     1.0
    228               (C2)      (W1)      (C1)      (W3)      (W2)        {}     1.0
    229               (C1)      (W3)      (W1)      (W2)       (~)        {}     2.0
    230               (C1)      (W3)      (W1)      (W2)      (C2)        {}     3.0
    231               (C1)      (W3)      (W1)       (~)       (~)        {}     1.0
    232               (C1)      (W3)      (W1)      (C2)      (W2)        {}     2.0
    233               (C1)      (W3)      (W2)       (~)       (~)        {}     1.0
    234               (C1)      (W3)      (W2)      (C2)      (W1)        {}     3.0
    235               (C1)      (W3)      (W2)      (C2)       (~)        {}     2.0
    236               (C1)      (W3)      (W2)      (W1)       (~)        {}     3.0
    237               (C1)      (W3)      (W2)      (W1)      (C2)        {}     3.0
    238               (C1)      (W3)       (~)       (~)       (~)        {}     3.0
    239               (C1)      (W3)      (C2)      (W2)      (W1)        {}     3.0
    240               (C1)      (W3)      (C2)      (W2)       (~)        {}     2.0
    241               (C1)      (W3)      (C2)       (~)       (~)        {}     8.0
    242               (C1)      (W3)      (C2)      (W1)       (~)        {}     1.0
    243               (C1)      (W3)      (C2)      (W1)      (W2)        {}     1.0
    244               (C1)      (W2)      (W3)       (~)       (~)        {}     1.0
    245               (C1)      (W2)      (W3)      (C2)      (W1)        {}     1.0
    246               (C1)      (W2)      (W3)      (C2)       (~)        {}     1.0
    247               (C1)      (W2)      (C2)      (W3)      (W1)        {}     1.0
    248               (C1)       (~)       (~)       (~)       (~)        {}     3.0
    249               (C1)      (C2)      (W3)      (W2)      (W1)        {}     3.0
    250               (C1)      (C2)      (W3)      (W2)       (~)        {}     3.0
    251               (C1)      (C2)      (W3)       (~)       (~)        {}     9.0
    252               (C1)      (C2)      (W3)      (W1)       (~)        {}     1.0
    253               (C1)      (C2)      (W2)       (~)       (~)        {}     1.0
    254               (C1)      (C2)      (W2)      (W3)      (W1)        {}     1.0
    255               (C1)      (C2)       (~)       (~)       (~)        {}     9.0
    256               (W3)      (C1)      (W1)      (W2)      (C2)        {}     2.0
    257               (W3)      (W2)      (C1)       (~)       (~)        {}     1.0
    258               (W3)      (W2)      (W1)       (~)       (~)        {}     4.0
    259               (W3)      (W2)       (~)       (~)       (~)        {}     1.0
    260               (W3)      (W2)      (C2)      (C1)      (W1)        {}     1.0
    261               (W3)      (W2)      (C2)      (C1)       (~)        {}     1.0
    262               (W3)       (~)       (~)       (~)       (~)        {}     1.0
    263               (W3)      (W1)      (C1)      (C2)       (~)        {}     1.0
    264               (W3)      (W1)      (C1)      (C2)      (W2)        {}     1.0
    265               (W3)      (W1)      (W2)      (C2)      (C1)        {}     1.0
    266               (W3)      (W1)      (C2)      (W2)      (C1)        {}     1.0
    267               (W3)      (C2)      (W2)       (~)       (~)        {}     1.0
    268               (W3)      (C2)      (W2)      (C1)      (W1)        {}     2.0
    269               (W3)      (C2)      (W2)      (C1)       (~)        {}     1.0
    270               (W2)      (W3)      (W1)       (~)       (~)        {}     1.0


Note: the ballot type (as in, Ws and Cs) is strictly drawn from the
historical frequencies. The candidate IDs (as in W1 and W2 among the W
slate) are filled in by sampling without replacement from the preference
interval that you either provided or made from Dirichlet alphas. That is
the only role of the preference interval.

Conclusion
----------

There are many other models of ballot generation in VoteKit, both for
ranked choice ballots and score based ballots (think cumulative or
approval voting). See the `ballot
generator <../../package_info/api.html#module-votekit.ballot_generator>`__
section of the VoteKit documentation for more.


