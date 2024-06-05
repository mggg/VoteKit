Elections
=========

Elections are the systems or algorithms by which a
``PreferenceProfile``, or collection of ballots, is converted into an
outcome. There are infinitely many different possible election methods,
whether the output is a single winner, a set of winners, or a consensus
ranking. VoteKit has a host of built-in election methods, as well as the
functionality to let you create your own system of election. By the end
of this section, you will have been introduced to the STV and Borda
elections, learned about the ``ElectionState`` object, and created your
own election type.

STV
---

To start, let’s return to the Minneapolis 2013 mayoral race. We first
saw this in `previous notebooks <2_real_and_simulated_profiles.html>`__.
As a reminder, this election had 35 named candidates running for one
seat, and used an IRV election system (which is mathematically
equivalent to a single-winner STV election) to choose the winner. Voters
were only allowed to rank their top three candidates.

Let’s load in the **cast vote record** (CVR) from the election, and do
all of the same cleaning we did before.

.. code:: ipython3

    from votekit.cvr_loaders import load_csv
    from votekit.elections import STV, fractional_transfer
    from votekit.cleaning import remove_noncands
    
    minneapolis_profile = load_csv("mn_2013_cast_vote_record.csv")
    minneapolis_profile = remove_noncands(minneapolis_profile, 
                                          ["undervote", "overvote", "UWI"])
    
    minn_election = STV(profile = minneapolis_profile, 
                        transfer = fractional_transfer, 
                        seats = 1)
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



First, what is this showing? Generally, the winners are listed in the
order they were elected, from the top down. Eliminated candidates are
filled in in the order they were eliminated, bottom-up. If any
candidates are still remaining without having been designated elected or
eliminated, they are in a middle category called ``Remaining``. Ties are
broken by strength, meaning for instance that if 3 candidates are
remaining at the end, they are listed in the order of their first-place
in the final election state. This means that this output can be thought
of as an aggregate ranking vector produced by applying the election
method to the voters’ ranking vectors.

So what exactly is happening in this STV election? STV stands for
“single transferable vote.” Voters cast ranked choice ballots. A
threshold is set of how much support is required for election; if a
candidate crosses the threshold, they are designated as a winner. The
threshold in VoteKit defaults to something called the **Droop quota**.
If there are :math:`N` voters and :math:`m` seats, then Droop quota is
computed as :math:`T=\lfloor N/(m+1)\rfloor +1`. Another option is the
**Hare quota**, which is just :math:`T=N/m`, which is a little bit
larger. Generally, all that is needed of a threshold is that it can’t be
the case that :math:`m+1` candidates exceed it.

In the first round, the first-place votes for each candidate are
tallied. If candidate :math:`A` crosses the threshold, they are elected.
If there were surplus votes, then the ballots with :math:`A` in first
place are transfered, with appropriately reduced weight, to the next
choice of those voters. If another candidate receives enough transfered
support to cross the threshold, they are elected. If no candidate does,
the candidate with the fewest first-place votes is removed from all
ballots, and their votes are transfered with full weight. This repeats
until all seats are filled.

Let’s work out a small example where it is easier to see how STV works.
We will use a fractional transfer rule. If the threshold is :math:`T`
and a candidate received :math:`rT` votes in a given round, where
:math:`r>1`, then the excess is :math:`(r-1)T` and so ballots are now
“discounted” to have new weight :math:`(r-1)/r`. For instance if the
candidate received 150 votes but only needed 100, there would be 50
“excess” votes. Instead of randomly picking 50 out of 150 ballots to
transfer, we transfer them all with a reduced weight of 50/150, or 1/3.
Here is a
`link <https://mggg.org/publications/political-geometry/20-WeighillDuchin.pdf>`__
to a more substantial explainer about ranked choice.

In our example, suppose there are :math:`N=23` voters and :math:`n=7`
candidates running for :math:`m=3` seats with the following profile.

.. code:: ipython3

    from votekit.ballot import Ballot
    from votekit.pref_profile import PreferenceProfile
    
    candidates = ["A", "B", "C", "D", "E", "F", "G"]
    
    ballots = [Ballot(ranking = [{"A"}, {"B"}], weight = 3),
               Ballot(ranking = [{"B"}, {"C"}, {"D"}], weight = 8),
               Ballot(ranking = [{"C"}, {"A"}, {"B"}], weight = 1),
               Ballot(ranking = [{"D"}, {"E"}], weight = 3),
               Ballot(ranking = [{"E"}, {"D"}, {"F"}], weight = 1),
               Ballot(ranking = [{"F"}, {"G"}], weight = 4),
               Ballot(ranking = [{"G"}, {"E"}, {"F"}], weight = 3)]
    
    profile = PreferenceProfile(ballots= ballots)
    
    print(profile)
    print("Number of ballots:", profile.num_ballots())
    print("Number of candidates:", len(profile.get_candidates()))
    
    election = STV(profile = profile, transfer = fractional_transfer, seats = 3)
    
    print("Threshold:", election.threshold)



.. parsed-literal::

      Ballots Weight
    (B, C, D)      8
       (F, G)      4
       (A, B)      3
       (D, E)      3
    (G, E, F)      3
    (C, A, B)      1
    (E, D, F)      1
    Number of ballots: 23
    Number of candidates: 7
    Threshold: 6


What this code block did is create an ``election`` object that lets us
access all the information, round-by-round, about what would happen
under the designated election method.

Now we can review it step by step instead of all at once. Just from a
brief glance at the profile and threshold, we see that candidate B
should be elected in the first round. Let’s see this happen in two ways.

First, observe the first-place votes for each candidate.

.. code:: ipython3

    from votekit.utils import first_place_votes
    print(first_place_votes(election.state.profile))


.. parsed-literal::

    {'B': Fraction(8, 1), 'F': Fraction(4, 1), 'D': Fraction(3, 1), 'G': Fraction(3, 1), 'A': Fraction(3, 1), 'E': Fraction(1, 1), 'C': Fraction(1, 1)}


We can see from this that only B is over the threshold. The other way we
can see who wins in the first round is by running just one step of the
election.

.. code:: ipython3

    print(election.run_step())


.. parsed-literal::

    Current Round: 1
    Candidate                       Status  Round
            B                      Elected      1
            F                    Remaining      1
            G Remaining (tie with C, D, A)      1
            C Remaining (tie with G, D, A)      1
            D Remaining (tie with G, C, A)      1
            A Remaining (tie with G, C, D)      1
            E                    Remaining      1


:math:`B` passed the threshold by 2 votes with a total of 8, so the
:math:`B,C,D` ballot is going to have :math:`B` removed and be given
weight :math:`2/8` (excess/total) times its previous weight of 8. To
check this, election objects have an ``ElectionState`` class within them
that stores this information.

Run this code block a few times, and you’re stepping through the rounds
of the election one at a time. However, once you run the block, the
previous state of the election is overwritten. To restore a state of the
election, you can use the following code.

.. code:: ipython3

    # automatically runs steps 1 through 5
    election.run_to_step(5)
    print("Round 5 state", election.state)
    print()
    
    # resets the election to its ground state
    election.reset()
    print("Ground state", election.state)
    print()


.. parsed-literal::

    Round 5 state Current Round: 5
    Candidate     Status  Round
            B    Elected      1
            F    Elected      4
            D  Remaining      5
            C  Remaining      5
            A Eliminated      5
            G Eliminated      3
            E Eliminated      2
    
    Ground state Current Round: 0
    Empty DataFrame
    Columns: [Candidate, Status, Round]
    Index: []
    


Back to what happened after one step…

.. code:: ipython3

    election.run_to_step(1)
    
    print(election.state.profile)


.. parsed-literal::

      Ballots Weight
       (F, G)      4
       (D, E)      3
    (G, E, F)      3
         (A,)      3
       (C, D)      2
    (E, D, F)      1
       (C, A)      1


Look, :math:`B` is now removed from all ballots, and the :math:`B,C,D`
ballot became :math:`C,D` with weight 2. No one has enough votes to
cross the 6 threshold, so the candidate with the least support will be
eliminated—that is candidate :math:`E`, with only one first-place vote.

.. code:: ipython3

    print(first_place_votes(election.state.profile))
    print()
    print(election.run_to_step(2))
    print()
    # now, since the election state is at step 2, 
    # the profile will return the status at that step
    print(election.state.profile)


.. parsed-literal::

    {'F': Fraction(4, 1), 'D': Fraction(3, 1), 'G': Fraction(3, 1), 'A': Fraction(3, 1), 'E': Fraction(1, 1), 'C': Fraction(3, 1)}
    
    Current Round: 2
    Candidate                    Status  Round
            B                   Elected      1
            F    Remaining (tie with D)      2
            D    Remaining (tie with F)      2
            C Remaining (tie with G, A)      2
            G Remaining (tie with C, A)      2
            A Remaining (tie with C, G)      2
            E                Eliminated      2
    
    Ballots Weight
     (F, G)      4
       (D,)      3
     (G, F)      3
       (A,)      3
     (C, D)      2
     (D, F)      1
     (C, A)      1


:math:`E` has been removed from all of the ballots. Again, no one
crosses the threshold so the candidate with the fewest first-place votes
will be eliminated.

.. code:: ipython3

    print(first_place_votes(election.state.profile))
    print()
    print(election.run_to_step(3))
    print()
    print(election.state.profile)


.. parsed-literal::

    {'F': Fraction(4, 1), 'D': Fraction(4, 1), 'G': Fraction(3, 1), 'A': Fraction(3, 1), 'C': Fraction(3, 1)}
    
    Current Round: 3
    Candidate                 Status  Round
            B                Elected      1
            F Remaining (tie with D)      3
            D Remaining (tie with F)      3
            C Remaining (tie with G)      3
            G Remaining (tie with C)      3
            A             Eliminated      3
            E             Eliminated      2
    
    Ballots Weight
     (F, G)      4
       (D,)      3
     (G, F)      3
         ()      3
     (C, D)      2
     (D, F)      1
       (C,)      1


Note that here, several candidates were tied for the fewest first-place
votes at this stage. When this happens, VoteKit uses a tiebreaker to
decide who advances. This is customizable; it defaults to ``random``,
but VoteKit also includes ``borda`` and ``firstplace``. The former
breaks ties based on Borda scores, while the latter breaks ties based on
(initial) first-place votes.

The randomization prevents us from saying what exactly is going to
happen as you run the code from here forward.

**Try it yourself**
~~~~~~~~~~~~~~~~~~~

   Keep printing the first-place votes and running a step of the
   election until all seats have been filled. At each step, think
   through why the election state transitioned as it did.

We now change the transfer type. Using the same profile as above, we’ll
now use ``random_transfer``. In fractional transfer, we reweighted all
of the ballots in proportion to the surplus. Here, we will randomly
choose the appropriate number of ballots to transfer (the same number as
the surplus). Though it sounds strange, this is the method actually used
in Cambridge, MA. (Recall that Cambridge has used STV continuously since
1941 so back in the day they probably needed a low-tech physical way to
do the transfers.)

.. code:: ipython3

    from votekit.elections import random_transfer
    candidates = ["A", "B", "C", "D", "E", "F", "G"]
    
    ballots = [Ballot(ranking = [{"A"}, {"B"}], weight = 3),
               Ballot(ranking = [{"B"}, {"C"}, {"D"}], weight = 8),
               Ballot(ranking = [{"B"}, {"D"}, {"C"}], weight = 8),
               Ballot(ranking = [{"C"}, {"A"}, {"B"}], weight = 1),
               Ballot(ranking = [{"D"}, {"E"}], weight = 1),
               Ballot(ranking = [{"E"}, {"D"}, {"F"}], weight = 1),
               Ballot(ranking = [{"F"}, {"G"}], weight = 4),
               Ballot(ranking = [{"G"}, {"E"}, {"F"}], weight = 1)]
    
    profile = PreferenceProfile(ballots= ballots)
    
    print(profile)
    print("Number of ballots:", profile.num_ballots())
    print("Number of candidates:", len(profile.get_candidates()))
    print()
    
    election = STV(profile = profile, transfer = random_transfer, seats = 2)
    
    election.run_election()
    



.. parsed-literal::

      Ballots Weight
    (B, C, D)      8
    (B, D, C)      8
       (F, G)      4
       (A, B)      3
    (C, A, B)      1
       (D, E)      1
    (E, D, F)      1
    (G, E, F)      1
    Number of ballots: 27
    Number of candidates: 7
    
    Current Round: 7




.. parsed-literal::

    Candidate     Status  Round
            B    Elected      1
            D    Elected      7
            F Eliminated      6
            A Eliminated      5
            C Eliminated      4
            E Eliminated      3
            G Eliminated      2



**Try it yourself**
~~~~~~~~~~~~~~~~~~~

   Rerun the code above until you see that different candidates can win
   under random transfer.

Election State
--------------

Let’s poke around the ``ElectionState`` class a bit more. It contains a
lot of useful information about what is happening in an election. We
will also introduce the Borda election.

Borda Election
~~~~~~~~~~~~~~

In a Borda election, ranked ballots are converted to a score for a
candidate, and then the candidates with the highest scores win. The
traditional score vector is :math:`(n,n-1,\dots,1)`: that is, if there
are :math:`n` candidates, the first-place candidate on a ballot is given
:math:`n` points, the second place :math:`n-1`, all the way down to
last, who is given :math:`1` point. You can change the score vector
using the ``score_vector`` parameter.

.. code:: ipython3

    from votekit.elections import Borda
    import votekit.ballot_generator as bg
    candidates  = ["A", "B", "C", "D", "E", "F"]
    
    # recall IAC generates an "all bets are off" profile
    iac = bg.ImpartialAnonymousCulture(candidates = candidates)
    profile = iac.generate_profile(number_of_ballots= 1000)
    
    election = Borda(profile, seats = 3)

At first, the ``ElectionState`` is empty and the associated profile is
just the raw profile, since nothing has occurred in the election. Only
after either running the election, or running a step in the election, is
an update made to ``ElectionState``.

.. code:: ipython3

    print(election.state.profile)
    print()
    
    print(election.run_step())
    state = election.state



.. parsed-literal::

    PreferenceProfile too long, only showing 15 out of 415 rows.
               Ballots Weight
    (E, A, B, D, C, F)     11
    (E, B, A, C, D, F)      9
    (C, F, E, D, B, A)      9
    (B, A, C, F, D, E)      9
    (C, F, B, D, A, E)      8
    (C, A, B, F, E, D)      8
    (F, C, B, D, E, A)      8
    (C, B, E, F, A, D)      8
    (A, B, E, C, D, F)      8
    (E, C, A, B, F, D)      8
    (C, F, E, B, A, D)      7
    (E, B, F, A, C, D)      7
    (F, A, C, D, B, E)      7
    (A, E, D, C, B, F)      7
    (C, A, B, D, E, F)      7
    
    Current Round: 1
    Candidate     Status  Round
            C    Elected      1
            A    Elected      1
            E    Elected      1
            B Eliminated      1
            D Eliminated      1
            F Eliminated      1


The Borda election is one-shot (like plurality), so running a step or
the election is equivalent. Let’s see what the election state stores.

.. code:: ipython3

    # the winners up to the current round
    print("Winners:", state.winners())
    
    # the eliminated candidates up to the current round
    print("Eliminated:", state.eliminated())
    
    # the current ranking of the candidates
    print("Ranking:", state.rankings())
    
    # the outcome of the given round
    print("Outcome of round 1:", state.round_outcome(1))
    
    # the pandas dataframe that stores information
    print("Pandas dataframe:")
    print(state.status())
    
    # as a dictionary
    print("Dictionary")
    print(state.to_dict())


.. parsed-literal::

    Winners: [{'C'}, {'A'}, {'E'}]
    Eliminated: [{'B'}, {'D'}, {'F'}]
    Ranking: [{'C'}, {'A'}, {'E'}, {'B'}, {'D'}, {'F'}]
    Outcome of round 1: {'Elected': [{'C'}, {'A'}, {'E'}], 'Eliminated': [{'B'}, {'D'}, {'F'}], 'Remaining': []}
    Pandas dataframe:
      Candidate Status       Round
    0  C            Elected  1    
    1  A            Elected  1    
    2  E            Elected  1    
    3  B         Eliminated  1    
    4  D         Eliminated  1    
    5  F         Eliminated  1    
    Dictionary
    {'elected': ['C', 'A', 'E'], 'eliminated': ['B', 'D', 'F'], 'remaining': [], 'ranking': ['C', 'A', 'E', 'B', 'D', 'F']}


We can also save the election state as a json file.

.. code:: ipython3

    state.to_json("borda_results.json")

**Try it yourself**
~~~~~~~~~~~~~~~~~~~

   Using the following preference profile, try changing the score vector
   of a Borda election. Try replacing 3,2,1 with other Borda weights
   (decreasing and non-negative) showing that each candidate can be
   elected.

.. code:: ipython3

    ballots = [Ballot(ranking = [{"A"}, {"B"}, {"C"}], weight = 3),
               Ballot(ranking = [{"A"}, {"C"}, {"B"}], weight = 2),
               Ballot(ranking = [{"B"}, {"C"}, {"A"}], weight = 2),
               Ballot(ranking = [{"C"}, {"B"}, {"A"}], weight = 4)]
    
    profile = PreferenceProfile(ballots=ballots, candidates = ["A", "B", "C"])
    
    # borda election
    score_vector = [3,2,1]
    election = Borda(profile, seats = 1, score_vector = score_vector)
    print(election.run_election())


.. parsed-literal::

    Current Round: 1
    Candidate     Status  Round
            C    Elected      1
            B Eliminated      1
            A Eliminated      1


Since a Borda election is a one-shot election, most of the information
stored in the ``ElectionState`` is extraneous, but you can see its
utility in an STV election where there are many rounds.

.. code:: ipython3

    minneapolis_profile = load_csv("mn_2013_cast_vote_record.csv")
    minneapolis_profile = remove_noncands(minneapolis_profile, 
                                          ["undervote", "overvote", "UWI"])
    
    minn_election = STV(profile = minneapolis_profile, 
                        transfer = fractional_transfer, 
                        seats = 1)
    
    for i in range(1,6):
      minn_election.run_step()
      state = minn_election.state
    
      print(f"Round {i+1}\n")
      # the winners up to the current round
      print("Winners:", state.winners())
    
      # the eliminated candidates up to the current round
      print("Eliminated:", state.eliminated())
    
      # the current ranking of the candidates
      print("Ranking:", state.rankings())
    
      # the outcome of the given round
      print(f"Outcome of round {i}:", state.round_outcome(i))
      print()


.. parsed-literal::

    Round 2
    
    Winners: []
    Eliminated: [{'JOHN CHARLES WILSON'}]
    Ranking: [{'BETSY HODGES'}, {'MARK ANDREW'}, {'DON SAMUELS'}, {'CAM WINTON'}, {'JACKIE CHERRYHOMES'}, {'BOB FINE'}, {'DAN COHEN'}, {'STEPHANIE WOODRUFF'}, {'MARK V ANDERSON'}, {'DOUG MANN'}, {'OLE SAVIOR'}, {'ABDUL M RAHAMAN "THE ROCK"'}, {'ALICIA K. BENNETT'}, {'JAMES EVERETT'}, {'CAPTAIN JACK SPARROW'}, {'TONY LANE'}, {'MIKE GOULD'}, {'KURTIS W. HANNA'}, {'JAYMIE KELLY'}, {'CHRISTOPHER CLARK'}, {'CHRISTOPHER ROBIN ZIMMERMAN'}, {'JEFFREY ALAN WAGNER'}, {'TROY BENJEGERDES'}, {'NEAL BAXTER', 'GREGG A. IVERSON'}, {'JOSHUA REA'}, {'MERRILL ANDERSON'}, {'BILL KAHN'}, {'JOHN LESLIE HARTWIG'}, {'EDMUND BERNARD BRUYERE'}, {'RAHN V. WORKCUFF', 'JAMES "JIMMY" L. STROUD, JR.'}, {'BOB "AGAIN" CARNEY JR'}, {'CYD GORMAN'}, {'JOHN CHARLES WILSON'}]
    Outcome of round 1: {'Elected': [], 'Eliminated': [{'JOHN CHARLES WILSON'}], 'Remaining': [{'BETSY HODGES'}, {'MARK ANDREW'}, {'DON SAMUELS'}, {'CAM WINTON'}, {'JACKIE CHERRYHOMES'}, {'BOB FINE'}, {'DAN COHEN'}, {'STEPHANIE WOODRUFF'}, {'MARK V ANDERSON'}, {'DOUG MANN'}, {'OLE SAVIOR'}, {'ABDUL M RAHAMAN "THE ROCK"'}, {'ALICIA K. BENNETT'}, {'JAMES EVERETT'}, {'CAPTAIN JACK SPARROW'}, {'TONY LANE'}, {'MIKE GOULD'}, {'KURTIS W. HANNA'}, {'JAYMIE KELLY'}, {'CHRISTOPHER CLARK'}, {'CHRISTOPHER ROBIN ZIMMERMAN'}, {'JEFFREY ALAN WAGNER'}, {'TROY BENJEGERDES'}, {'NEAL BAXTER', 'GREGG A. IVERSON'}, {'JOSHUA REA'}, {'MERRILL ANDERSON'}, {'BILL KAHN'}, {'JOHN LESLIE HARTWIG'}, {'EDMUND BERNARD BRUYERE'}, {'RAHN V. WORKCUFF', 'JAMES "JIMMY" L. STROUD, JR.'}, {'BOB "AGAIN" CARNEY JR'}, {'CYD GORMAN'}]}
    
    Round 3
    
    Winners: []
    Eliminated: [{'CYD GORMAN'}, {'JOHN CHARLES WILSON'}]
    Ranking: [{'BETSY HODGES'}, {'MARK ANDREW'}, {'DON SAMUELS'}, {'CAM WINTON'}, {'JACKIE CHERRYHOMES'}, {'BOB FINE'}, {'DAN COHEN'}, {'STEPHANIE WOODRUFF'}, {'MARK V ANDERSON'}, {'DOUG MANN'}, {'OLE SAVIOR'}, {'ABDUL M RAHAMAN "THE ROCK"'}, {'ALICIA K. BENNETT'}, {'JAMES EVERETT'}, {'CAPTAIN JACK SPARROW'}, {'TONY LANE'}, {'MIKE GOULD'}, {'KURTIS W. HANNA'}, {'JAYMIE KELLY'}, {'CHRISTOPHER CLARK'}, {'CHRISTOPHER ROBIN ZIMMERMAN'}, {'JEFFREY ALAN WAGNER'}, {'TROY BENJEGERDES'}, {'GREGG A. IVERSON'}, {'NEAL BAXTER'}, {'JOSHUA REA'}, {'MERRILL ANDERSON'}, {'BILL KAHN'}, {'JOHN LESLIE HARTWIG'}, {'EDMUND BERNARD BRUYERE'}, {'RAHN V. WORKCUFF', 'JAMES "JIMMY" L. STROUD, JR.'}, {'BOB "AGAIN" CARNEY JR'}, {'CYD GORMAN'}, {'JOHN CHARLES WILSON'}]
    Outcome of round 2: {'Elected': [], 'Eliminated': [{'CYD GORMAN'}], 'Remaining': [{'BETSY HODGES'}, {'MARK ANDREW'}, {'DON SAMUELS'}, {'CAM WINTON'}, {'JACKIE CHERRYHOMES'}, {'BOB FINE'}, {'DAN COHEN'}, {'STEPHANIE WOODRUFF'}, {'MARK V ANDERSON'}, {'DOUG MANN'}, {'OLE SAVIOR'}, {'ABDUL M RAHAMAN "THE ROCK"'}, {'ALICIA K. BENNETT'}, {'JAMES EVERETT'}, {'CAPTAIN JACK SPARROW'}, {'TONY LANE'}, {'MIKE GOULD'}, {'KURTIS W. HANNA'}, {'JAYMIE KELLY'}, {'CHRISTOPHER CLARK'}, {'CHRISTOPHER ROBIN ZIMMERMAN'}, {'JEFFREY ALAN WAGNER'}, {'TROY BENJEGERDES'}, {'GREGG A. IVERSON'}, {'NEAL BAXTER'}, {'JOSHUA REA'}, {'MERRILL ANDERSON'}, {'BILL KAHN'}, {'JOHN LESLIE HARTWIG'}, {'EDMUND BERNARD BRUYERE'}, {'RAHN V. WORKCUFF', 'JAMES "JIMMY" L. STROUD, JR.'}, {'BOB "AGAIN" CARNEY JR'}]}
    
    Round 4
    
    Winners: []
    Eliminated: [{'BOB "AGAIN" CARNEY JR'}, {'CYD GORMAN'}, {'JOHN CHARLES WILSON'}]
    Ranking: [{'BETSY HODGES'}, {'MARK ANDREW'}, {'DON SAMUELS'}, {'CAM WINTON'}, {'JACKIE CHERRYHOMES'}, {'BOB FINE'}, {'DAN COHEN'}, {'STEPHANIE WOODRUFF'}, {'MARK V ANDERSON'}, {'DOUG MANN'}, {'OLE SAVIOR'}, {'ABDUL M RAHAMAN "THE ROCK"'}, {'ALICIA K. BENNETT'}, {'JAMES EVERETT'}, {'CAPTAIN JACK SPARROW'}, {'TONY LANE'}, {'MIKE GOULD'}, {'KURTIS W. HANNA'}, {'JAYMIE KELLY'}, {'CHRISTOPHER CLARK'}, {'CHRISTOPHER ROBIN ZIMMERMAN'}, {'JEFFREY ALAN WAGNER'}, {'TROY BENJEGERDES'}, {'GREGG A. IVERSON'}, {'NEAL BAXTER'}, {'MERRILL ANDERSON', 'JOSHUA REA'}, {'BILL KAHN'}, {'JOHN LESLIE HARTWIG'}, {'EDMUND BERNARD BRUYERE'}, {'JAMES "JIMMY" L. STROUD, JR.'}, {'RAHN V. WORKCUFF'}, {'BOB "AGAIN" CARNEY JR'}, {'CYD GORMAN'}, {'JOHN CHARLES WILSON'}]
    Outcome of round 3: {'Elected': [], 'Eliminated': [{'BOB "AGAIN" CARNEY JR'}], 'Remaining': [{'BETSY HODGES'}, {'MARK ANDREW'}, {'DON SAMUELS'}, {'CAM WINTON'}, {'JACKIE CHERRYHOMES'}, {'BOB FINE'}, {'DAN COHEN'}, {'STEPHANIE WOODRUFF'}, {'MARK V ANDERSON'}, {'DOUG MANN'}, {'OLE SAVIOR'}, {'ABDUL M RAHAMAN "THE ROCK"'}, {'ALICIA K. BENNETT'}, {'JAMES EVERETT'}, {'CAPTAIN JACK SPARROW'}, {'TONY LANE'}, {'MIKE GOULD'}, {'KURTIS W. HANNA'}, {'JAYMIE KELLY'}, {'CHRISTOPHER CLARK'}, {'CHRISTOPHER ROBIN ZIMMERMAN'}, {'JEFFREY ALAN WAGNER'}, {'TROY BENJEGERDES'}, {'GREGG A. IVERSON'}, {'NEAL BAXTER'}, {'MERRILL ANDERSON', 'JOSHUA REA'}, {'BILL KAHN'}, {'JOHN LESLIE HARTWIG'}, {'EDMUND BERNARD BRUYERE'}, {'JAMES "JIMMY" L. STROUD, JR.'}, {'RAHN V. WORKCUFF'}]}
    
    Round 5
    
    Winners: []
    Eliminated: [{'RAHN V. WORKCUFF'}, {'BOB "AGAIN" CARNEY JR'}, {'CYD GORMAN'}, {'JOHN CHARLES WILSON'}]
    Ranking: [{'BETSY HODGES'}, {'MARK ANDREW'}, {'DON SAMUELS'}, {'CAM WINTON'}, {'JACKIE CHERRYHOMES'}, {'BOB FINE'}, {'DAN COHEN'}, {'STEPHANIE WOODRUFF'}, {'MARK V ANDERSON'}, {'DOUG MANN'}, {'OLE SAVIOR'}, {'JAMES EVERETT', 'ABDUL M RAHAMAN "THE ROCK"'}, {'ALICIA K. BENNETT'}, {'CAPTAIN JACK SPARROW'}, {'TONY LANE'}, {'MIKE GOULD'}, {'KURTIS W. HANNA'}, {'JAYMIE KELLY'}, {'CHRISTOPHER CLARK'}, {'CHRISTOPHER ROBIN ZIMMERMAN'}, {'JEFFREY ALAN WAGNER'}, {'NEAL BAXTER'}, {'TROY BENJEGERDES'}, {'GREGG A. IVERSON'}, {'JOSHUA REA'}, {'MERRILL ANDERSON'}, {'BILL KAHN'}, {'JOHN LESLIE HARTWIG'}, {'EDMUND BERNARD BRUYERE'}, {'JAMES "JIMMY" L. STROUD, JR.'}, {'RAHN V. WORKCUFF'}, {'BOB "AGAIN" CARNEY JR'}, {'CYD GORMAN'}, {'JOHN CHARLES WILSON'}]
    Outcome of round 4: {'Elected': [], 'Eliminated': [{'RAHN V. WORKCUFF'}], 'Remaining': [{'BETSY HODGES'}, {'MARK ANDREW'}, {'DON SAMUELS'}, {'CAM WINTON'}, {'JACKIE CHERRYHOMES'}, {'BOB FINE'}, {'DAN COHEN'}, {'STEPHANIE WOODRUFF'}, {'MARK V ANDERSON'}, {'DOUG MANN'}, {'OLE SAVIOR'}, {'JAMES EVERETT', 'ABDUL M RAHAMAN "THE ROCK"'}, {'ALICIA K. BENNETT'}, {'CAPTAIN JACK SPARROW'}, {'TONY LANE'}, {'MIKE GOULD'}, {'KURTIS W. HANNA'}, {'JAYMIE KELLY'}, {'CHRISTOPHER CLARK'}, {'CHRISTOPHER ROBIN ZIMMERMAN'}, {'JEFFREY ALAN WAGNER'}, {'NEAL BAXTER'}, {'TROY BENJEGERDES'}, {'GREGG A. IVERSON'}, {'JOSHUA REA'}, {'MERRILL ANDERSON'}, {'BILL KAHN'}, {'JOHN LESLIE HARTWIG'}, {'EDMUND BERNARD BRUYERE'}, {'JAMES "JIMMY" L. STROUD, JR.'}]}
    
    Round 6
    
    Winners: []
    Eliminated: [{'JAMES "JIMMY" L. STROUD, JR.'}, {'RAHN V. WORKCUFF'}, {'BOB "AGAIN" CARNEY JR'}, {'CYD GORMAN'}, {'JOHN CHARLES WILSON'}]
    Ranking: [{'BETSY HODGES'}, {'MARK ANDREW'}, {'DON SAMUELS'}, {'CAM WINTON'}, {'JACKIE CHERRYHOMES'}, {'BOB FINE'}, {'DAN COHEN'}, {'STEPHANIE WOODRUFF'}, {'MARK V ANDERSON'}, {'DOUG MANN'}, {'OLE SAVIOR'}, {'ABDUL M RAHAMAN "THE ROCK"'}, {'ALICIA K. BENNETT'}, {'JAMES EVERETT'}, {'CAPTAIN JACK SPARROW'}, {'TONY LANE'}, {'MIKE GOULD'}, {'JAYMIE KELLY'}, {'KURTIS W. HANNA'}, {'CHRISTOPHER CLARK'}, {'CHRISTOPHER ROBIN ZIMMERMAN'}, {'JEFFREY ALAN WAGNER'}, {'NEAL BAXTER'}, {'TROY BENJEGERDES'}, {'GREGG A. IVERSON'}, {'MERRILL ANDERSON'}, {'JOSHUA REA'}, {'BILL KAHN'}, {'JOHN LESLIE HARTWIG'}, {'EDMUND BERNARD BRUYERE'}, {'JAMES "JIMMY" L. STROUD, JR.'}, {'RAHN V. WORKCUFF'}, {'BOB "AGAIN" CARNEY JR'}, {'CYD GORMAN'}, {'JOHN CHARLES WILSON'}]
    Outcome of round 5: {'Elected': [], 'Eliminated': [{'JAMES "JIMMY" L. STROUD, JR.'}], 'Remaining': [{'BETSY HODGES'}, {'MARK ANDREW'}, {'DON SAMUELS'}, {'CAM WINTON'}, {'JACKIE CHERRYHOMES'}, {'BOB FINE'}, {'DAN COHEN'}, {'STEPHANIE WOODRUFF'}, {'MARK V ANDERSON'}, {'DOUG MANN'}, {'OLE SAVIOR'}, {'ABDUL M RAHAMAN "THE ROCK"'}, {'ALICIA K. BENNETT'}, {'JAMES EVERETT'}, {'CAPTAIN JACK SPARROW'}, {'TONY LANE'}, {'MIKE GOULD'}, {'JAYMIE KELLY'}, {'KURTIS W. HANNA'}, {'CHRISTOPHER CLARK'}, {'CHRISTOPHER ROBIN ZIMMERMAN'}, {'JEFFREY ALAN WAGNER'}, {'NEAL BAXTER'}, {'TROY BENJEGERDES'}, {'GREGG A. IVERSON'}, {'MERRILL ANDERSON'}, {'JOSHUA REA'}, {'BILL KAHN'}, {'JOHN LESLIE HARTWIG'}, {'EDMUND BERNARD BRUYERE'}]}
    


Conclusion
----------

There are many different possible election methods, both for choosing a
single seat or multiple seats. VoteKit has a host of built-in election
methods, as well as the functionality to let you create your own kind of
election. You have been introduced to the STV and Borda elections and
learned about the ``ElectionState`` object. This should allow you to
model any kind of elections you see in the real world, including rules
that have not yet been implemented in VoteKit.

Further Prompts: Creating your own election system
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

VoteKit can’t be comprehensive in terms of possible election rules.
However, with the ``Election`` and ``ElectionState`` classes, you can
create your own. Let’s create a bit of a silly example; to elect
:math:`m` seats, at each stage of the election we randomly choose one
candidate to elect.

.. code:: ipython3

    from votekit.models import Election
    from votekit.election_state import ElectionState
    from votekit.utils import remove_cand
    import random
    
    class RandomWinners(Election):
        """
        Simulates an election where we randomly choose winners at each stage.
    
        **Attributes**
    
        `profile`
        :   PreferenceProfile to run election on
    
        `seats`
        :   number of seats to be elected
    
    
        """
    
        def __init__(self, profile: PreferenceProfile, seats: int):
            # the super method says call the Election class
            # ballot_ties = True means it will resolve any ties in our ballots
            super().__init__(profile, ballot_ties = True)
    
            self.seats = seats
    
        def next_round(self) -> bool:
            """
            Determines if another round is needed.
    
            Returns:
                True if number of seats has not been met, False otherwise
            """
            cands_elected = 0
            for s in self.state.winners():
                cands_elected += len(s)
            return cands_elected < self.seats
    
        def run_step(self):
            if self.next_round():
              # get the remaining candidates
              remaining = self.state.profile.get_candidates()
    
              # randomly choose one
              winning_candidate  = random.choice(remaining)
              # some formatting to make it compatible with ElectionState, which
              # requires a list of sets of strings
              elected =[{winning_candidate}]
    
              # remove the winner from the ballots
              new_ballots = remove_cand(winning_candidate, self.state.profile.ballots)
              new_profile = PreferenceProfile(ballots= new_ballots)
    
              # determine who remains
              remaining = [{c} for c in remaining if c != winning_candidate]
    
    
              # update for the next round
              self.state = ElectionState(curr_round = self.state.curr_round + 1,
                                        elected = elected,
                                        eliminated_cands = [],
                                        remaining = remaining,
                                        profile = new_profile,
                                        previous= self.state)
    
              # if this is the last round, move remaining to eliminated
              if not self.next_round():
                self.state = ElectionState(curr_round = self.state.curr_round,
                                        elected = elected,
                                        eliminated_cands = remaining,
                                        remaining = [],
                                        profile = new_profile,
                                        previous= self.state.previous)
              return(self.state)
    
    
        def run_election(self):
            # run steps until we elect the required number of candidates
            while self.next_round():
                self.run_step()
    
            return(self.state)


.. code:: ipython3

    candidates  = ["A", "B", "C", "D", "E", "F"]
    profile = bg.ImpartialCulture(candidates = candidates).generate_profile(1000)
    
    election = RandomWinners(profile= profile, seats  = 3)

.. code:: ipython3

    print(election.run_step())
    print(election.run_step())
    print(election.run_step())


.. parsed-literal::

    Current Round: 1
    Candidate    Status  Round
            F   Elected      1
            B Remaining      1
            D Remaining      1
            A Remaining      1
            E Remaining      1
            C Remaining      1
    Current Round: 2
    Candidate    Status  Round
            F   Elected      1
            E   Elected      2
            B Remaining      2
            D Remaining      2
            A Remaining      2
            C Remaining      2
    Current Round: 3
    Candidate     Status  Round
            F    Elected      1
            E    Elected      2
            D    Elected      3
            B Eliminated      3
            A Eliminated      3
            C Eliminated      3


**Try it yourself**
~~~~~~~~~~~~~~~~~~~

   Create an election class called ``AlphabeticalElection`` that elects
   a number of candidates decided by the user simply based on
   alphabetical order. You mind find it helpful to use the following
   code which sorts a list of strings:

.. code:: ipython3

    # Original list of strings
    original_list = ["banana", "apple", "grape", "orange"]
    
    # Alphabetically sorted list
    sorted_list = sorted(original_list)
    
    # Print the sorted list
    print(sorted_list)


.. parsed-literal::

    ['apple', 'banana', 'grape', 'orange']


.. code:: ipython3

    class AlphabeticaElection(Election):
        """
        Simulates an election where we choose alphabetically.
    
        **Attributes**
    
        `profile`
        :   PreferenceProfile to run election on
    
        `seats`
        :   number of seats to be elected
    
    
        """
    
        def __init__(self, profile: PreferenceProfile, seats: int):
            # the super method says call the Election class
            # ballot_ties = True means it will resolve any ties in our ballots
            super().__init__(profile, ballot_ties = True)
    
            self.seats = seats
    
        def next_round(self) -> bool:
            """
            Determines if another round is needed.
    
            Returns:
                True if number of seats has not been met, False otherwise
            """
            cands_elected = 0
            for s in self.state.winners():
                cands_elected += len(s)
            return cands_elected < self.seats
    
        def run_step(self):
            if self.next_round():
    
              # do some stuff!
    
              return(self.state)
    
    
        def run_election(self):
            # run steps until we elect the required number of candidates
            while self.next_round():
                self.run_step()
    
            return(self.state)

