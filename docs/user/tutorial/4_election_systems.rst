Elections
=========

Elections are the systems or algorithms by which a
``PreferenceProfile``, or collection of ballots, is converted into an
outcome. There are infinitely many different possible election methods,
whether the output is a single winner, a set of winners, or a consensus
ranking. VoteKit has a host of built-in election methods, as well as the
functionality to let you create your own system of election. By the end
of this section, you will have been introduced to the STV and Borda
elections, learned about the ``Election`` object, and created your own
election type.

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

    from votekit.cvr_loaders import load_ranking_csv
    from votekit.elections import STV
    from votekit.cleaning import remove_repeated_candidates, remove_cand, condense_profile
    
    minneapolis_profile = load_ranking_csv("mn_2013_cast_vote_record.csv", rank_cols=[0,1,2], header_row=0)
    minneapolis_profile = condense_profile(remove_repeated_candidates(remove_cand(["undervote", "overvote", "UWI"], minneapolis_profile)))
    
    # m = 1 means 1 seat
    minn_election = STV(profile=minneapolis_profile, m=1)
    print(minn_election)


.. parsed-literal::

    Profile contains rankings: True
    Maximum ranking length: 3
    Profile contains scores: False
    Candidates: ('KURTIS W. HANNA', 'undervote', 'STEPHANIE WOODRUFF', 'GREGG A. IVERSON', 'JOHN CHARLES WILSON', 'JOSHUA REA', 'JACKIE CHERRYHOMES', 'BILL KAHN', 'CHRISTOPHER ROBIN ZIMMERMAN', 'CHRISTOPHER CLARK', 'MARK V ANDERSON', 'JAMES EVERETT', 'JAMES "JIMMY" L. STROUD, JR.', 'NEAL BAXTER', 'EDMUND BERNARD BRUYERE', 'overvote', 'MARK ANDREW', 'UWI', 'CAPTAIN JACK SPARROW', 'ALICIA K. BENNETT', 'CYD GORMAN', 'BOB "AGAIN" CARNEY JR', 'JEFFREY ALAN WAGNER', 'JOHN LESLIE HARTWIG', 'OLE SAVIOR', 'TROY BENJEGERDES', 'TONY LANE', 'CAM WINTON', 'BETSY HODGES', 'DAN COHEN', 'DON SAMUELS', 'BOB FINE', 'RAHN V. WORKCUFF', 'MERRILL ANDERSON', 'ABDUL M RAHAMAN "THE ROCK"', 'JAYMIE KELLY', 'MIKE GOULD', 'DOUG MANN')
    Candidates who received votes: ('KURTIS W. HANNA', 'undervote', 'STEPHANIE WOODRUFF', 'GREGG A. IVERSON', 'JOHN CHARLES WILSON', 'JOSHUA REA', 'JACKIE CHERRYHOMES', 'BILL KAHN', 'CHRISTOPHER ROBIN ZIMMERMAN', 'CHRISTOPHER CLARK', 'MARK V ANDERSON', 'JAMES EVERETT', 'JAMES "JIMMY" L. STROUD, JR.', 'NEAL BAXTER', 'EDMUND BERNARD BRUYERE', 'overvote', 'MARK ANDREW', 'UWI', 'CAPTAIN JACK SPARROW', 'ALICIA K. BENNETT', 'CYD GORMAN', 'BOB "AGAIN" CARNEY JR', 'JEFFREY ALAN WAGNER', 'JOHN LESLIE HARTWIG', 'OLE SAVIOR', 'TROY BENJEGERDES', 'TONY LANE', 'CAM WINTON', 'BETSY HODGES', 'DAN COHEN', 'DON SAMUELS', 'BOB FINE', 'RAHN V. WORKCUFF', 'MERRILL ANDERSON', 'ABDUL M RAHAMAN "THE ROCK"', 'JAYMIE KELLY', 'MIKE GOULD', 'DOUG MANN')
    Total number of Ballot objects: 80101
    Total weight of Ballot objects: 80101.0
    
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
    
    ballots = [
        Ballot(ranking=[{"A"}, {"B"}], weight=3),
        Ballot(ranking=[{"B"}, {"C"}, {"D"}], weight=8),
        Ballot(ranking=[{"C"}, {"A"}, {"B"}], weight=1),
        Ballot(ranking=[{"D"}, {"E"}], weight=3),
        Ballot(ranking=[{"E"}, {"D"}, {"F"}], weight=1),
        Ballot(ranking=[{"F"}, {"G"}], weight=4),
        Ballot(ranking=[{"G"}, {"E"}, {"F"}], weight=3),
    ]
    
    profile = PreferenceProfile(ballots=ballots)
    
    print(profile.df)
    print("Sum of ballot weights:", profile.total_ballot_wt)
    print("Number of candidates:", len(profile.candidates))
    print()
    election = STV(profile=profile, m=3)
    
    print("Threshold:", election.threshold)
    print("Number of rounds", len(election))
    print(election)


.. parsed-literal::

                 Ranking_1 Ranking_2 Ranking_3 Voter Set  Weight
    Ballot Index                                                
    0                  (A)       (B)       (~)        {}     3.0
    1                  (B)       (C)       (D)        {}     8.0
    2                  (C)       (A)       (B)        {}     1.0
    3                  (D)       (E)       (~)        {}     3.0
    4                  (E)       (D)       (F)        {}     1.0
    5                  (F)       (G)       (~)        {}     4.0
    6                  (G)       (E)       (F)        {}     3.0
    Sum of ballot weights: 23.0
    Number of candidates: 7
    
    Initial tiebreak was unsuccessful, performing random tiebreak
    Threshold: 6
    Number of rounds 6
           Status  Round
    B     Elected      1
    D     Elected      4
    F     Elected      6
    A   Remaining      6
    G  Eliminated      5
    C  Eliminated      3
    E  Eliminated      2


What this code block did is create an ``Election`` object that lets us
access all the information, round-by-round, about what would happen
under the designated election method. The message about a tiebreak
indicates that in some round, a random tiebreak was needed.

We can review it step-by-step instead of all at once. Just from a brief
glance at the profile and threshold, we see that candidate B should be
elected in the first round. Let’s see this happen in two ways.

First, observe the first-place votes for each candidate. These are
stored in the round 0 ``ElectionState`` object, which can be accessed as
follows.

.. code:: ipython3

    election.election_states[0].scores




.. parsed-literal::

    {'A': np.float64(3.0),
     'B': np.float64(8.0),
     'C': np.float64(1.0),
     'D': np.float64(3.0),
     'E': np.float64(1.0),
     'F': np.float64(4.0),
     'G': np.float64(3.0)}



We can see from this that only B is over the threshold. The other way we
can see who wins in the first round is by looking at the next
``ElectionState``.

.. code:: ipython3

    print("elected", election.election_states[1].elected)
    print("\neliminated", election.election_states[1].eliminated)
    print("\nremaining", election.election_states[1].remaining)


.. parsed-literal::

    elected (frozenset({'B'}),)
    
    eliminated (frozenset(),)
    
    remaining (frozenset({'F'}), frozenset({'A', 'C', 'D', 'G'}), frozenset({'E'}))


:math:`B` passed the threshold by 2 votes with a total of 8, so the
:math:`B,C,D` ballot is going to have :math:`B` removed and be given
weight :math:`2/8` (excess/total) times its previous weight of 8. To
check this, election objects have a method called ``get_profile()`` that
returns the ``PreferenceProfile`` after a particular round.

.. code:: ipython3

    election.get_profile(1).df




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Ranking_1</th>
          <th>Ranking_2</th>
          <th>Ranking_3</th>
          <th>Voter Set</th>
          <th>Weight</th>
        </tr>
        <tr>
          <th>Ballot Index</th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>(C)</td>
          <td>(D)</td>
          <td>(~)</td>
          <td>{}</td>
          <td>2.0</td>
        </tr>
        <tr>
          <th>1</th>
          <td>(E)</td>
          <td>(D)</td>
          <td>(F)</td>
          <td>{}</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>2</th>
          <td>(G)</td>
          <td>(E)</td>
          <td>(F)</td>
          <td>{}</td>
          <td>3.0</td>
        </tr>
        <tr>
          <th>3</th>
          <td>(A)</td>
          <td>(~)</td>
          <td>(~)</td>
          <td>{}</td>
          <td>3.0</td>
        </tr>
        <tr>
          <th>4</th>
          <td>(C)</td>
          <td>(A)</td>
          <td>(~)</td>
          <td>{}</td>
          <td>1.0</td>
        </tr>
        <tr>
          <th>5</th>
          <td>(D)</td>
          <td>(E)</td>
          <td>(~)</td>
          <td>{}</td>
          <td>3.0</td>
        </tr>
        <tr>
          <th>6</th>
          <td>(F)</td>
          <td>(G)</td>
          <td>(~)</td>
          <td>{}</td>
          <td>4.0</td>
        </tr>
      </tbody>
    </table>
    </div>



Look, :math:`B` is now removed from all ballots, and the :math:`B,C,D`
ballot became :math:`C,D` with weight 2. No one has enough votes to
cross the 6 threshold, so the candidate with the least support will be
eliminated—that is candidate :math:`E`, with only one first-place vote.

We also introduce the ``get_step()`` method which accesses the profile
and state of a given round.

.. code:: ipython3

    print("fpv after round 1:", election.election_states[1].scores)
    print("go to the next step\n")
    
    profile, state = election.get_step(2)
    print("elected", state.elected)
    print("\neliminated", state.eliminated)
    print("\nremaining", state.remaining)
    print(profile.df)


.. parsed-literal::

    fpv after round 1: {'C': np.float64(3.0), 'D': np.float64(3.0), 'E': np.float64(1.0), 'F': np.float64(4.0), 'G': np.float64(3.0), 'A': np.float64(3.0)}
    go to the next step
    
    elected (frozenset(),)
    
    eliminated (frozenset({'E'}),)
    
    remaining (frozenset({'D', 'F'}), frozenset({'A', 'C', 'G'}))
                 Ranking_1 Ranking_2 Ranking_3 Voter Set  Weight
    Ballot Index                                                
    0                  (C)       (D)       (~)        {}     2.0
    1                  (D)       (F)       (~)        {}     1.0
    2                  (G)       (F)       (~)        {}     3.0
    3                  (A)       (~)       (~)        {}     3.0
    4                  (C)       (A)       (~)        {}     1.0
    5                  (D)       (~)       (~)        {}     3.0
    6                  (F)       (G)       (~)        {}     4.0


:math:`E` has been removed from all of the ballots. Again, no one
crosses the threshold so the candidate with the fewest first-place votes
will be eliminated.

.. code:: ipython3

    print("fpv after round 2:", election.election_states[2].scores)
    print("go to the next step\n")
    
    
    print("elected", election.election_states[3].elected)
    print("\neliminated", election.election_states[3].eliminated)
    print("\nremaining", election.election_states[3].remaining)
    print("\ntiebreak resolution", election.election_states[3].tiebreaks)
    print()
    print(election.get_profile(3).df)


.. parsed-literal::

    fpv after round 2: {'G': np.float64(3.0), 'A': np.float64(3.0), 'C': np.float64(3.0), 'D': np.float64(4.0), 'F': np.float64(4.0)}
    go to the next step
    
    elected (frozenset(),)
    
    eliminated (frozenset({'C'}),)
    
    remaining (frozenset({'D'}), frozenset({'A', 'F'}), frozenset({'G'}))
    
    tiebreak resolution {frozenset({'A', 'C', 'G'}): (frozenset({'A'}), frozenset({'G'}), frozenset({'C'}))}
    
    Initial tiebreak was unsuccessful, performing random tiebreak
                 Ranking_1 Ranking_2 Ranking_3 Voter Set  Weight
    Ballot Index                                                
    0                  (D)       (~)       (~)        {}     2.0
    1                  (D)       (F)       (~)        {}     1.0
    2                  (G)       (F)       (~)        {}     3.0
    3                  (A)       (~)       (~)        {}     3.0
    4                  (A)       (~)       (~)        {}     1.0
    5                  (D)       (~)       (~)        {}     3.0
    6                  (F)       (G)       (~)        {}     4.0


Note that here, several candidates were tied for the fewest first-place
votes at this stage. When this happens in STV, you use the first-place
votes from the original profile to break ties. This means C will be
eliminated. The ``tiebreaks`` parameter records the resolution of the
tie; since we are looking for the person with the least first-place
votes, the candidate in the final entry of the tuple is eliminated. The
reason the message “Initial tiebreak was unsuccessful, performing random
tiebreak” appeared is that A and G were tied by first-place votes, and
thus a random tiebreak was needed to separate them. This didn’t affect
the outcome, since C had the fewest first-place votes.

**Try it yourself**
~~~~~~~~~~~~~~~~~~~

   Keep printing the first-place votes and running a step of the
   election until all seats have been filled. At each step, think
   through why the election state transitioned as it did.

We now change the transfer type. Using the same profile as above, we’ll
now use ``random_transfer``. In the default fractional transfer, we
reweighted all of the ballots in proportion to the surplus. Here, we
will randomly choose the appropriate number of ballots to transfer (the
same number as the surplus). Though it sounds strange, this is the
method actually used in Cambridge, MA. (Recall that Cambridge has used
STV continuously since 1941 so back in the day they probably needed a
low-tech physical way to do the transfers.)

.. code:: ipython3

    from votekit.elections import random_transfer
    
    candidates = ["A", "B", "C", "D", "E", "F", "G"]
    
    ballots = [
        Ballot(ranking=[{"A"}, {"B"}], weight=3),
        Ballot(ranking=[{"B"}, {"C"}, {"D"}], weight=8),
        Ballot(ranking=[{"B"}, {"D"}, {"C"}], weight=8),
        Ballot(ranking=[{"C"}, {"A"}, {"B"}], weight=1),
        Ballot(ranking=[{"D"}, {"E"}], weight=1),
        Ballot(ranking=[{"E"}, {"D"}, {"F"}], weight=1),
        Ballot(ranking=[{"F"}, {"G"}], weight=4),
        Ballot(ranking=[{"G"}, {"E"}, {"F"}], weight=1),
    ]
    
    profile = PreferenceProfile(ballots=ballots)
    
    print(profile.df)
    print("Sum of ballot weights:", profile.total_ballot_wt)
    print("Number of candidates:", len(profile.candidates))
    print()
    
    election = STV(profile=profile, transfer=random_transfer, m=2)
    
    print(election)


.. parsed-literal::

                 Ranking_1 Ranking_2 Ranking_3 Voter Set  Weight
    Ballot Index                                                
    0                  (A)       (B)       (~)        {}     3.0
    1                  (B)       (C)       (D)        {}     8.0
    2                  (B)       (D)       (C)        {}     8.0
    3                  (C)       (A)       (B)        {}     1.0
    4                  (D)       (E)       (~)        {}     1.0
    5                  (E)       (D)       (F)        {}     1.0
    6                  (F)       (G)       (~)        {}     4.0
    7                  (G)       (E)       (F)        {}     1.0
    Sum of ballot weights: 27.0
    Number of candidates: 7
    
    Initial tiebreak was unsuccessful, performing random tiebreak
           Status  Round
    B     Elected      1
    D     Elected      7
    F  Eliminated      6
    C  Eliminated      5
    A  Eliminated      4
    G  Eliminated      3
    E  Eliminated      2


**Try it yourself**
~~~~~~~~~~~~~~~~~~~

   Rerun the code above until you see that different candidates can win
   under random transfer.

Election
--------

Let’s poke around the ``Election`` class a bit more. It contains a lot
of useful information about what is happening in an election. We will
also introduce the Borda election.

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
    
    candidates = ["A", "B", "C", "D", "E", "F"]
    
    # recall IAC generates an "all bets are off" profile
    iac = bg.ImpartialAnonymousCulture(candidates=candidates)
    profile = iac.generate_profile(number_of_ballots=1000)
    
    election = Borda(profile, m=3)

.. code:: ipython3

    print(election.get_profile(0).df.head(10).to_string())
    print()
    
    print(election)


.. parsed-literal::

                 Ranking_1 Ranking_2 Ranking_3 Ranking_4 Ranking_5 Ranking_6 Voter Set  Weight
    Ballot Index                                                                              
    0                  (B)       (E)       (F)       (C)       (D)       (A)        {}     3.0
    1                  (E)       (D)       (F)       (B)       (C)       (A)        {}     1.0
    2                  (A)       (B)       (C)       (E)       (F)       (D)        {}     3.0
    3                  (B)       (C)       (F)       (E)       (A)       (D)        {}     1.0
    4                  (A)       (E)       (B)       (C)       (F)       (D)        {}     2.0
    5                  (E)       (B)       (D)       (F)       (C)       (A)        {}     1.0
    6                  (B)       (A)       (E)       (F)       (C)       (D)        {}     1.0
    7                  (F)       (A)       (B)       (E)       (D)       (C)        {}     3.0
    8                  (C)       (F)       (A)       (B)       (E)       (D)        {}     6.0
    9                  (D)       (B)       (C)       (E)       (A)       (F)        {}     2.0
    
          Status  Round
    A    Elected      1
    F    Elected      1
    B    Elected      1
    D  Remaining      1
    C  Remaining      1
    E  Remaining      1


The Borda election is one-shot (like plurality), so running a step or
the election is equivalent. Let’s see what the election stores.

.. code:: ipython3

    # the winners up to the given round, -1 means final round
    print("Winners:", election.get_elected(-1))
    
    # the eliminated candidates up to the given round
    print("Eliminated:", election.get_eliminated(-1))
    
    # the ranking of the candidates up to the given round
    print("Ranking:", election.get_ranking(-1))
    
    # the outcome of the given round
    print("Outcome of round 1:\n", election.get_status_df(1))


.. parsed-literal::

    Winners: (frozenset({'A'}), frozenset({'F'}), frozenset({'B'}))
    Eliminated: ()
    Ranking: (frozenset({'A'}), frozenset({'F'}), frozenset({'B'}), frozenset({'D'}), frozenset({'C'}), frozenset({'E'}))
    Outcome of round 1:
           Status  Round
    A    Elected      1
    F    Elected      1
    B    Elected      1
    D  Remaining      1
    C  Remaining      1
    E  Remaining      1


**Try it yourself**
~~~~~~~~~~~~~~~~~~~

   Using the following preference profile, try changing the score vector
   of a Borda election. Try replacing 3,2,1 with other Borda weights
   (decreasing and non-negative) showing that each candidate can be
   elected.

.. code:: ipython3

    ballots = [
        Ballot(ranking=[{"A"}, {"B"}, {"C"}], weight=3),
        Ballot(ranking=[{"A"}, {"C"}, {"B"}], weight=2),
        Ballot(ranking=[{"B"}, {"C"}, {"A"}], weight=2),
        Ballot(ranking=[{"C"}, {"B"}, {"A"}], weight=4),
    ]
    
    profile = PreferenceProfile(ballots=ballots, candidates=["A", "B", "C"])
    
    # borda election
    score_vector = [3, 2, 1]
    election = Borda(profile, m=1, score_vector=score_vector)
    print(election)


.. parsed-literal::

          Status  Round
    C    Elected      1
    B  Remaining      1
    A  Remaining      1


Since a Borda election is a one-shot election, most of the information
stored in the ``Election`` is extraneous, but you can see its utility in
an STV election where there are many rounds.

.. code:: ipython3

    minneapolis_profile = load_ranking_csv("mn_2013_cast_vote_record.csv", rank_cols=[0,1,2], header_row=0)
    minneapolis_profile = condense_profile(remove_repeated_candidates(remove_cand(["undervote", "overvote", "UWI"], minneapolis_profile)))
    
    minn_election = STV(profile=minneapolis_profile, m=1)
    
    for i in range(1, 6):
        print(f"Round {i}\n")
        # the winners up to the current round
        print("Winners:", minn_election.get_elected(i))
    
        # the eliminated candidates up to the current round
        print("Eliminated:", minn_election.get_eliminated(i))
    
        # the remaining candidates, sorted by first-place votes
        print("Remaining:", minn_election.get_remaining(i))
    
        # the same information as a df
        print(minn_election.get_status_df(i))
    
        print()


.. parsed-literal::

    Profile contains rankings: True
    Maximum ranking length: 3
    Profile contains scores: False
    Candidates: ('KURTIS W. HANNA', 'undervote', 'STEPHANIE WOODRUFF', 'GREGG A. IVERSON', 'JOHN CHARLES WILSON', 'JOSHUA REA', 'JACKIE CHERRYHOMES', 'BILL KAHN', 'CHRISTOPHER ROBIN ZIMMERMAN', 'CHRISTOPHER CLARK', 'MARK V ANDERSON', 'JAMES EVERETT', 'JAMES "JIMMY" L. STROUD, JR.', 'NEAL BAXTER', 'EDMUND BERNARD BRUYERE', 'overvote', 'MARK ANDREW', 'UWI', 'CAPTAIN JACK SPARROW', 'ALICIA K. BENNETT', 'CYD GORMAN', 'BOB "AGAIN" CARNEY JR', 'JEFFREY ALAN WAGNER', 'JOHN LESLIE HARTWIG', 'OLE SAVIOR', 'TROY BENJEGERDES', 'TONY LANE', 'CAM WINTON', 'BETSY HODGES', 'DAN COHEN', 'DON SAMUELS', 'BOB FINE', 'RAHN V. WORKCUFF', 'MERRILL ANDERSON', 'ABDUL M RAHAMAN "THE ROCK"', 'JAYMIE KELLY', 'MIKE GOULD', 'DOUG MANN')
    Candidates who received votes: ('KURTIS W. HANNA', 'undervote', 'STEPHANIE WOODRUFF', 'GREGG A. IVERSON', 'JOHN CHARLES WILSON', 'JOSHUA REA', 'JACKIE CHERRYHOMES', 'BILL KAHN', 'CHRISTOPHER ROBIN ZIMMERMAN', 'CHRISTOPHER CLARK', 'MARK V ANDERSON', 'JAMES EVERETT', 'JAMES "JIMMY" L. STROUD, JR.', 'NEAL BAXTER', 'EDMUND BERNARD BRUYERE', 'overvote', 'MARK ANDREW', 'UWI', 'CAPTAIN JACK SPARROW', 'ALICIA K. BENNETT', 'CYD GORMAN', 'BOB "AGAIN" CARNEY JR', 'JEFFREY ALAN WAGNER', 'JOHN LESLIE HARTWIG', 'OLE SAVIOR', 'TROY BENJEGERDES', 'TONY LANE', 'CAM WINTON', 'BETSY HODGES', 'DAN COHEN', 'DON SAMUELS', 'BOB FINE', 'RAHN V. WORKCUFF', 'MERRILL ANDERSON', 'ABDUL M RAHAMAN "THE ROCK"', 'JAYMIE KELLY', 'MIKE GOULD', 'DOUG MANN')
    Total number of Ballot objects: 80101
    Total weight of Ballot objects: 80101.0
    
    Round 1
    
    Winners: ()
    Eliminated: (frozenset({'JOHN CHARLES WILSON'}),)
    Remaining: (frozenset({'BETSY HODGES'}), frozenset({'MARK ANDREW'}), frozenset({'DON SAMUELS'}), frozenset({'CAM WINTON'}), frozenset({'JACKIE CHERRYHOMES'}), frozenset({'BOB FINE'}), frozenset({'DAN COHEN'}), frozenset({'STEPHANIE WOODRUFF'}), frozenset({'MARK V ANDERSON'}), frozenset({'DOUG MANN'}), frozenset({'OLE SAVIOR'}), frozenset({'ABDUL M RAHAMAN "THE ROCK"'}), frozenset({'ALICIA K. BENNETT'}), frozenset({'JAMES EVERETT'}), frozenset({'CAPTAIN JACK SPARROW'}), frozenset({'TONY LANE'}), frozenset({'MIKE GOULD'}), frozenset({'KURTIS W. HANNA'}), frozenset({'JAYMIE KELLY'}), frozenset({'CHRISTOPHER CLARK'}), frozenset({'CHRISTOPHER ROBIN ZIMMERMAN'}), frozenset({'JEFFREY ALAN WAGNER'}), frozenset({'TROY BENJEGERDES'}), frozenset({'NEAL BAXTER', 'GREGG A. IVERSON'}), frozenset({'JOSHUA REA'}), frozenset({'MERRILL ANDERSON'}), frozenset({'BILL KAHN'}), frozenset({'JOHN LESLIE HARTWIG'}), frozenset({'EDMUND BERNARD BRUYERE'}), frozenset({'RAHN V. WORKCUFF', 'JAMES "JIMMY" L. STROUD, JR.'}), frozenset({'BOB "AGAIN" CARNEY JR'}), frozenset({'CYD GORMAN'}))
                                      Status  Round
    BETSY HODGES                   Remaining      1
    MARK ANDREW                    Remaining      1
    DON SAMUELS                    Remaining      1
    CAM WINTON                     Remaining      1
    JACKIE CHERRYHOMES             Remaining      1
    BOB FINE                       Remaining      1
    DAN COHEN                      Remaining      1
    STEPHANIE WOODRUFF             Remaining      1
    MARK V ANDERSON                Remaining      1
    DOUG MANN                      Remaining      1
    OLE SAVIOR                     Remaining      1
    ABDUL M RAHAMAN "THE ROCK"     Remaining      1
    ALICIA K. BENNETT              Remaining      1
    JAMES EVERETT                  Remaining      1
    CAPTAIN JACK SPARROW           Remaining      1
    TONY LANE                      Remaining      1
    MIKE GOULD                     Remaining      1
    KURTIS W. HANNA                Remaining      1
    JAYMIE KELLY                   Remaining      1
    CHRISTOPHER CLARK              Remaining      1
    CHRISTOPHER ROBIN ZIMMERMAN    Remaining      1
    JEFFREY ALAN WAGNER            Remaining      1
    TROY BENJEGERDES               Remaining      1
    NEAL BAXTER                    Remaining      1
    GREGG A. IVERSON               Remaining      1
    JOSHUA REA                     Remaining      1
    MERRILL ANDERSON               Remaining      1
    BILL KAHN                      Remaining      1
    JOHN LESLIE HARTWIG            Remaining      1
    EDMUND BERNARD BRUYERE         Remaining      1
    RAHN V. WORKCUFF               Remaining      1
    JAMES "JIMMY" L. STROUD, JR.   Remaining      1
    BOB "AGAIN" CARNEY JR          Remaining      1
    CYD GORMAN                     Remaining      1
    JOHN CHARLES WILSON           Eliminated      1
    
    Round 2
    
    Winners: ()
    Eliminated: (frozenset({'CYD GORMAN'}), frozenset({'JOHN CHARLES WILSON'}))
    Remaining: (frozenset({'BETSY HODGES'}), frozenset({'MARK ANDREW'}), frozenset({'DON SAMUELS'}), frozenset({'CAM WINTON'}), frozenset({'JACKIE CHERRYHOMES'}), frozenset({'BOB FINE'}), frozenset({'DAN COHEN'}), frozenset({'STEPHANIE WOODRUFF'}), frozenset({'MARK V ANDERSON'}), frozenset({'DOUG MANN'}), frozenset({'OLE SAVIOR'}), frozenset({'ABDUL M RAHAMAN "THE ROCK"'}), frozenset({'ALICIA K. BENNETT'}), frozenset({'JAMES EVERETT'}), frozenset({'CAPTAIN JACK SPARROW'}), frozenset({'TONY LANE'}), frozenset({'MIKE GOULD'}), frozenset({'KURTIS W. HANNA'}), frozenset({'JAYMIE KELLY'}), frozenset({'CHRISTOPHER CLARK'}), frozenset({'CHRISTOPHER ROBIN ZIMMERMAN'}), frozenset({'JEFFREY ALAN WAGNER'}), frozenset({'TROY BENJEGERDES'}), frozenset({'GREGG A. IVERSON'}), frozenset({'NEAL BAXTER'}), frozenset({'JOSHUA REA'}), frozenset({'MERRILL ANDERSON'}), frozenset({'BILL KAHN'}), frozenset({'JOHN LESLIE HARTWIG'}), frozenset({'EDMUND BERNARD BRUYERE'}), frozenset({'RAHN V. WORKCUFF', 'JAMES "JIMMY" L. STROUD, JR.'}), frozenset({'BOB "AGAIN" CARNEY JR'}))
                                      Status  Round
    BETSY HODGES                   Remaining      2
    MARK ANDREW                    Remaining      2
    DON SAMUELS                    Remaining      2
    CAM WINTON                     Remaining      2
    JACKIE CHERRYHOMES             Remaining      2
    BOB FINE                       Remaining      2
    DAN COHEN                      Remaining      2
    STEPHANIE WOODRUFF             Remaining      2
    MARK V ANDERSON                Remaining      2
    DOUG MANN                      Remaining      2
    OLE SAVIOR                     Remaining      2
    ABDUL M RAHAMAN "THE ROCK"     Remaining      2
    ALICIA K. BENNETT              Remaining      2
    JAMES EVERETT                  Remaining      2
    CAPTAIN JACK SPARROW           Remaining      2
    TONY LANE                      Remaining      2
    MIKE GOULD                     Remaining      2
    KURTIS W. HANNA                Remaining      2
    JAYMIE KELLY                   Remaining      2
    CHRISTOPHER CLARK              Remaining      2
    CHRISTOPHER ROBIN ZIMMERMAN    Remaining      2
    JEFFREY ALAN WAGNER            Remaining      2
    TROY BENJEGERDES               Remaining      2
    GREGG A. IVERSON               Remaining      2
    NEAL BAXTER                    Remaining      2
    JOSHUA REA                     Remaining      2
    MERRILL ANDERSON               Remaining      2
    BILL KAHN                      Remaining      2
    JOHN LESLIE HARTWIG            Remaining      2
    EDMUND BERNARD BRUYERE         Remaining      2
    RAHN V. WORKCUFF               Remaining      2
    JAMES "JIMMY" L. STROUD, JR.   Remaining      2
    BOB "AGAIN" CARNEY JR          Remaining      2
    CYD GORMAN                    Eliminated      2
    JOHN CHARLES WILSON           Eliminated      1
    
    Round 3
    
    Winners: ()
    Eliminated: (frozenset({'BOB "AGAIN" CARNEY JR'}), frozenset({'CYD GORMAN'}), frozenset({'JOHN CHARLES WILSON'}))
    Remaining: (frozenset({'BETSY HODGES'}), frozenset({'MARK ANDREW'}), frozenset({'DON SAMUELS'}), frozenset({'CAM WINTON'}), frozenset({'JACKIE CHERRYHOMES'}), frozenset({'BOB FINE'}), frozenset({'DAN COHEN'}), frozenset({'STEPHANIE WOODRUFF'}), frozenset({'MARK V ANDERSON'}), frozenset({'DOUG MANN'}), frozenset({'OLE SAVIOR'}), frozenset({'ABDUL M RAHAMAN "THE ROCK"'}), frozenset({'ALICIA K. BENNETT'}), frozenset({'JAMES EVERETT'}), frozenset({'CAPTAIN JACK SPARROW'}), frozenset({'TONY LANE'}), frozenset({'MIKE GOULD'}), frozenset({'KURTIS W. HANNA'}), frozenset({'JAYMIE KELLY'}), frozenset({'CHRISTOPHER CLARK'}), frozenset({'CHRISTOPHER ROBIN ZIMMERMAN'}), frozenset({'JEFFREY ALAN WAGNER'}), frozenset({'TROY BENJEGERDES'}), frozenset({'GREGG A. IVERSON'}), frozenset({'NEAL BAXTER'}), frozenset({'JOSHUA REA', 'MERRILL ANDERSON'}), frozenset({'BILL KAHN'}), frozenset({'JOHN LESLIE HARTWIG'}), frozenset({'EDMUND BERNARD BRUYERE'}), frozenset({'JAMES "JIMMY" L. STROUD, JR.'}), frozenset({'RAHN V. WORKCUFF'}))
                                      Status  Round
    BETSY HODGES                   Remaining      3
    MARK ANDREW                    Remaining      3
    DON SAMUELS                    Remaining      3
    CAM WINTON                     Remaining      3
    JACKIE CHERRYHOMES             Remaining      3
    BOB FINE                       Remaining      3
    DAN COHEN                      Remaining      3
    STEPHANIE WOODRUFF             Remaining      3
    MARK V ANDERSON                Remaining      3
    DOUG MANN                      Remaining      3
    OLE SAVIOR                     Remaining      3
    ABDUL M RAHAMAN "THE ROCK"     Remaining      3
    ALICIA K. BENNETT              Remaining      3
    JAMES EVERETT                  Remaining      3
    CAPTAIN JACK SPARROW           Remaining      3
    TONY LANE                      Remaining      3
    MIKE GOULD                     Remaining      3
    KURTIS W. HANNA                Remaining      3
    JAYMIE KELLY                   Remaining      3
    CHRISTOPHER CLARK              Remaining      3
    CHRISTOPHER ROBIN ZIMMERMAN    Remaining      3
    JEFFREY ALAN WAGNER            Remaining      3
    TROY BENJEGERDES               Remaining      3
    GREGG A. IVERSON               Remaining      3
    NEAL BAXTER                    Remaining      3
    JOSHUA REA                     Remaining      3
    MERRILL ANDERSON               Remaining      3
    BILL KAHN                      Remaining      3
    JOHN LESLIE HARTWIG            Remaining      3
    EDMUND BERNARD BRUYERE         Remaining      3
    JAMES "JIMMY" L. STROUD, JR.   Remaining      3
    RAHN V. WORKCUFF               Remaining      3
    BOB "AGAIN" CARNEY JR         Eliminated      3
    CYD GORMAN                    Eliminated      2
    JOHN CHARLES WILSON           Eliminated      1
    
    Round 4
    
    Winners: ()
    Eliminated: (frozenset({'RAHN V. WORKCUFF'}), frozenset({'BOB "AGAIN" CARNEY JR'}), frozenset({'CYD GORMAN'}), frozenset({'JOHN CHARLES WILSON'}))
    Remaining: (frozenset({'BETSY HODGES'}), frozenset({'MARK ANDREW'}), frozenset({'DON SAMUELS'}), frozenset({'CAM WINTON'}), frozenset({'JACKIE CHERRYHOMES'}), frozenset({'BOB FINE'}), frozenset({'DAN COHEN'}), frozenset({'STEPHANIE WOODRUFF'}), frozenset({'MARK V ANDERSON'}), frozenset({'DOUG MANN'}), frozenset({'OLE SAVIOR'}), frozenset({'JAMES EVERETT', 'ABDUL M RAHAMAN "THE ROCK"'}), frozenset({'ALICIA K. BENNETT'}), frozenset({'CAPTAIN JACK SPARROW'}), frozenset({'TONY LANE'}), frozenset({'MIKE GOULD'}), frozenset({'KURTIS W. HANNA'}), frozenset({'JAYMIE KELLY'}), frozenset({'CHRISTOPHER CLARK'}), frozenset({'CHRISTOPHER ROBIN ZIMMERMAN'}), frozenset({'JEFFREY ALAN WAGNER'}), frozenset({'NEAL BAXTER'}), frozenset({'TROY BENJEGERDES'}), frozenset({'GREGG A. IVERSON'}), frozenset({'JOSHUA REA'}), frozenset({'MERRILL ANDERSON'}), frozenset({'BILL KAHN'}), frozenset({'JOHN LESLIE HARTWIG'}), frozenset({'EDMUND BERNARD BRUYERE'}), frozenset({'JAMES "JIMMY" L. STROUD, JR.'}))
                                      Status  Round
    BETSY HODGES                   Remaining      4
    MARK ANDREW                    Remaining      4
    DON SAMUELS                    Remaining      4
    CAM WINTON                     Remaining      4
    JACKIE CHERRYHOMES             Remaining      4
    BOB FINE                       Remaining      4
    DAN COHEN                      Remaining      4
    STEPHANIE WOODRUFF             Remaining      4
    MARK V ANDERSON                Remaining      4
    DOUG MANN                      Remaining      4
    OLE SAVIOR                     Remaining      4
    JAMES EVERETT                  Remaining      4
    ABDUL M RAHAMAN "THE ROCK"     Remaining      4
    ALICIA K. BENNETT              Remaining      4
    CAPTAIN JACK SPARROW           Remaining      4
    TONY LANE                      Remaining      4
    MIKE GOULD                     Remaining      4
    KURTIS W. HANNA                Remaining      4
    JAYMIE KELLY                   Remaining      4
    CHRISTOPHER CLARK              Remaining      4
    CHRISTOPHER ROBIN ZIMMERMAN    Remaining      4
    JEFFREY ALAN WAGNER            Remaining      4
    NEAL BAXTER                    Remaining      4
    TROY BENJEGERDES               Remaining      4
    GREGG A. IVERSON               Remaining      4
    JOSHUA REA                     Remaining      4
    MERRILL ANDERSON               Remaining      4
    BILL KAHN                      Remaining      4
    JOHN LESLIE HARTWIG            Remaining      4
    EDMUND BERNARD BRUYERE         Remaining      4
    JAMES "JIMMY" L. STROUD, JR.   Remaining      4
    RAHN V. WORKCUFF              Eliminated      4
    BOB "AGAIN" CARNEY JR         Eliminated      3
    CYD GORMAN                    Eliminated      2
    JOHN CHARLES WILSON           Eliminated      1
    
    Round 5
    
    Winners: ()
    Eliminated: (frozenset({'JAMES "JIMMY" L. STROUD, JR.'}), frozenset({'RAHN V. WORKCUFF'}), frozenset({'BOB "AGAIN" CARNEY JR'}), frozenset({'CYD GORMAN'}), frozenset({'JOHN CHARLES WILSON'}))
    Remaining: (frozenset({'BETSY HODGES'}), frozenset({'MARK ANDREW'}), frozenset({'DON SAMUELS'}), frozenset({'CAM WINTON'}), frozenset({'JACKIE CHERRYHOMES'}), frozenset({'BOB FINE'}), frozenset({'DAN COHEN'}), frozenset({'STEPHANIE WOODRUFF'}), frozenset({'MARK V ANDERSON'}), frozenset({'DOUG MANN'}), frozenset({'OLE SAVIOR'}), frozenset({'ABDUL M RAHAMAN "THE ROCK"'}), frozenset({'ALICIA K. BENNETT'}), frozenset({'JAMES EVERETT'}), frozenset({'CAPTAIN JACK SPARROW'}), frozenset({'TONY LANE'}), frozenset({'MIKE GOULD'}), frozenset({'JAYMIE KELLY'}), frozenset({'KURTIS W. HANNA'}), frozenset({'CHRISTOPHER CLARK'}), frozenset({'CHRISTOPHER ROBIN ZIMMERMAN'}), frozenset({'JEFFREY ALAN WAGNER'}), frozenset({'NEAL BAXTER'}), frozenset({'TROY BENJEGERDES'}), frozenset({'GREGG A. IVERSON'}), frozenset({'MERRILL ANDERSON'}), frozenset({'JOSHUA REA'}), frozenset({'BILL KAHN'}), frozenset({'JOHN LESLIE HARTWIG'}), frozenset({'EDMUND BERNARD BRUYERE'}))
                                      Status  Round
    BETSY HODGES                   Remaining      5
    MARK ANDREW                    Remaining      5
    DON SAMUELS                    Remaining      5
    CAM WINTON                     Remaining      5
    JACKIE CHERRYHOMES             Remaining      5
    BOB FINE                       Remaining      5
    DAN COHEN                      Remaining      5
    STEPHANIE WOODRUFF             Remaining      5
    MARK V ANDERSON                Remaining      5
    DOUG MANN                      Remaining      5
    OLE SAVIOR                     Remaining      5
    ABDUL M RAHAMAN "THE ROCK"     Remaining      5
    ALICIA K. BENNETT              Remaining      5
    JAMES EVERETT                  Remaining      5
    CAPTAIN JACK SPARROW           Remaining      5
    TONY LANE                      Remaining      5
    MIKE GOULD                     Remaining      5
    JAYMIE KELLY                   Remaining      5
    KURTIS W. HANNA                Remaining      5
    CHRISTOPHER CLARK              Remaining      5
    CHRISTOPHER ROBIN ZIMMERMAN    Remaining      5
    JEFFREY ALAN WAGNER            Remaining      5
    NEAL BAXTER                    Remaining      5
    TROY BENJEGERDES               Remaining      5
    GREGG A. IVERSON               Remaining      5
    MERRILL ANDERSON               Remaining      5
    JOSHUA REA                     Remaining      5
    BILL KAHN                      Remaining      5
    JOHN LESLIE HARTWIG            Remaining      5
    EDMUND BERNARD BRUYERE         Remaining      5
    JAMES "JIMMY" L. STROUD, JR.  Eliminated      5
    RAHN V. WORKCUFF              Eliminated      4
    BOB "AGAIN" CARNEY JR         Eliminated      3
    CYD GORMAN                    Eliminated      2
    JOHN CHARLES WILSON           Eliminated      1
    


Conclusion
----------

There are many different possible election methods, both for choosing a
single seat or multiple seats. ``VoteKit`` has a host of built-in
election methods, as well as the functionality to let you create your
own kind of election. You have been introduced to the STV and Borda
elections and learned about the ``Election`` object. This should allow
you to model any kind of elections you see in the real world, including
rules that have not yet been implemented in ``VoteKit``.

Further Prompts: Creating your own election system
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``VoteKit`` can’t be comprehensive in terms of possible election rules.
However, with the ``Election`` and ``ElectionState`` classes, you can
create your own. Let’s create a bit of a silly example; to elect
:math:`m` seats, at each stage of the election we randomly choose one
candidate to elect. Most of the methods are handled by the
``RankingElection`` class, so we really only need to define how a step
works, and how to know when it’s over.

.. code:: ipython3

    from votekit.elections import RankingElection, ElectionState
    from votekit.cleaning import remove_cand
    import random
    
    
    class RandomWinners(RankingElection):
        """
        Simulates an election where we randomly choose winners at each stage.
    
        Args:
            profile (PreferenceProfile): Profile to run election on.
            m (int, optional): Number of seats to elect.
        """
    
        def __init__(self, profile: PreferenceProfile, m: int = 1):
            # the super method says call the RankingElection class
            self.m = m
            super().__init__(profile)
    
        def _is_finished(self) -> bool:
            """
            Determines if another round is needed.
    
            Returns:
                bool: True if number of seats has been met, False otherwise.
            """
            # need to unpack list of sets
            elected = [c for s in self.get_elected() for c in s]
    
            if len(elected) == self.m:
                return True
    
            return False
    
        def _run_step(
            self, profile: PreferenceProfile, prev_state: ElectionState, store_states=False
        ) -> PreferenceProfile:
            """
            Run one step of an election from the given profile and previous state.
    
            Args:
                profile (PreferenceProfile): Profile of ballots.
                prev_state (ElectionState): The previous ElectionState.
                store_states (bool, optional): True if `self.election_states` should be updated with the
                    ElectionState generated by this round. This should only be True when used by
                    `self._run_election()`. Defaults to False.
    
            Returns:
                PreferenceProfile: The profile of ballots after the round is completed.
            """
    
            elected_cand = random.choice(profile.candidates)
            new_profile = remove_cand(elected_cand, profile)
    
            # we only store the states the first time an election is run,
            # but this is all handled by the other methods of the class
            if store_states:
                self.election_states.append(
                    ElectionState(
                        round_number=prev_state.round_number + 1,
                        remaining=(frozenset(new_profile.candidates),),
                        elected=(frozenset(elected_cand),),
                    )
                )
    
            return new_profile

.. code:: ipython3

    candidates = ["A", "B", "C", "D", "E", "F"]
    profile = bg.ImpartialCulture(candidates=candidates).generate_profile(1000)
    
    election = RandomWinners(profile=profile, m=3)

.. code:: ipython3

    print(election)


.. parsed-literal::

          Status  Round
    F    Elected      1
    B    Elected      2
    C    Elected      3
    A  Remaining      3
    E  Remaining      3
    D  Remaining      3


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

    class AlphabeticaElection(RankingElection):
        """
        Simulates an election where we choose winners alphabetically at each stage.
    
        Args:
            profile (PreferenceProfile): Profile to run election on.
            m (int, optional): Number of seats to elect.
        """
    
        def __init__(self, profile: PreferenceProfile, m: int = 1):
            # the super method says call the RankingElection class
            self.m = m
            super().__init__(profile)
    
        def _is_finished(self) -> bool:
            """
            Determines if another round is needed.
    
            Returns:
                bool: True if number of seats has been met, False otherwise.
            """
            # need to unpack list of sets
            elected = [c for s in self.get_elected() for c in s]
    
            if len(elected) == self.m:
                return True
    
            return False
    
        def _run_step(
            self, profile: PreferenceProfile, prev_state: ElectionState, store_states=False
        ) -> PreferenceProfile:
            """
            Run one step of an election from the given profile and previous state.
    
            Args:
                profile (PreferenceProfile): Profile of ballots.
                prev_state (ElectionState): The previous ElectionState.
                store_states (bool, optional): True if `self.election_states` should be updated with the
                    ElectionState generated by this round. This should only be True when used by
                    `self._run_election()`. Defaults to False.
    
            Returns:
                PreferenceProfile: The profile of ballots after the round is completed.
            """
    
            pass
