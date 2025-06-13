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

    from votekit.cvr_loaders import load_csv
    from votekit.elections import STV
    from votekit.cleaning import remove_and_condense
    
    minneapolis_profile = load_csv("mn_2013_cast_vote_record.csv")
    minneapolis_profile = remove_and_condense(["undervote", "overvote", "UWI"], minneapolis_profile)
    
    # m = 1 means 1 seat
    minn_election = STV(profile=minneapolis_profile, m=1)
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

    {'A': 3.0, 'B': 8.0, 'C': 1.0, 'D': 3.0, 'E': 1.0, 'F': 4.0, 'G': 3.0}



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
    
    remaining (frozenset({'F'}), frozenset({'C', 'G', 'A', 'D'}), frozenset({'E'}))


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
          <td>(G)</td>
          <td>(E)</td>
          <td>(F)</td>
          <td>{}</td>
          <td>3.0</td>
        </tr>
        <tr>
          <th>2</th>
          <td>(C)</td>
          <td>(A)</td>
          <td>(~)</td>
          <td>{}</td>
          <td>1.0</td>
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
          <td>(F)</td>
          <td>(G)</td>
          <td>(~)</td>
          <td>{}</td>
          <td>4.0</td>
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
          <td>(E)</td>
          <td>(D)</td>
          <td>(F)</td>
          <td>{}</td>
          <td>1.0</td>
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

    fpv after round 1: {'C': 3.0, 'D': 3.0, 'G': 3.0, 'E': 1.0, 'F': 4.0, 'A': 3.0}
    go to the next step
    
    elected (frozenset(),)
    
    eliminated (frozenset({'E'}),)
    
    remaining (frozenset({'F', 'D'}), frozenset({'C', 'G', 'A'}))
                 Ranking_1 Ranking_2 Ranking_3 Voter Set  Weight
    Ballot Index                                                
    0                  (C)       (D)       (~)        {}     2.0
    1                  (G)       (F)       (~)        {}     3.0
    2                  (C)       (A)       (~)        {}     1.0
    3                  (A)       (~)       (~)        {}     3.0
    4                  (F)       (G)       (~)        {}     4.0
    5                  (D)       (~)       (~)        {}     3.0
    6                  (D)       (F)       (~)        {}     1.0


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

    fpv after round 2: {'C': 3.0, 'G': 3.0, 'A': 3.0, 'F': 4.0, 'D': 4.0}
    go to the next step
    
    elected (frozenset(),)
    
    eliminated (frozenset({'C'}),)
    
    remaining (frozenset({'D'}), frozenset({'F', 'A'}), frozenset({'G'}))
    
    tiebreak resolution {frozenset({'C', 'G', 'A'}): (frozenset({'A'}), frozenset({'G'}), frozenset({'C'}))}
    
    Initial tiebreak was unsuccessful, performing random tiebreak
                 Ranking_1 Ranking_2 Ranking_3 Voter Set  Weight
    Ballot Index                                                
    0                  (D)       (~)       (~)        {}     2.0
    1                  (G)       (F)       (~)        {}     3.0
    2                  (A)       (~)       (~)        {}     1.0
    3                  (A)       (~)       (~)        {}     3.0
    4                  (F)       (G)       (~)        {}     4.0
    5                  (D)       (~)       (~)        {}     3.0
    6                  (D)       (F)       (~)        {}     1.0


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
    A  Eliminated      5
    C  Eliminated      4
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

    print(election.get_profile(0).df.to_string())
    print()
    
    print(election)


.. parsed-literal::

                 Ranking_1 Ranking_2 Ranking_3 Ranking_4 Ranking_5 Ranking_6 Voter Set  Weight
    Ballot Index                                                                              
    0                  (E)       (F)       (B)       (A)       (D)       (C)        {}    13.0
    1                  (C)       (A)       (F)       (B)       (E)       (D)        {}     4.0
    2                  (F)       (D)       (A)       (E)       (B)       (C)        {}     3.0
    3                  (B)       (F)       (E)       (C)       (A)       (D)        {}     4.0
    4                  (F)       (E)       (B)       (C)       (D)       (A)        {}     3.0
    5                  (A)       (D)       (C)       (E)       (F)       (B)        {}     4.0
    6                  (F)       (D)       (C)       (A)       (E)       (B)        {}     6.0
    7                  (E)       (A)       (D)       (B)       (F)       (C)        {}     2.0
    8                  (D)       (E)       (C)       (B)       (F)       (A)        {}     2.0
    9                  (B)       (C)       (A)       (F)       (E)       (D)        {}     1.0
    10                 (E)       (A)       (B)       (C)       (D)       (F)        {}     5.0
    11                 (F)       (B)       (A)       (D)       (C)       (E)        {}     3.0
    12                 (D)       (F)       (E)       (A)       (C)       (B)        {}     5.0
    13                 (E)       (C)       (D)       (F)       (B)       (A)        {}     1.0
    14                 (A)       (B)       (C)       (F)       (D)       (E)        {}     3.0
    15                 (F)       (C)       (E)       (A)       (B)       (D)        {}     4.0
    16                 (A)       (B)       (E)       (D)       (C)       (F)        {}     2.0
    17                 (F)       (A)       (E)       (B)       (D)       (C)        {}     5.0
    18                 (F)       (E)       (D)       (B)       (A)       (C)        {}     1.0
    19                 (B)       (C)       (F)       (E)       (A)       (D)        {}     5.0
    20                 (B)       (C)       (D)       (A)       (E)       (F)        {}     5.0
    21                 (F)       (E)       (B)       (D)       (C)       (A)        {}     3.0
    22                 (A)       (B)       (D)       (F)       (E)       (C)        {}     1.0
    23                 (D)       (A)       (B)       (C)       (E)       (F)        {}    14.0
    24                 (C)       (E)       (D)       (F)       (B)       (A)        {}     8.0
    25                 (A)       (F)       (B)       (C)       (D)       (E)        {}     3.0
    26                 (A)       (C)       (E)       (F)       (D)       (B)        {}     9.0
    27                 (A)       (C)       (E)       (F)       (B)       (D)        {}     2.0
    28                 (B)       (C)       (D)       (E)       (F)       (A)        {}     4.0
    29                 (B)       (E)       (F)       (A)       (D)       (C)        {}     1.0
    30                 (E)       (F)       (A)       (C)       (D)       (B)        {}     1.0
    31                 (F)       (B)       (D)       (E)       (A)       (C)        {}     7.0
    32                 (A)       (D)       (E)       (C)       (F)       (B)        {}     6.0
    33                 (F)       (D)       (A)       (B)       (E)       (C)        {}     1.0
    34                 (C)       (F)       (D)       (B)       (A)       (E)        {}     2.0
    35                 (D)       (C)       (A)       (B)       (E)       (F)        {}     5.0
    36                 (D)       (A)       (F)       (B)       (E)       (C)        {}     3.0
    37                 (C)       (F)       (B)       (A)       (E)       (D)        {}     2.0
    38                 (C)       (B)       (A)       (F)       (D)       (E)        {}    12.0
    39                 (D)       (F)       (A)       (E)       (B)       (C)        {}     5.0
    40                 (C)       (E)       (A)       (F)       (B)       (D)        {}     2.0
    41                 (E)       (F)       (C)       (B)       (A)       (D)        {}     3.0
    42                 (A)       (F)       (C)       (E)       (B)       (D)        {}     3.0
    43                 (A)       (C)       (D)       (B)       (E)       (F)        {}     4.0
    44                 (E)       (A)       (D)       (F)       (C)       (B)        {}     4.0
    45                 (B)       (A)       (C)       (E)       (F)       (D)        {}     3.0
    46                 (B)       (E)       (F)       (C)       (D)       (A)        {}     1.0
    47                 (E)       (C)       (B)       (F)       (A)       (D)        {}     2.0
    48                 (F)       (B)       (E)       (C)       (D)       (A)        {}     4.0
    49                 (E)       (C)       (D)       (B)       (F)       (A)        {}     5.0
    50                 (A)       (C)       (D)       (B)       (F)       (E)        {}     4.0
    51                 (C)       (B)       (A)       (D)       (F)       (E)        {}     3.0
    52                 (F)       (B)       (D)       (A)       (C)       (E)        {}     1.0
    53                 (E)       (F)       (B)       (C)       (D)       (A)        {}     6.0
    54                 (F)       (A)       (D)       (B)       (C)       (E)        {}     2.0
    55                 (F)       (A)       (D)       (C)       (E)       (B)        {}     7.0
    56                 (B)       (A)       (F)       (C)       (D)       (E)        {}     4.0
    57                 (B)       (F)       (E)       (A)       (D)       (C)        {}     1.0
    58                 (A)       (F)       (E)       (B)       (D)       (C)        {}     9.0
    59                 (E)       (A)       (D)       (C)       (B)       (F)        {}     4.0
    60                 (A)       (F)       (D)       (C)       (B)       (E)        {}     4.0
    61                 (F)       (B)       (C)       (A)       (D)       (E)        {}     1.0
    62                 (B)       (C)       (F)       (D)       (E)       (A)        {}     3.0
    63                 (B)       (A)       (F)       (D)       (C)       (E)        {}     2.0
    64                 (F)       (D)       (B)       (A)       (E)       (C)        {}     1.0
    65                 (F)       (B)       (C)       (D)       (A)       (E)        {}     2.0
    66                 (F)       (B)       (E)       (A)       (C)       (D)        {}     1.0
    67                 (B)       (C)       (A)       (D)       (F)       (E)        {}     2.0
    68                 (C)       (F)       (E)       (B)       (A)       (D)        {}     4.0
    69                 (B)       (E)       (C)       (F)       (A)       (D)        {}     6.0
    70                 (E)       (F)       (C)       (B)       (D)       (A)        {}     4.0
    71                 (F)       (B)       (C)       (E)       (D)       (A)        {}     7.0
    72                 (A)       (F)       (B)       (E)       (D)       (C)        {}     5.0
    73                 (F)       (C)       (B)       (D)       (A)       (E)        {}     3.0
    74                 (E)       (A)       (F)       (D)       (B)       (C)        {}     2.0
    75                 (C)       (D)       (E)       (F)       (A)       (B)        {}     6.0
    76                 (A)       (F)       (B)       (E)       (C)       (D)        {}     2.0
    77                 (A)       (B)       (D)       (E)       (C)       (F)        {}     1.0
    78                 (C)       (D)       (F)       (B)       (A)       (E)        {}     3.0
    79                 (F)       (E)       (D)       (A)       (C)       (B)        {}     1.0
    80                 (E)       (F)       (C)       (D)       (A)       (B)        {}     1.0
    81                 (B)       (F)       (C)       (E)       (A)       (D)        {}     7.0
    82                 (F)       (A)       (D)       (B)       (E)       (C)        {}    13.0
    83                 (D)       (F)       (E)       (C)       (B)       (A)        {}     6.0
    84                 (F)       (A)       (B)       (C)       (E)       (D)        {}     5.0
    85                 (D)       (B)       (A)       (F)       (E)       (C)        {}     7.0
    86                 (D)       (E)       (C)       (F)       (B)       (A)        {}     1.0
    87                 (E)       (A)       (C)       (B)       (D)       (F)        {}     1.0
    88                 (F)       (D)       (E)       (B)       (A)       (C)        {}     3.0
    89                 (A)       (C)       (B)       (F)       (D)       (E)        {}     6.0
    90                 (B)       (A)       (D)       (C)       (F)       (E)        {}     4.0
    91                 (D)       (F)       (A)       (E)       (C)       (B)        {}     9.0
    92                 (C)       (D)       (B)       (F)       (A)       (E)        {}     5.0
    93                 (F)       (A)       (B)       (D)       (E)       (C)        {}     9.0
    94                 (F)       (B)       (D)       (C)       (E)       (A)        {}     2.0
    95                 (F)       (A)       (D)       (E)       (B)       (C)        {}     4.0
    96                 (E)       (A)       (C)       (D)       (B)       (F)        {}     2.0
    97                 (D)       (B)       (E)       (A)       (C)       (F)        {}     2.0
    98                 (E)       (F)       (D)       (C)       (B)       (A)        {}     4.0
    99                 (B)       (A)       (D)       (F)       (E)       (C)        {}     6.0
    100                (B)       (D)       (A)       (C)       (E)       (F)        {}     3.0
    101                (D)       (A)       (E)       (F)       (B)       (C)        {}     5.0
    102                (C)       (D)       (B)       (A)       (F)       (E)        {}     1.0
    103                (A)       (D)       (F)       (C)       (E)       (B)        {}     2.0
    104                (D)       (E)       (B)       (C)       (A)       (F)        {}     2.0
    105                (A)       (B)       (C)       (D)       (F)       (E)        {}     1.0
    106                (C)       (F)       (D)       (B)       (E)       (A)        {}     3.0
    107                (C)       (E)       (F)       (A)       (D)       (B)        {}     2.0
    108                (C)       (D)       (F)       (E)       (A)       (B)        {}     5.0
    109                (E)       (F)       (A)       (D)       (B)       (C)        {}     2.0
    110                (F)       (C)       (A)       (E)       (B)       (D)        {}     2.0
    111                (C)       (A)       (D)       (F)       (E)       (B)        {}     5.0
    112                (B)       (C)       (E)       (D)       (A)       (F)        {}     1.0
    113                (A)       (F)       (C)       (D)       (E)       (B)        {}     4.0
    114                (B)       (E)       (C)       (D)       (F)       (A)        {}     1.0
    115                (B)       (E)       (D)       (C)       (A)       (F)        {}     3.0
    116                (B)       (E)       (F)       (A)       (C)       (D)        {}     6.0
    117                (C)       (B)       (E)       (A)       (D)       (F)        {}     7.0
    118                (C)       (D)       (A)       (F)       (E)       (B)        {}     5.0
    119                (B)       (E)       (A)       (C)       (F)       (D)        {}     1.0
    120                (C)       (A)       (E)       (F)       (D)       (B)        {}     4.0
    121                (B)       (E)       (A)       (F)       (C)       (D)        {}     1.0
    122                (C)       (A)       (F)       (E)       (D)       (B)        {}     1.0
    123                (D)       (C)       (B)       (A)       (F)       (E)        {}     4.0
    124                (A)       (B)       (F)       (E)       (C)       (D)        {}     4.0
    125                (C)       (A)       (E)       (D)       (F)       (B)        {}     2.0
    126                (F)       (C)       (A)       (B)       (D)       (E)        {}     1.0
    127                (E)       (B)       (D)       (A)       (C)       (F)        {}     3.0
    128                (E)       (F)       (D)       (A)       (B)       (C)        {}     3.0
    129                (D)       (A)       (B)       (E)       (C)       (F)        {}     5.0
    130                (D)       (F)       (E)       (A)       (B)       (C)        {}     4.0
    131                (C)       (E)       (D)       (B)       (A)       (F)        {}     2.0
    132                (E)       (B)       (F)       (C)       (A)       (D)        {}     2.0
    133                (E)       (C)       (A)       (D)       (B)       (F)        {}     1.0
    134                (D)       (B)       (A)       (E)       (C)       (F)        {}     1.0
    135                (B)       (D)       (F)       (E)       (A)       (C)        {}    11.0
    136                (C)       (A)       (F)       (D)       (B)       (E)        {}     5.0
    137                (F)       (C)       (A)       (B)       (E)       (D)        {}     3.0
    138                (D)       (A)       (E)       (C)       (F)       (B)        {}     1.0
    139                (F)       (C)       (E)       (A)       (D)       (B)        {}     3.0
    140                (B)       (C)       (D)       (E)       (A)       (F)        {}     2.0
    141                (D)       (F)       (A)       (C)       (E)       (B)        {}     1.0
    142                (B)       (F)       (C)       (A)       (E)       (D)        {}     4.0
    143                (F)       (D)       (B)       (C)       (A)       (E)        {}     1.0
    144                (A)       (E)       (C)       (B)       (F)       (D)        {}     3.0
    145                (D)       (F)       (C)       (B)       (E)       (A)        {}     4.0
    146                (E)       (F)       (A)       (C)       (B)       (D)        {}     1.0
    147                (F)       (A)       (E)       (C)       (D)       (B)        {}     5.0
    148                (F)       (B)       (D)       (A)       (E)       (C)        {}     3.0
    149                (B)       (D)       (A)       (E)       (C)       (F)        {}     1.0
    150                (C)       (B)       (D)       (E)       (A)       (F)        {}     4.0
    151                (F)       (A)       (C)       (B)       (D)       (E)        {}     1.0
    152                (C)       (A)       (D)       (B)       (E)       (F)        {}     1.0
    153                (D)       (A)       (F)       (C)       (B)       (E)        {}     7.0
    154                (E)       (F)       (B)       (D)       (C)       (A)        {}     2.0
    155                (C)       (E)       (A)       (B)       (F)       (D)        {}     1.0
    156                (B)       (A)       (F)       (D)       (E)       (C)        {}     3.0
    157                (D)       (A)       (F)       (B)       (C)       (E)        {}     1.0
    158                (F)       (E)       (B)       (C)       (A)       (D)        {}     2.0
    159                (D)       (B)       (A)       (C)       (E)       (F)        {}     2.0
    160                (D)       (E)       (A)       (B)       (C)       (F)        {}     1.0
    161                (E)       (D)       (C)       (A)       (B)       (F)        {}     2.0
    162                (D)       (E)       (B)       (C)       (F)       (A)        {}     2.0
    163                (D)       (E)       (A)       (F)       (C)       (B)        {}     3.0
    164                (B)       (F)       (E)       (D)       (C)       (A)        {}     1.0
    165                (E)       (C)       (B)       (D)       (F)       (A)        {}     6.0
    166                (B)       (E)       (A)       (F)       (D)       (C)        {}     3.0
    167                (D)       (C)       (A)       (E)       (B)       (F)        {}     1.0
    168                (C)       (F)       (A)       (E)       (B)       (D)        {}     3.0
    169                (C)       (E)       (A)       (D)       (B)       (F)        {}     1.0
    170                (F)       (B)       (D)       (E)       (C)       (A)        {}     1.0
    171                (B)       (E)       (F)       (D)       (A)       (C)        {}     7.0
    172                (A)       (F)       (C)       (D)       (B)       (E)        {}     1.0
    173                (D)       (F)       (C)       (A)       (B)       (E)        {}     8.0
    174                (E)       (D)       (F)       (A)       (B)       (C)        {}     2.0
    175                (C)       (A)       (E)       (F)       (B)       (D)        {}     2.0
    176                (C)       (E)       (A)       (B)       (D)       (F)        {}     6.0
    177                (B)       (D)       (C)       (F)       (A)       (E)        {}     1.0
    178                (E)       (C)       (D)       (A)       (B)       (F)        {}     1.0
    179                (F)       (C)       (A)       (E)       (D)       (B)        {}     1.0
    180                (E)       (A)       (C)       (F)       (D)       (B)        {}     1.0
    181                (D)       (C)       (A)       (F)       (E)       (B)        {}     3.0
    182                (E)       (B)       (F)       (A)       (D)       (C)        {}     2.0
    183                (A)       (B)       (C)       (E)       (F)       (D)        {}     8.0
    184                (A)       (F)       (B)       (C)       (E)       (D)        {}     1.0
    185                (D)       (C)       (A)       (F)       (B)       (E)        {}     2.0
    186                (D)       (C)       (B)       (F)       (A)       (E)        {}     2.0
    187                (B)       (E)       (D)       (F)       (A)       (C)        {}     1.0
    188                (A)       (C)       (D)       (F)       (E)       (B)        {}     1.0
    189                (B)       (E)       (C)       (A)       (F)       (D)        {}     2.0
    190                (A)       (E)       (D)       (C)       (F)       (B)        {}     4.0
    191                (C)       (D)       (E)       (A)       (B)       (F)        {}     5.0
    192                (C)       (A)       (D)       (F)       (B)       (E)        {}     1.0
    193                (E)       (C)       (D)       (F)       (A)       (B)        {}     3.0
    194                (B)       (F)       (C)       (E)       (D)       (A)        {}     3.0
    195                (C)       (A)       (B)       (F)       (E)       (D)        {}     2.0
    196                (F)       (C)       (A)       (D)       (B)       (E)        {}     1.0
    197                (C)       (B)       (F)       (E)       (A)       (D)        {}     2.0
    198                (C)       (D)       (F)       (B)       (E)       (A)        {}     1.0
    199                (D)       (C)       (A)       (B)       (F)       (E)        {}     2.0
    200                (D)       (C)       (F)       (E)       (A)       (B)        {}     5.0
    201                (F)       (D)       (A)       (C)       (B)       (E)        {}     3.0
    202                (D)       (F)       (C)       (E)       (B)       (A)        {}     1.0
    203                (F)       (C)       (D)       (A)       (B)       (E)        {}     2.0
    204                (A)       (F)       (B)       (D)       (C)       (E)        {}     7.0
    205                (C)       (B)       (D)       (A)       (E)       (F)        {}     2.0
    206                (A)       (C)       (E)       (B)       (F)       (D)        {}     2.0
    207                (D)       (A)       (E)       (B)       (F)       (C)        {}     1.0
    208                (A)       (F)       (D)       (E)       (C)       (B)        {}     2.0
    209                (C)       (D)       (A)       (E)       (F)       (B)        {}     3.0
    210                (A)       (D)       (B)       (C)       (E)       (F)        {}     2.0
    211                (A)       (D)       (E)       (B)       (F)       (C)        {}     2.0
    212                (B)       (F)       (D)       (A)       (E)       (C)        {}     2.0
    213                (F)       (C)       (D)       (E)       (B)       (A)        {}     3.0
    214                (A)       (B)       (D)       (C)       (F)       (E)        {}     4.0
    215                (E)       (B)       (C)       (F)       (A)       (D)        {}     3.0
    216                (B)       (F)       (A)       (E)       (C)       (D)        {}     2.0
    217                (B)       (D)       (C)       (E)       (F)       (A)        {}     3.0
    218                (F)       (C)       (A)       (D)       (E)       (B)        {}     3.0
    219                (D)       (F)       (C)       (A)       (E)       (B)        {}     3.0
    220                (B)       (C)       (A)       (D)       (E)       (F)        {}     1.0
    221                (A)       (E)       (B)       (D)       (F)       (C)        {}     1.0
    222                (C)       (F)       (D)       (E)       (A)       (B)        {}     1.0
    223                (A)       (C)       (B)       (F)       (E)       (D)        {}     1.0
    224                (B)       (A)       (F)       (E)       (C)       (D)        {}     2.0
    225                (E)       (B)       (A)       (C)       (D)       (F)        {}     2.0
    226                (A)       (C)       (B)       (D)       (F)       (E)        {}     1.0
    227                (F)       (D)       (C)       (B)       (A)       (E)        {}     1.0
    228                (F)       (E)       (A)       (D)       (C)       (B)        {}     2.0
    229                (B)       (C)       (E)       (A)       (F)       (D)        {}     2.0
    230                (F)       (E)       (C)       (B)       (A)       (D)        {}     5.0
    231                (B)       (E)       (A)       (C)       (D)       (F)        {}     2.0
    232                (A)       (B)       (D)       (E)       (F)       (C)        {}     4.0
    233                (F)       (C)       (D)       (A)       (E)       (B)        {}     3.0
    234                (A)       (E)       (C)       (F)       (B)       (D)        {}     3.0
    235                (C)       (E)       (B)       (A)       (F)       (D)        {}     1.0
    236                (B)       (D)       (A)       (F)       (C)       (E)        {}     2.0
    237                (B)       (D)       (F)       (E)       (C)       (A)        {}     1.0
    238                (D)       (F)       (C)       (B)       (A)       (E)        {}     1.0
    239                (C)       (A)       (B)       (E)       (F)       (D)        {}     3.0
    240                (D)       (F)       (E)       (B)       (A)       (C)        {}     1.0
    241                (C)       (B)       (F)       (D)       (A)       (E)        {}     2.0
    242                (A)       (E)       (C)       (D)       (F)       (B)        {}     1.0
    243                (A)       (E)       (F)       (B)       (C)       (D)        {}     3.0
    244                (B)       (F)       (A)       (C)       (E)       (D)        {}     1.0
    245                (E)       (D)       (B)       (F)       (A)       (C)        {}     3.0
    246                (C)       (E)       (F)       (A)       (B)       (D)        {}     1.0
    247                (D)       (C)       (E)       (F)       (B)       (A)        {}     1.0
    248                (B)       (A)       (D)       (C)       (E)       (F)        {}     1.0
    249                (D)       (B)       (E)       (A)       (F)       (C)        {}     2.0
    250                (D)       (F)       (E)       (C)       (A)       (B)        {}     1.0
    251                (F)       (A)       (B)       (D)       (C)       (E)        {}     2.0
    252                (E)       (B)       (C)       (A)       (D)       (F)        {}     2.0
    253                (C)       (A)       (F)       (E)       (B)       (D)        {}     3.0
    254                (A)       (F)       (E)       (C)       (B)       (D)        {}     1.0
    255                (A)       (D)       (F)       (E)       (B)       (C)        {}     2.0
    256                (C)       (A)       (B)       (E)       (D)       (F)        {}     3.0
    257                (E)       (F)       (D)       (B)       (C)       (A)        {}     3.0
    258                (F)       (A)       (E)       (C)       (B)       (D)        {}     1.0
    259                (E)       (B)       (C)       (D)       (A)       (F)        {}     1.0
    260                (E)       (D)       (F)       (B)       (A)       (C)        {}     4.0
    261                (C)       (B)       (F)       (D)       (E)       (A)        {}     2.0
    262                (E)       (F)       (A)       (B)       (C)       (D)        {}     3.0
    263                (D)       (C)       (B)       (E)       (F)       (A)        {}     2.0
    264                (D)       (B)       (E)       (C)       (F)       (A)        {}     2.0
    265                (F)       (C)       (E)       (B)       (A)       (D)        {}     2.0
    266                (D)       (B)       (C)       (A)       (F)       (E)        {}     1.0
    267                (F)       (B)       (E)       (D)       (A)       (C)        {}     3.0
    268                (C)       (D)       (B)       (E)       (F)       (A)        {}     3.0
    269                (A)       (B)       (F)       (C)       (E)       (D)        {}     3.0
    270                (A)       (E)       (F)       (D)       (C)       (B)        {}     2.0
    271                (C)       (B)       (F)       (A)       (D)       (E)        {}     2.0
    272                (D)       (E)       (B)       (F)       (A)       (C)        {}     2.0
    273                (F)       (C)       (B)       (E)       (D)       (A)        {}     2.0
    274                (A)       (F)       (D)       (B)       (E)       (C)        {}     2.0
    275                (B)       (F)       (E)       (C)       (D)       (A)        {}     1.0
    276                (E)       (B)       (A)       (C)       (F)       (D)        {}     2.0
    277                (F)       (D)       (B)       (E)       (A)       (C)        {}     1.0
    278                (B)       (A)       (C)       (F)       (D)       (E)        {}     1.0
    279                (F)       (D)       (B)       (C)       (E)       (A)        {}     1.0
    280                (C)       (D)       (A)       (B)       (E)       (F)        {}     2.0
    281                (F)       (A)       (C)       (E)       (D)       (B)        {}     2.0
    282                (B)       (A)       (D)       (E)       (C)       (F)        {}     1.0
    283                (D)       (B)       (F)       (E)       (C)       (A)        {}     1.0
    284                (D)       (B)       (C)       (F)       (E)       (A)        {}     4.0
    285                (D)       (E)       (A)       (F)       (B)       (C)        {}     2.0
    286                (A)       (D)       (B)       (F)       (C)       (E)        {}     3.0
    287                (C)       (D)       (E)       (B)       (A)       (F)        {}     1.0
    288                (F)       (A)       (C)       (B)       (E)       (D)        {}     1.0
    289                (D)       (B)       (C)       (E)       (F)       (A)        {}     1.0
    290                (E)       (C)       (A)       (F)       (B)       (D)        {}     1.0
    291                (A)       (C)       (E)       (B)       (D)       (F)        {}     2.0
    292                (D)       (A)       (C)       (E)       (B)       (F)        {}     1.0
    293                (A)       (B)       (E)       (C)       (D)       (F)        {}     1.0
    294                (B)       (E)       (C)       (D)       (A)       (F)        {}     1.0
    295                (C)       (A)       (B)       (D)       (E)       (F)        {}     3.0
    296                (F)       (E)       (B)       (A)       (D)       (C)        {}     2.0
    297                (F)       (A)       (D)       (E)       (C)       (B)        {}     2.0
    298                (B)       (F)       (E)       (A)       (C)       (D)        {}     1.0
    299                (E)       (B)       (D)       (C)       (F)       (A)        {}     1.0
    300                (A)       (E)       (F)       (C)       (B)       (D)        {}     1.0
    301                (E)       (A)       (F)       (C)       (B)       (D)        {}     1.0
    302                (F)       (C)       (E)       (D)       (B)       (A)        {}     3.0
    303                (E)       (D)       (C)       (B)       (F)       (A)        {}     1.0
    304                (D)       (B)       (E)       (F)       (A)       (C)        {}     2.0
    305                (E)       (B)       (C)       (A)       (F)       (D)        {}     2.0
    306                (E)       (A)       (F)       (C)       (D)       (B)        {}     1.0
    307                (A)       (E)       (D)       (F)       (C)       (B)        {}     3.0
    308                (C)       (F)       (D)       (A)       (B)       (E)        {}     2.0
    309                (D)       (B)       (F)       (C)       (E)       (A)        {}     1.0
    310                (C)       (E)       (D)       (A)       (B)       (F)        {}     1.0
    311                (A)       (C)       (F)       (E)       (B)       (D)        {}     1.0
    312                (D)       (E)       (C)       (A)       (B)       (F)        {}     1.0
    313                (E)       (F)       (C)       (D)       (B)       (A)        {}     1.0
    314                (F)       (C)       (D)       (B)       (E)       (A)        {}     1.0
    315                (C)       (F)       (D)       (A)       (E)       (B)        {}     1.0
    316                (A)       (C)       (F)       (E)       (D)       (B)        {}     2.0
    317                (C)       (D)       (B)       (F)       (E)       (A)        {}     3.0
    318                (D)       (E)       (F)       (A)       (C)       (B)        {}     2.0
    319                (B)       (D)       (F)       (C)       (E)       (A)        {}     2.0
    320                (F)       (E)       (D)       (C)       (B)       (A)        {}     1.0
    321                (E)       (B)       (D)       (A)       (F)       (C)        {}     2.0
    322                (C)       (F)       (B)       (E)       (D)       (A)        {}     1.0
    323                (E)       (A)       (B)       (D)       (C)       (F)        {}     1.0
    324                (C)       (F)       (E)       (A)       (B)       (D)        {}     3.0
    325                (F)       (E)       (D)       (A)       (B)       (C)        {}     1.0
    326                (D)       (B)       (C)       (E)       (A)       (F)        {}     1.0
    327                (C)       (B)       (D)       (F)       (E)       (A)        {}     2.0
    328                (F)       (E)       (C)       (D)       (B)       (A)        {}     1.0
    329                (C)       (A)       (F)       (D)       (E)       (B)        {}     1.0
    330                (C)       (D)       (B)       (E)       (A)       (F)        {}     1.0
    331                (E)       (B)       (F)       (C)       (D)       (A)        {}     1.0
    332                (E)       (B)       (F)       (D)       (C)       (A)        {}     1.0
    333                (D)       (B)       (F)       (C)       (A)       (E)        {}     1.0
    334                (B)       (C)       (E)       (F)       (D)       (A)        {}     1.0
    335                (A)       (B)       (F)       (C)       (D)       (E)        {}     1.0
    336                (A)       (C)       (F)       (D)       (E)       (B)        {}     1.0
    337                (A)       (C)       (B)       (E)       (F)       (D)        {}     1.0
    338                (C)       (B)       (A)       (E)       (D)       (F)        {}     3.0
    339                (B)       (A)       (D)       (E)       (F)       (C)        {}     1.0
    340                (C)       (E)       (F)       (D)       (A)       (B)        {}     2.0
    341                (A)       (E)       (B)       (F)       (D)       (C)        {}     1.0
    342                (F)       (D)       (A)       (C)       (E)       (B)        {}     2.0
    343                (F)       (A)       (C)       (D)       (B)       (E)        {}     3.0
    344                (D)       (E)       (C)       (A)       (F)       (B)        {}     1.0
    345                (B)       (E)       (D)       (A)       (C)       (F)        {}     1.0
    346                (C)       (F)       (B)       (D)       (A)       (E)        {}     1.0
    347                (D)       (C)       (B)       (E)       (A)       (F)        {}     1.0
    348                (C)       (B)       (D)       (E)       (F)       (A)        {}     1.0
    349                (B)       (A)       (C)       (D)       (E)       (F)        {}     1.0
    350                (C)       (D)       (F)       (A)       (B)       (E)        {}     1.0
    351                (A)       (D)       (E)       (C)       (B)       (F)        {}     1.0
    352                (E)       (B)       (D)       (F)       (A)       (C)        {}     1.0
    353                (B)       (C)       (F)       (A)       (E)       (D)        {}     2.0
    354                (F)       (D)       (C)       (E)       (B)       (A)        {}     1.0
    355                (C)       (F)       (A)       (B)       (D)       (E)        {}     2.0
    356                (A)       (F)       (E)       (D)       (B)       (C)        {}     2.0
    357                (A)       (D)       (C)       (E)       (B)       (F)        {}     1.0
    358                (F)       (E)       (A)       (B)       (C)       (D)        {}     1.0
    359                (B)       (E)       (F)       (C)       (A)       (D)        {}     1.0
    360                (B)       (A)       (C)       (D)       (F)       (E)        {}     2.0
    361                (E)       (F)       (D)       (A)       (C)       (B)        {}     1.0
    362                (F)       (C)       (B)       (A)       (D)       (E)        {}     1.0
    363                (C)       (A)       (D)       (E)       (F)       (B)        {}     2.0
    364                (C)       (E)       (A)       (F)       (D)       (B)        {}     1.0
    365                (D)       (B)       (C)       (F)       (A)       (E)        {}     2.0
    366                (B)       (F)       (D)       (E)       (A)       (C)        {}     2.0
    367                (E)       (F)       (C)       (A)       (B)       (D)        {}     1.0
    368                (E)       (B)       (D)       (C)       (A)       (F)        {}     1.0
    369                (C)       (E)       (D)       (B)       (F)       (A)        {}     1.0
    370                (F)       (A)       (E)       (D)       (B)       (C)        {}     1.0
    371                (B)       (C)       (F)       (E)       (D)       (A)        {}     1.0
    372                (A)       (C)       (D)       (E)       (B)       (F)        {}     1.0
    373                (D)       (C)       (B)       (A)       (E)       (F)        {}     1.0
    374                (E)       (B)       (A)       (D)       (C)       (F)        {}     1.0
    375                (D)       (F)       (B)       (C)       (E)       (A)        {}     1.0
    376                (D)       (E)       (C)       (F)       (A)       (B)        {}     1.0
    377                (E)       (B)       (C)       (D)       (F)       (A)        {}     1.0
    378                (C)       (B)       (F)       (A)       (E)       (D)        {}     1.0
    379                (E)       (A)       (B)       (D)       (F)       (C)        {}     1.0
    380                (E)       (D)       (A)       (F)       (C)       (B)        {}     1.0
    381                (A)       (C)       (D)       (F)       (B)       (E)        {}     1.0
    382                (A)       (C)       (D)       (E)       (F)       (B)        {}     1.0
    383                (D)       (C)       (E)       (B)       (F)       (A)        {}     1.0
    384                (C)       (E)       (A)       (D)       (F)       (B)        {}     1.0
    385                (A)       (B)       (C)       (E)       (D)       (F)        {}     1.0
    386                (A)       (E)       (F)       (C)       (D)       (B)        {}     1.0
    387                (A)       (D)       (F)       (E)       (C)       (B)        {}     2.0
    388                (F)       (B)       (C)       (A)       (E)       (D)        {}     1.0
    389                (D)       (A)       (B)       (E)       (F)       (C)        {}     1.0
    390                (F)       (E)       (C)       (A)       (B)       (D)        {}     1.0
    391                (E)       (D)       (C)       (B)       (A)       (F)        {}     1.0
    392                (C)       (A)       (B)       (D)       (F)       (E)        {}     1.0
    393                (F)       (B)       (E)       (A)       (D)       (C)        {}     1.0
    394                (A)       (F)       (C)       (E)       (D)       (B)        {}     1.0
    395                (D)       (E)       (A)       (C)       (B)       (F)        {}     1.0
    396                (C)       (D)       (A)       (B)       (F)       (E)        {}     1.0
    
          Status  Round
    F    Elected      1
    A    Elected      1
    C    Elected      1
    B  Remaining      1
    D  Remaining      1
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

    Winners: (frozenset({'F'}), frozenset({'A'}), frozenset({'C'}))
    Eliminated: ()
    Ranking: (frozenset({'F'}), frozenset({'A'}), frozenset({'C'}), frozenset({'B'}), frozenset({'D'}), frozenset({'E'}))
    Outcome of round 1:
           Status  Round
    F    Elected      1
    A    Elected      1
    C    Elected      1
    B  Remaining      1
    D  Remaining      1
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

    minneapolis_profile = load_csv("mn_2013_cast_vote_record.csv")
    minneapolis_profile = remove_and_condense(["undervote", "overvote", "UWI"], minneapolis_profile)
    
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

    Round 1
    
    Winners: ()
    Eliminated: (frozenset({'JOHN CHARLES WILSON'}),)
    Remaining: (frozenset({'BETSY HODGES'}), frozenset({'MARK ANDREW'}), frozenset({'DON SAMUELS'}), frozenset({'CAM WINTON'}), frozenset({'JACKIE CHERRYHOMES'}), frozenset({'BOB FINE'}), frozenset({'DAN COHEN'}), frozenset({'STEPHANIE WOODRUFF'}), frozenset({'MARK V ANDERSON'}), frozenset({'DOUG MANN'}), frozenset({'OLE SAVIOR'}), frozenset({'ABDUL M RAHAMAN "THE ROCK"'}), frozenset({'ALICIA K. BENNETT'}), frozenset({'JAMES EVERETT'}), frozenset({'CAPTAIN JACK SPARROW'}), frozenset({'TONY LANE'}), frozenset({'MIKE GOULD'}), frozenset({'KURTIS W. HANNA'}), frozenset({'JAYMIE KELLY'}), frozenset({'CHRISTOPHER CLARK'}), frozenset({'CHRISTOPHER ROBIN ZIMMERMAN'}), frozenset({'JEFFREY ALAN WAGNER'}), frozenset({'TROY BENJEGERDES'}), frozenset({'GREGG A. IVERSON', 'NEAL BAXTER'}), frozenset({'JOSHUA REA'}), frozenset({'MERRILL ANDERSON'}), frozenset({'BILL KAHN'}), frozenset({'JOHN LESLIE HARTWIG'}), frozenset({'EDMUND BERNARD BRUYERE'}), frozenset({'JAMES "JIMMY" L. STROUD, JR.', 'RAHN V. WORKCUFF'}), frozenset({'BOB "AGAIN" CARNEY JR'}), frozenset({'CYD GORMAN'}))
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
    GREGG A. IVERSON               Remaining      1
    NEAL BAXTER                    Remaining      1
    JOSHUA REA                     Remaining      1
    MERRILL ANDERSON               Remaining      1
    BILL KAHN                      Remaining      1
    JOHN LESLIE HARTWIG            Remaining      1
    EDMUND BERNARD BRUYERE         Remaining      1
    JAMES "JIMMY" L. STROUD, JR.   Remaining      1
    RAHN V. WORKCUFF               Remaining      1
    BOB "AGAIN" CARNEY JR          Remaining      1
    CYD GORMAN                     Remaining      1
    JOHN CHARLES WILSON           Eliminated      1
    
    Round 2
    
    Winners: ()
    Eliminated: (frozenset({'CYD GORMAN'}), frozenset({'JOHN CHARLES WILSON'}))
    Remaining: (frozenset({'BETSY HODGES'}), frozenset({'MARK ANDREW'}), frozenset({'DON SAMUELS'}), frozenset({'CAM WINTON'}), frozenset({'JACKIE CHERRYHOMES'}), frozenset({'BOB FINE'}), frozenset({'DAN COHEN'}), frozenset({'STEPHANIE WOODRUFF'}), frozenset({'MARK V ANDERSON'}), frozenset({'DOUG MANN'}), frozenset({'OLE SAVIOR'}), frozenset({'ABDUL M RAHAMAN "THE ROCK"'}), frozenset({'ALICIA K. BENNETT'}), frozenset({'JAMES EVERETT'}), frozenset({'CAPTAIN JACK SPARROW'}), frozenset({'TONY LANE'}), frozenset({'MIKE GOULD'}), frozenset({'KURTIS W. HANNA'}), frozenset({'JAYMIE KELLY'}), frozenset({'CHRISTOPHER CLARK'}), frozenset({'CHRISTOPHER ROBIN ZIMMERMAN'}), frozenset({'JEFFREY ALAN WAGNER'}), frozenset({'TROY BENJEGERDES'}), frozenset({'GREGG A. IVERSON'}), frozenset({'NEAL BAXTER'}), frozenset({'JOSHUA REA'}), frozenset({'MERRILL ANDERSON'}), frozenset({'BILL KAHN'}), frozenset({'JOHN LESLIE HARTWIG'}), frozenset({'EDMUND BERNARD BRUYERE'}), frozenset({'JAMES "JIMMY" L. STROUD, JR.', 'RAHN V. WORKCUFF'}), frozenset({'BOB "AGAIN" CARNEY JR'}))
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
    JAMES "JIMMY" L. STROUD, JR.   Remaining      2
    RAHN V. WORKCUFF               Remaining      2
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
    


.. parsed-literal::

    /Users/cdonnay/Documents/GitHub/MGGG/VoteKit/src/votekit/pref_profile/pref_profile.py:1109: UserWarning: Profile does not contain rankings but max_ranking_length=3. Setting max_ranking_length to 0.
      warnings.warn(


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
    E    Elected      1
    D    Elected      2
    A    Elected      3
    F  Remaining      3
    C  Remaining      3
    B  Remaining      3


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
