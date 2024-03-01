# Ballot Generators: Generating `PreferenceProfiles`
We have already seen the use of a  `PreferenceProfile` generator (the Impartial Culture Model) in the Plotting and Ballot Graph tutorials. Now, let's dive into the rest that are included in `votekit`. This tutorial will cover
- Impartial Culture
- Impartial Anonymous Culture
- name-Plackett Luce
- name-Bradley Terry
- name-Cumulative


```python
import votekit.ballot_generator as bg
from votekit.pref_interval import PreferenceInterval
```

The two simplest to use are the Impartial Culture and Impartial Anonymous Culture. For $m$ candidates and $n$ voters, the Impartial Culture model generates `PreferenceProfiles` uniformly at random out of the $(m!)^n$ possible profiles. Remember, a `PreferenceProfile` is a tuple of length $n$ that stores a linear ranking $m$ in each slot.

The Impartial Anonymous Culture model works a little bit differently. When it generates ballots, it chooses a candidate support vector uniformly at random from among all possible support vectors, and then generates ballots according to that vector.


```python
candidates = ["A", "B", "C"]
number_of_ballots = 50
#Impartial Culture
ic = bg.ImpartialCulture(candidates = candidates)
ic_profile = ic.generate_profile(number_of_ballots)

#Impartial Anonymous Culture
iac = bg.ImpartialAnonymousCulture(candidates = candidates)
iac_profile = iac.generate_profile(number_of_ballots)

```

The 1-D Spatial model assigns each candidate a random point on the real line according to the standard normal distribution. It then does the same for each voter, and then a voter ranks candidates by their distance from the voter.


```python
one_d = bg.OneDimSpatial(candidates = candidates)
one_d_profile = one_d.generate_profile(number_of_ballots)
```

# Ballots Generated Using Intervals

The following generative models all depend on preference intervals.


The name-Plackett-Luce, name-Bradley-Terry, and name-Cumulative models all use the interval $[0,1]$. To use these models, we need a bit more information than just the candidates. Suppose for now that there is one type of voter (or bloc $Q$) in the state (these models can be generalized to more than one bloc, but we will start simple for now). We record the proportion of voters in this bloc in a dictionary.

### Name-PL and Name-BT
In the upcoming election, there are three candidates, $A$, $B$, and $C$. In general, the bloc $Q$ prefers $A$ 1/2  of the time, $B$ 1/3 of the time, and $C$ 1/6 of the time. We can visualize this as the line segment $[0,1]$, with the segment $[0,1/2]$ labeled $A$, $[1/2, 5/6]$ labeled $B$, and $[5/6,1]$ labeled $C$. Note the slight abuse of notation in using the same name for the candidates and their intervals. We store this information in a `PreferenceInterval` object.


<!-- Suppose there are two blocs (or groups) of voters, $Q$ and $R$. The $Q$ bloc is estimated to be about 70% of the voting population, while the $R$ block is about 30%. Within each bloc there is preference for different candidates, which we record in the variable `pref_interval_by_bloc`. 

In this example, suppose each bloc has two candidates running, but there is some crossover in which some voters from bloc $Q$ actually prefer the candidates from bloc $R$. The $R$ bloc, being much more insular, does not prefer either of $Q$'s candidates. -->


```python
candidates = ["A", "B", "C"]
number_of_ballots = 50

bloc_voter_prop = {"Q":1}

pref_intervals_by_bloc = {"Q" : {"Q": 
                                 PreferenceInterval({"A": 1/2,  "B": 1/3, "C": 1/6})
                                 }
                                 }

```

For each voter, the name-Plackett-Luce (PL) model samples from the list of candidates without replacement according to the distribution defined by the preference intervals. The first candidate it samples is in first place, then second, etc. Visualizing this as the line segment, the PL model uniformly at random selects a point in $[0,1]$. Whichever candidate's interval that point lies in is listed first in the ballot. It then removes that candidate's preference interval from $[0,1]$, rescales so the segment has length 1 again, and then samples a second candidate. Repeat until all candidates have been sampled. We will discuss the `cohesion_parameters` argument later.


```python
# Plackett-Luce
pl = bg.name_PlackettLuce(pref_intervals_by_bloc=pref_intervals_by_bloc,
                     bloc_voter_prop=bloc_voter_prop, 
                     candidates=candidates,
                     cohesion_parameters={"Q":{"Q":1}})

pl_profile = pl.generate_profile(number_of_ballots)
print(pl_profile)
```

      Ballots Weight
    (A, B, C)     21
    (B, A, C)     12
    (C, A, B)      7
    (A, C, B)      4
    (C, B, A)      3
    (B, C, A)      3


The name-Bradley-Terry (BT) model also fundamentally relies on these preference intervals. The probability that BT samples the ballot $(A>B>C)$ is proportional to the the product of the pairwise probabilities $(A>B), (A>C),$ and $(B>C)$. Using our preference intervals, the probability that $A>B$ is $\frac{A}{A+B}$; out of a line segment of length $A+B$, this is the probability that a uniform random point lies in the $A$ portion. The other probabilities are computed similarly.


```python
# Bradley-Terry
bt = bg.name_BradleyTerry(pref_intervals_by_bloc=pref_intervals_by_bloc,
                     bloc_voter_prop=bloc_voter_prop, 
                     candidates=candidates,
                     cohesion_parameters = {"Q":{"Q":1}})

bt_profile = bt.generate_profile(number_of_ballots)

print(bt_profile)
```

      Ballots Weight
    (A, B, C)     22
    (B, A, C)     11
    (A, C, B)     10
    (B, C, A)      5
    (C, A, B)      1
    (C, B, A)      1


We can do a more complicated example of PL and BT. Consider an election where there are 2 blocs of voters, $Q$ and $R$. There are two candidates from the $Q$ bloc, and two from the $R$ bloc. The $R$ block is more insular, and expresses no interest in any of the $Q$ candidates, while the $Q$ bloc does have some preference for $R$'s candidates. We express this using `cohesion_parameters`, which stores the preference of each slate for the other slate's candidates.


```python
candidates = ["Q1", "Q2", "R1", "R2"]
number_of_ballots = 50

bloc_voter_prop = {"Q": 0.7, "R": 0.3}

pref_intervals_by_bloc = {
    "Q": {"Q":PreferenceInterval({"Q1": 0.4, "Q2": 0.3}),
          "R":PreferenceInterval({"R1": 0.2, "R2": 0.1})},
    "R": {"Q":PreferenceInterval({"Q1": 0.3, "Q2": 0.7}),
          "R":PreferenceInterval({"R1": 0.4, "R2": 0.6})}
}

cohesion_parameters = {"Q": {"Q": .8, "R":.2},
                       "R": {"R":1, "Q":0}}
```


```python
pl = bg.name_PlackettLuce(pref_intervals_by_bloc=pref_intervals_by_bloc,
                     bloc_voter_prop=bloc_voter_prop, 
                     candidates=candidates,
                     cohesion_parameters=cohesion_parameters)

pl_profile = pl.generate_profile(number_of_ballots)

print("Number of ballots:", pl_profile.num_ballots())
print(pl_profile)
```

    Number of ballots: 50
    PreferenceProfile too long, only showing 15 out of 15 rows.
                                    Ballots Weight
                           (Q1, Q2, R1, R2)     11
    (R2, R1, frozenset({'Q1', 'Q2'}) (Tie))     11
                           (R1, Q2, Q1, R2)      6
                           (Q2, Q1, R1, R2)      5
    (R1, R2, frozenset({'Q1', 'Q2'}) (Tie))      4
                           (Q1, R1, Q2, R2)      2
                           (Q2, R1, Q1, R2)      2
                           (Q1, Q2, R2, R1)      2
                           (Q1, R1, R2, Q2)      1
                           (Q2, R2, Q1, R1)      1
                           (Q2, R2, R1, Q1)      1
                           (R1, R2, Q2, Q1)      1
                           (Q2, R1, R2, Q1)      1
                           (Q1, R2, Q2, R1)      1
                           (R2, Q2, Q1, R1)      1


Notice that for the first time we have ties on the ballots! The notation `{'Q1', 'Q2'} (Tie)` means that these two candidates are tied for third place.


```python
# Bradley-Terry
bt = bg.name_BradleyTerry(pref_intervals_by_bloc=pref_intervals_by_bloc,
                     bloc_voter_prop=bloc_voter_prop, 
                     candidates=candidates,
                     cohesion_parameters=cohesion_parameters)

bt_profile = bt.generate_profile(number_of_ballots)
print("Number of ballots:", bt_profile.num_ballots())
print(bt_profile)
```

    Number of ballots: 50
                                    Ballots Weight
                           (Q1, Q2, R1, R2)      9
    (R1, R2, frozenset({'Q1', 'Q2'}) (Tie))      9
                           (Q1, R1, Q2, R2)      7
                           (Q2, Q1, R1, R2)      6
    (R2, R1, frozenset({'Q1', 'Q2'}) (Tie))      6
                           (Q2, Q1, R2, R1)      3
                           (Q1, Q2, R2, R1)      3
                           (R1, Q1, Q2, R2)      3
                           (Q2, R1, Q1, R2)      3
                           (R1, Q2, Q1, R2)      1


### Name-Cumulative

Cumulative voting is a method in which voters are allowed to put candidates on the ballot with multiplicity.


```python
candidates = ["Q1", "Q2", "R1", "R2"]
number_of_ballots = 50

bloc_voter_prop = {"Q": 0.7, "R": 0.3}

pref_intervals_by_bloc = {
    "Q": {"Q":PreferenceInterval({"Q1": 0.4, "Q2": 0.3}),
          "R":PreferenceInterval({"R1": 0.2, "R2": 0.1})},
    "R": {"Q":PreferenceInterval({"Q1": 0.3, "Q2": 0.7}),
          "R":PreferenceInterval({"R1": 0.4, "R2": 0.6})}
}

cohesion_parameters = {"Q": {"Q": .8, "R":.2},
                       "R": {"R":1, "Q":0}}

num_votes_per_ballot = 3
```

We will also take this chance to introduce the `by_bloc` parameter to the `generate_profile` method, which when set to `True` returns a tuple. The first entry is a dictionary, which records the ballots cast by each bloc. The second entry is the full profile, i.e. what you would get if you just ran `generate_profile` with `by_bloc=False`.


```python
c  = bg.name_Cumulative(pref_intervals_by_bloc=pref_intervals_by_bloc,
                     bloc_voter_prop=bloc_voter_prop, 
                     candidates=candidates,
                     cohesion_parameters=cohesion_parameters,
                     num_votes=num_votes_per_ballot)

c_profile_dict, agg_profile = c.generate_profile(number_of_ballots=100, by_bloc=True)
```


```python
c_profile_dict["Q"]
```

    PreferenceProfile too long, only showing 15 out of 31 rows.





         Ballots Weight
    (Q1, Q1, Q1)      6
    (Q1, Q1, R1)      6
    (Q1, Q2, Q1)      6
    (Q2, Q2, Q1)      4
    (Q1, R1, Q1)      4
    (Q1, Q1, Q2)      4
    (Q1, Q2, R1)      3
    (Q2, Q2, Q2)      3
    (Q2, Q1, Q1)      3
    (R2, Q2, Q2)      3
    (R1, Q1, Q2)      2
    (Q1, Q2, Q2)      2
    (Q2, Q2, R1)      2
    (Q2, Q1, R1)      2
    (R2, Q2, Q1)      2




```python
c_profile_dict["R"]
```




         Ballots Weight
    (R2, R2, R2)     12
    (R1, R1, R2)      4
    (R1, R2, R1)      3
    (R1, R1, R1)      3
    (R2, R1, R2)      3
    (R2, R1, R1)      2
    (R2, R2, R1)      2
    (R1, R2, R2)      1




```python
agg_profile
```

    PreferenceProfile too long, only showing 15 out of 39 rows.





         Ballots Weight
    (R2, R2, R2)     12
    (Q1, Q1, R1)      6
    (Q1, Q2, Q1)      6
    (Q1, Q1, Q1)      6
    (Q1, R1, Q1)      4
    (Q2, Q2, Q1)      4
    (Q1, Q1, Q2)      4
    (R1, R1, R2)      4
    (R2, Q2, Q2)      3
    (Q2, Q2, Q2)      3
    (R1, R2, R1)      3
    (Q2, Q1, Q1)      3
    (R1, R1, R1)      3
    (Q1, Q2, R1)      3
    (R2, R1, R2)      3



Observe the multiplicity of candidates, as well as the fact that no voter in the `R` bloc cast a vote for `Q` candidates. To make the `Ballot` object as flexible as possible over different methods of election, we have implemented cumulative voting ballots as follows. The ranking on the ballot holds no meaning; all that matters is the multiplicity. That is, the ballot (R1, R1, R2) is the same as (R2, R1, R1). The `PreferenceProfile` object does not know that and thus displays them as different ballots, but our cumulative election class will handle tallying results for you.

We will discuss the slate models, as well as AC and CS in a later tutorial.


```python

```
