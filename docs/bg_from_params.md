# Ballot Generators from Parameters: the Dirichlet Distribution

In the VoteKit world of ballot generating, there are usually 4 parameters that need to be chosen.
- Bloc proportions
- Cohesion parameters
- Preference intervals
- Candidates per bloc

Instead of choosing a preference interval, VoteKit also makes it possible to generate a preference interval using the Dirichlet distribution.
The Dirichlet distribution samples a point from a simplex, in this case the candidate simplex. For three candidates, the candidate simplex is a triangle, where each corner represents a candidate. A point in the triangle is a vector of length 3 whose entries are non-negative and sum to 1 (exactly what a preference interval is!). A point that is close to a vertex is a preference interval with high support for one candidate. A point near the center of the triangle is a preference interval with near equal support for all three.

The Dirichlet distribution is parameterized by a parameter $\alpha \in (0,\infty)$.  As $\alpha\to\infty$, the distribution's mass moves towards the center of the simplex, so we get preference intervals that have more equal support for all candidates. As $\alpha\to 0$, the distribution's mass moves towards the corners, so we get preference intervals that have strong support for one candidate. When $\alpha=1$, all bets are off and could produce any preference interval.

By using the Dirichlet distribution instead of a fixed preference interval, you can study how the behavior of voters impacts elections.
- What happens in STV elections when voters have a strong preference for one candidate? A diffuse preference for all?

Let's see an example of how to construct ballots using the Dirichlet parameters.




```python
import votekit.ballot_generator as bg
```


```python
# the slate-Plackett-Luce model

bloc_proportions = {"A": .8, "B": .2}
cohesion_parameters = {"A":{"A": .9, "B":.1},
                       "B":{ "B": .9, "A":.1}}
dirichlet_alphas = {"A": {"A":1, "B":1},
                    "B": {"A":1, "B":1}}

slate_to_candidates = {"A": ["A1", "A2"],
                        "B": ["B1", "B2"]}
```

We need four different Dirichlet parameter's; $\alpha_{AA}$ generates the $A$ voters preference interval for $A$ candidates and $\alpha_{AB}$ generates the $A$ voters preference interval for $B$ candidates. Likewise for the $B$ voters.

Also notice that we need a bit more information in this case than if we gave the PL model a preference interval; we must specify the cohesion parameters and which candidates are in which bloc.


```python
pl = bg.slate_PlackettLuce.from_params(slate_to_candidates=slate_to_candidates,
                                 bloc_voter_prop=bloc_proportions,
                                 cohesion_parameters=cohesion_parameters,
                                 alphas=dirichlet_alphas)

profile = pl.generate_profile(number_of_ballots=1000)
print(profile)
```

    PreferenceProfile too long, only showing 15 out of 34 rows.
             Ballots Weight
    (A1, A2, B2, B1)    298
    (A2, A1, B2, B1)    222
    (B1, B2, A2, A1)    136
    (A1, A2, B1, B2)     72
    (A2, A1, B1, B2)     42
    (B2, A1, A2, B1)     30
    (A1, B2, A2, B1)     30
    (B2, A2, A1, B1)     27
    (A2, B2, A1, B1)     18
    (B1, A2, B2, A1)     18
    (B1, B2, A1, A2)     16
    (B1, A1, A2, B2)     14
    (A2, B1, B2, A1)     13
    (B2, B1, A2, A1)     11
    (B2, A2, B1, A1)      8


We can see what preference intervals were generated. Check for understanding; are these intervals what you would expect given the choices of parameter above?


```python
pl.pref_intervals_by_bloc
```




    {'A': {'A': {'A1': 0.586, 'A2': 0.414}, 'B': {'B1': 0.1574, 'B2': 0.8426}},
     'B': {'A': {'A1': 0.0915, 'A2': 0.9085}, 'B': {'B1': 0.9184, 'B2': 0.0816}}}



Let's fiddle with the Dirichlet parameter's to see how they impact things. By lowering $\alpha_{AB}$, we expect to see that $A$ voters have a strong preference for a particular $B$ candidate. By raising $\alpha_{BB}$, we expect $B$ voters to have relatively uniform preferences for $B$ candidates.


```python
# the slate-Plackett-Luce model

bloc_proportions = {"A": .8, "B": .2}
cohesion_parameters = {"A":{"A": .9, "B":.1},
                       "B":{ "B": .9, "A":.1}}
dirichlet_alphas = {"A": {"A":1, "B":.1},
                    "B": {"A":1, "B":1000}}

slate_to_candidates = {"A": ["A1", "A2"],
                        "B": ["B1", "B2"]}

pl = bg.slate_PlackettLuce.from_params(slate_to_candidates=slate_to_candidates,
                                 bloc_voter_prop=bloc_proportions,
                                 cohesion_parameters=cohesion_parameters,
                                 alphas=dirichlet_alphas)

print("A preference interval", pl.pref_intervals_by_bloc["A"])
print("B preference interval", pl.pref_intervals_by_bloc["B"], "\n")

profile_dict, pp = pl.generate_profile(number_of_ballots=1000, by_bloc=True)
print("A ballots\n", profile_dict["A"])
print()
print("B ballots\n", profile_dict["B"])
```

    A preference interval {'A': {'A1': 0.3228, 'A2': 0.6772}, 'B': {'B1': 0.0702, 'B2': 0.9298}}
    B preference interval {'A': {'A1': 0.5083, 'A2': 0.4917}, 'B': {'B1': 0.4888, 'B2': 0.5112}} 
    
    A ballots
     PreferenceProfile too long, only showing 15 out of 19 rows.
             Ballots Weight
    (A2, A1, B2, B1)    419
    (A1, A2, B2, B1)    206
    (A2, B2, A1, B1)     35
    (B2, A2, A1, B1)     34
    (A1, A2, B1, B2)     20
    (A1, B2, A2, B1)     19
    (A2, A1, B1, B2)     16
    (B2, A1, A2, B1)     15
    (B2, B1, A2, A1)      6
    (A2, B1, A1, B2)      6
    (B2, A2, B1, A1)      5
    (B2, A1, B1, A2)      5
    (B2, B1, A1, A2)      3
    (B1, A2, A1, B2)      3
    (A1, B1, A2, B2)      2
    
    B ballots
     PreferenceProfile too long, only showing 15 out of 16 rows.
             Ballots Weight
    (B2, B1, A1, A2)     46
    (B2, B1, A2, A1)     42
    (B1, B2, A1, A2)     40
    (B1, B2, A2, A1)     37
    (B2, A1, B1, A2)      8
    (A2, B1, B2, A1)      6
    (A1, B1, B2, A2)      4
    (A1, B2, B1, A2)      4
    (A2, B2, B1, A1)      3
    (B1, A2, B2, A1)      3
    (B2, A2, B1, A1)      2
    (A2, B2, A1, B1)      1
    (A1, A2, B1, B2)      1
    (B1, A2, A1, B2)      1
    (A2, A1, B2, B1)      1


Check for understanding; are the intervals and ballots what you'd expect?

Any of our other ballot generating models that rely on preference intervals can be generated from the Dirichlet distribution in a similar way.
