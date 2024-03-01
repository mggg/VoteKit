# More Ballot Generators: Slate, AC, and CS
We now turn to the remaining models of ballot generator.
- slate-Plackett Luce
- slate-Bradley Terry
- Alternating Crossover
- Cambridge Sampler

### Slate Models

The slate-Plackett Luce and slate-Bradley Terry models function very similarly to their name counterparts. In the name models, ballots were constructed directly from preference intervals and candidate names. In the slate models, we will first construct a ballot type where each entry of the ballot is the name of a slate, then fill in the candidate names separately. See our social choice documentation for more information.


```python
import votekit.ballot_generator as bg
from votekit.pref_interval import PreferenceInterval
```

In order to properly use the slate models, we must delineate which candidates belong to which slate. We do so with the `slate_to_candidates` parameter.


```python
slate_to_candidates = {"Q": ["Q1", "Q2"],
                       "R":["R1", "R2"]}
number_of_ballots = 50

bloc_voter_prop = {"Q": 0.7, "R": 0.3}

pref_intervals_by_bloc = {
    "Q": {"Q":PreferenceInterval({"Q1": 0.4, "Q2": 0.3}),
          "R":PreferenceInterval({"R1": 0.2, "R2": 0.1})},
    "R": {"Q":PreferenceInterval({"Q1": 0.3, "Q2": 0.7}),
          "R":PreferenceInterval({"R1": 0.4, "R2": 0.6})}
}

cohesion_parameters = {"Q": {"Q": .8, "R":.2},
                       "R": {"R":.9, "Q":.1}}

pl = bg.slate_PlackettLuce(pref_intervals_by_bloc=pref_intervals_by_bloc,
                     bloc_voter_prop=bloc_voter_prop, 
                     slate_to_candidates=slate_to_candidates,
                     cohesion_parameters=cohesion_parameters)

bt = bg.slate_BradleyTerry(pref_intervals_by_bloc=pref_intervals_by_bloc,
                     bloc_voter_prop=bloc_voter_prop, 
                     slate_to_candidates=slate_to_candidates,
                     cohesion_parameters=cohesion_parameters)
```

### Alternating Crossover

The Alternating Crossover model was first introduced in Benade et al. "Ranked Choice Voting and Proportional Representation" (February 2, 2021). Available at SSRN: https://ssrn.com/abstract=3778021. This model assumes there are two blocs over voters. Within a bloc, voters either vote with the bloc, or "crossover" to the other bloc. The proportion of such voters is controlled by the cohesion parameter.

Bloc voters rank all of the candidates from their bloc above all of the candidates from the other bloc. They choose their ranking of candidates via the PL model. Crossover voters first rank a candidate from the other bloc, then their bloc, etc, alternating until they run out of candidates from one bloc (at which point they stop.) Note that this means the AC model can generate incomplete ballots. Again, they choose their ranking via the PL model. 

A note on the preference intervals for this model. In this context, there are really two preference intervals for each bloc: the preference interval for their own candidates, and the preference interval for the opposing candidates. To input this as one preference interval, simply divide every value by 2.



```python
candidates = ["Q1", "Q2", "R1", "R2"]

pref_intervals_by_bloc = {
    "Q": {"Q":PreferenceInterval({"Q1": 0.4, "Q2": 0.3}),
          "R":PreferenceInterval({"R1": 0.2, "R2": 0.1})},
    "R": {"Q":PreferenceInterval({"Q1": 0.3, "Q2": 0.7}),
          "R":PreferenceInterval({"R1": 0.4, "R2": 0.6})}
}

bloc_voter_prop = {"Q": .7, "R": .3}
slate_to_candidates = {"Q": ["Q1", "Q2"],
                       "R": ["R1", "R2"]}

cohesion_parameters = {"Q": {"Q": .8, "R":.2},
                       "R": {"R":.9, "Q":.1}}

ac = bg.AlternatingCrossover(candidates = candidates,
                             pref_intervals_by_bloc =pref_intervals_by_bloc,
                             bloc_voter_prop = bloc_voter_prop,
                             slate_to_candidates = slate_to_candidates,
                             cohesion_parameters = cohesion_parameters)
```


```python
ac.generate_profile(100)
```

    PreferenceProfile too long, only showing 15 out of 15 rows.





             Ballots Weight
    (Q1, Q2, R2, R1)     20
    (Q2, Q1, R1, R2)     15
    (Q1, Q2, R1, R2)     13
    (R1, R2, Q1, Q2)      9
    (Q2, Q1, R2, R1)      8
    (R2, R1, Q2, Q1)      7
    (R1, R2, Q2, Q1)      6
    (R2, Q1, R1, Q2)      5
    (R2, R1, Q1, Q2)      5
    (R1, Q2, R2, Q1)      4
    (R2, Q2, R1, Q1)      3
    (R1, Q1, R2, Q2)      2
    (Q2, R2, Q1, R1)      1
    (Q1, R2, Q2, R1)      1
    (Q1, R1, Q2, R2)      1



### Cambridge Sampler

The Cambridge Sampler uses historical election data from Cambridge, MA to generate new ballots. You can use your own historical data with some of the provided optional parameters. The model assumes there is a majority and a minority bloc. Again there is a cohesion parameter measuring how often voters defect from the bloc. If voters vote with the bloc, they rank a bloc candidate first, and then the ballot is sampled from historical data with matching first entry. If they vote with the opposing bloc, they rank an opposing candidate first and then sample.

The historical ballots only give the order in which majority/minority bloc candidates are listed ( for example, WWC says there were two majority candidates and then a minority on the ballot).
Once the model decides which ballot type a voter has, it fills in the ballot with actual candidates using the preference interval ala PL.

Since it samples from historical data, it's possible to generate incomplete ballots.


```python
candidates = ["Q1", "Q2", "R1", "R2"]

pref_intervals_by_bloc = {
    "Q": {"Q":PreferenceInterval({"Q1": 0.4, "Q2": 0.3}),
          "R":PreferenceInterval({"R1": 0.2, "R2": 0.1})},
    "R": {"Q":PreferenceInterval({"Q1": 0.3, "Q2": 0.7}),
          "R":PreferenceInterval({"R1": 0.4, "R2": 0.6})}
}

bloc_voter_prop = {"Q": .7, "R": .3}
slate_to_candidates = {"Q": ["Q1", "Q2"],
                       "R": ["R1", "R2"]}

cohesion_parameters = {"Q": {"Q": .8, "R":.2},
                       "R": {"R":.9, "Q":.1}}

cs = bg.CambridgeSampler(pref_intervals_by_bloc=pref_intervals_by_bloc,
                         bloc_voter_prop=bloc_voter_prop, candidates=candidates,
                         slate_to_candidates=slate_to_candidates, cohesion_parameters=cohesion_parameters)
```


```python
cs.generate_profile(number_of_ballots=100)
```

    PreferenceProfile too long, only showing 15 out of 51 rows.





             Ballots Weight
        (Q2, Q1, R1)      6
            (Q2, Q1)      5
    (R2, R1, Q2, Q1)      4
    (Q2, Q1, R1, R2)      4
    (R2, Q2, Q1, R1)      4
               (Q1,)      4
            (Q1, Q2)      4
    (Q1, R1, Q2, R2)      4
    (Q1, Q2, R1, R2)      4
    (Q2, R1, Q1, R2)      3
        (R2, Q2, Q1)      3
    (Q2, R1, R2, Q1)      3
    (R1, R2, Q1, Q2)      3
        (Q2, Q1, R2)      3
            (R1, R2)      2




```python

```
