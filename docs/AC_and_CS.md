Ballot Generators: AC and CS
============================

# Alternating Crossover

 This Alternating Crossover model assumes there are two blocs over voters. Within a bloc, voters either vote with the bloc, or "crossover" to the other bloc. The proportion of such voters is controlled by the cohesion parameter. It was first introduced in Benade et al. "Ranked Choice Voting and Proportional Representation" (you can read more about it [here](https://ssrn.com/abstract=3778021)).

Bloc voters rank all of the candidates from their bloc above all of the candidates from the other bloc. They choose their ranking of candidates via the Plackett-Luce model. Crossover voters first rank a candidate from the other bloc, then their bloc, etc, alternating until they run out of candidates from one bloc (at which point they stop.) Note that this means the AC model can generate incomplete ballots. Again, they choose their ranking via the PL model. 

A note on the preference intervals for this model. In this context, there are really two preference intervals for each bloc: the preference interval for their own candidates, and the preference interval for the opposing candidates. To input this as one preference interval, simply divide every value by 2.


```python
from votekit import AlternatingCrossover

candidates = ["Q1", "Q2", "R1", "R2"]

# this is really saying the Q bloc has bloc preference .6,.4
# and cross preference .5,.5
pref_interval_by_bloc = {"Q": {"Q1":.3, "Q2":.2, "R1":.25, "R2":.25},
                          "R": {"Q1":.15, "Q2":.35, "R1":.1, "R2":.4}}

bloc_voter_prop = {"Q": .7, "R": .3}
slate_to_candidates = {"Q": ["Q1", "Q2"],
                       "R": ["R1", "R2"]}
cohesion_parameters = {"Q": .9, "R": .7}

ac = AlternatingCrossover(candidates = candidates,
                          pref_interval_by_bloc =pref_interval_by_bloc,
                          bloc_voter_prop = bloc_voter_prop,
                          slate_to_candidates = slate_to_candidates,
                          cohesion_parameters = cohesion_parameters)
                        
ac.generate_profile(100)
```

    PreferenceProfile truncated to 15 ballots.

             Ballots  Weight
    (Q1, Q2, R2, R1)      19
    (Q1, Q2, R1, R2)      19
    (Q2, Q1, R1, R2)      13
    (Q2, Q1, R2, R1)      12
    (R1, R2, Q2, Q1)       8
    (R2, R1, Q1, Q2)       6
    (R1, R2, Q1, Q2)       4
    (R2, Q1, R1, Q2)       3
    (Q1, R2, Q2, R1)       3
    (R2, R1, Q2, Q1)       3
    (R1, Q1, R2, Q2)       2
    (R1, Q2, R2, Q1)       2
    (Q2, R2, Q1, R1)       2
    (Q1, R1, Q2, R2)       2
    (Q2, R1, Q1, R2)       2



# Cambridge Sampler

The Cambridge Sampler uses historical data from city council elections for Cambridge, MA to generate new ballots. Historical Cambridge results from 2009 to 2017 are built into `votekit` but you can also use different historical data by  pointing the `path` argument to a file of election results. The model assumes there is a majority and a minority bloc. Again there is a cohesion parameter measuring how often voters defect from the bloc. If voters vote with the bloc, they rank a bloc candidate first, then the ballot is filled by sampling from historical data matching the first-ranked candidate in the ballot. If they vote with the opposing bloc, they rank an opposing candidate first and then sample.

The historical ballots only give the order in which majority/minority bloc candidates are listed (for example, in the Cambridge data the ballot ('W', 'W', 'C') has two majority candidates ('W') and one minority candidate ('C')).
Once the model decides which ballot type a voter has, it fills in the ballot with actual candidates using the preference interval constructed ala Plackett-Luce.

Since it samples from historical data, it's possible to generate incomplete ballots.


```python
from votekit import CambridgeSampler

candidates = ["Q1", "Q2", "R1", "R2"]
slate_to_candidates = {"Q": ["Q1", "Q2"],
                       "R": ["R1", "R2"]}

bloc_voter_prop = {"Q": .6, "R": .4}
pref_interval_by_bloc = {
    "Q": {"Q1": 0.4, "Q2": 0.3, "R1": 0.2, "R2": 0.1},
    "R": {"Q1": 0.2, "Q2": 0.2, "R1": 0.3, "R2": 0.3},
}

cohesion_parameters = {"Q": .7, "R": .9}

cs = CambridgeSampler(pref_interval_by_bloc=pref_interval_by_bloc,
                      bloc_voter_prop=bloc_voter_prop, candidates=candidates,
                      slate_to_candidates=slate_to_candidates, cohesion_parameters=cohesion_parameters)

cs.generate_profile(number_of_ballots=100)
```



    PreferenceProfile truncated to 15 ballots.

             Ballots  Weight
    (R2, R1, Q2, Q1)       6
            (Q2, Q1)       5
    (Q2, R1, R2, Q1)       5
    (R1, R2, Q2, Q1)       5
               (R1,)       4
    (R2, Q2, Q1, R1)       4
    (R1, Q2, Q1, R2)       4
            (Q1, Q2)       4
    (Q1, Q2, R1, R2)       4
               (Q2,)       3
    (R2, R1, Q1, Q2)       3
            (R1, R2)       3
        (R1, Q1, Q2)       3
    (Q2, Q1, R2, R1)       3
    (R1, Q1, Q2, R2)       3



