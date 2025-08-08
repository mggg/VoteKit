from votekit import PreferenceProfile
import numpy as np
import pandas as pd
from votekit.cleaning import remove_and_condense


def make_random_profile(n_voters: int, cand_list: list[str]) -> PreferenceProfile:
    weights = np.unique_counts(list(map(int, np.random.gamma(5, 1, n_voters))))[1]
    n_ballots = len(weights)

    cand_to_ratings = {
        c: list(map(int, np.random.choice(range(0, 5), size=n_ballots, replace=True)))
        for c in cand_list
    }

    df = pd.DataFrame(cand_to_ratings)
    df["Weight"] = list(map(int, weights))
    df["Voter Set"] = [set()] * n_ballots
    df["Ballot Index"] = range(n_ballots)
    df.set_index("Ballot Index", inplace=True)
    return PreferenceProfile(candidates=tuple(cand_list), df=df, contains_scores=True)


prof = make_random_profile(10, ["A", "B", "C"])
remove_and_condense(["A", "B"], prof)
