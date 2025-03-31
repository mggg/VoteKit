from votekit.pref_profile import PreferenceProfile
import pandas as pd
from votekit.ballot import Ballot
from functools import partial
from typing import Optional
import numpy as np


def _convert_ranking_cols_to_ranking(
    row: pd.Series, ranking_cols: list[str]
) -> Optional[tuple[frozenset, ...]]:
    """
    Convert the ranking cols to a ranking tuple.

    """
    ranking = []

    for i, col in enumerate(ranking_cols):
        if pd.isna(row[col]):
            if not all(pd.isna(row[c]) for c in ranking_cols[i:]):
                raise ValueError(
                    (
                        f"Row {row} has NaN values between valid ranking positions."
                        " NaN values can only trail on a ranking."
                    )
                )

            break

        ranking.append(row[col])

    return tuple(ranking) if ranking else None


def _convert_row_to_ballot(
    row: pd.Series,
    ranking_cols: list[str],
    weight_col: str,
    id_col: str,
    voter_set_col: str,
    candidates: list[str],
) -> Ballot:
    """
    Convert a row of a properly formatted df to a Ballot.
    """

    ranking = _convert_ranking_cols_to_ranking(row, ranking_cols)
    scores = {c: row[c] if not pd.isna(row[c]) else 0 for c in candidates}
    id = row[id_col] if not pd.isna(row[id_col]) else None
    voter_set = row[voter_set_col] if not pd.isna(row[voter_set_col]) else set()
    weight = row[weight_col]

    return Ballot(
        ranking=ranking,
        scores=scores,
        weight=weight,
        id=id if id else None,
        voter_set=voter_set if voter_set else set(),
    )


def profile_to_df(profile: PreferenceProfile) -> pd.DataFrame:
    """
    Convert a PrefProfile into a pandas dataframe.

    """
    ballot_data = {
        c: [np.nan] * profile.num_ballots for c in sorted(profile.candidates)
    }
    ballot_data.update(
        {
            f"ranking_{i}": [np.nan] * profile.num_ballots
            for i in range(profile.max_ballot_length)
        }
    )
    ballot_data.update({"weight": [np.nan] * profile.num_ballots})
    ballot_data.update({"id": [np.nan] * profile.num_ballots})
    ballot_data.update({"voter_set": [np.nan] * profile.num_ballots})

    for i, b in enumerate(profile.ballots):
        ballot_data["weight"][i] = b.weight

        if b.id:
            ballot_data["id"][i] = b.id
        if b.voter_set:
            ballot_data["voter_set"][i] = b.voter_set

        if b.scores:
            for c, score in b.scores.items():
                ballot_data[c][i] = score

        if b.ranking:
            for j, cand_set in enumerate(b.ranking):
                ballot_data[f"ranking_{j}"][i] = cand_set

    df = pd.DataFrame(ballot_data)
    df.index.name = "Ballot Index"
    return df


def ballot_list_to_df(
    ballots: list[Ballot], max_ballot_length: int = 0, candidates: list[str] = []
) -> pd.DataFrame:
    """
    Convert a list of ballots into a pandas dataframe.

    """
    num_ballots = len(ballots)

    ballot_data = {
        "weight": [np.nan] * num_ballots,
        "id": [np.nan] * num_ballots,
        "voter_set": [np.nan] * num_ballots,
    }

    if candidates:
        ballot_data.update({c: [np.nan] * num_ballots for c in candidates})

    if max_ballot_length:
        ballot_data.update(
            {f"ranking_{i}": [np.nan] * num_ballots for i in range(max_ballot_length)}
        )

    for i, b in enumerate(ballots):
        ballot_data["weight"][i] = b.weight

        if b.id:
            ballot_data["id"][i] = b.id
        if b.voter_set:
            ballot_data["voter_set"][i] = b.voter_set

        if b.scores:
            for c, score in b.scores.items():
                if c not in ballot_data:
                    if candidates:
                        raise ValueError(
                            (
                                f"Candidate {c} found in ballot {b} "
                                "but not in candidate list {candidates}."
                            )
                        )
                    ballot_data[c] = [np.nan] * num_ballots
                ballot_data[c][i] = score

        if b.ranking:
            for j, cand_set in enumerate(b.ranking):
                if f"ranking_{j}" not in ballot_data:
                    if max_ballot_length:
                        raise ValueError(
                            (
                                f"Max ballot length {max_ballot_length} given "
                                f"but ballot {b} has length at least {j}."
                            )
                        )
                    ballot_data[f"ranking_{j}"] = [np.nan] * num_ballots

                ballot_data[f"ranking_{j}"][i] = cand_set

    df = pd.DataFrame(ballot_data)
    temp_col_order = [c for c in df.columns if "ranking" in c] + [
        "weight",
        "id",
        "voter_set",
    ]
    col_order = (
        candidates + temp_col_order
        if candidates
        else sorted([c for c in df.columns if c not in temp_col_order]) + temp_col_order
    )
    df = df[col_order]
    df.index.name = "Ballot Index"
    return df


def df_to_profile(
    df: pd.DataFrame,
    candidates: list[str],
    ranking_cols: list[str] = [],
    weight_col: str = "weight",
    id_col: str = "id",
    voter_set_col: str = "voter_set",
) -> PreferenceProfile:
    """
    Convert a df into a profile.

    ranking_cols (list[str], optional): This seems trickiest. Want to make sure that these are
        provided in decreasing order from top ranking to bottom ranking. Want to then use
        these to compute max ballot length.

    Need lots of safeguards here.
    """

    ballots = list(
        df.apply(
            partial(
                _convert_row_to_ballot,
                ranking_cols=ranking_cols,
                weight_col=weight_col,
                id_col=id_col,
                candidates=candidates,
                voter_set_col=voter_set_col,
            ),
            axis="columns",
        )
    )

    return PreferenceProfile(
        ballots=ballots, max_ballot_length=len(ranking_cols), candidates=candidates
    )
