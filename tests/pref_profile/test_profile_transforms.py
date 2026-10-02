from votekit.ballot import RankBallot
from votekit.pref_profile import RankProfile
from votekit.pref_profile.transform import (
    BernoulliBallotProbability,
    SwapCandidates,
    transform,
)


def make_profile() -> RankProfile:
    return RankProfile(
        ballots=[
            RankBallot(ranking=[{"A"}, {"B"}, {"C"}], weight=10),
            RankBallot(ranking=[{"A"}, {"C"}, {"B"}], weight=10),
        ]
    )


def ranking_weights(profile: RankProfile) -> dict[tuple[str, ...], float]:
    """Map each ranking to its total weight, ignoring row order."""
    weights: dict[tuple[str, ...], float] = {}
    ranking_cols = [col for col in profile.df.columns if "Ranking_" in col]
    for _, row in profile.df.iterrows():
        ranking = tuple(next(iter(row[col])) for col in ranking_cols)
        weights[ranking] = weights.get(ranking, 0.0) + row["Weight"]
    return weights


def test_swap_candidates():
    transformed = transform(make_profile(), SwapCandidates("A", "B"))

    assert ranking_weights(transformed) == {("B", "A", "C"): 10.0, ("B", "C", "A"): 10.0}


def test_swap_candidates_with_probability():
    transformed = transform(
        make_profile(),
        SwapCandidates("A", "B"),
        probability=BernoulliBallotProbability(0.5, rng_seed=42),
    )

    assert ranking_weights(transformed) == {
        ("A", "B", "C"): 6.0,
        ("A", "C", "B"): 6.0,
        ("B", "A", "C"): 4.0,
        ("B", "C", "A"): 4.0,
    }
    assert transformed.total_ballot_wt == 20.0
