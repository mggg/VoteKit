from votekit.pref_profile import RankProfile
from votekit.ballot import RankBallot
from votekit.elections import ElectionState
from votekit.elections.election_types.ranking.stv.stv import AlbanySTV
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
CSV_DIR = BASE_DIR / "data/csv"

albany_profile = RankProfile.from_csv(CSV_DIR / "albany_profile.csv")

# all of these states except the last come from the official tabulation:
# https://alamedacountyca.gov/rovresults/rcv/248/rcvresults.htm?race=Albany%2F001-CityCouncil
albany_states = [
    ElectionState(
        round_number=0,
        remaining=(
            frozenset({"JOHN ANTHONY MIKI"}),
            frozenset({"JENNIFER HANSEN ROMERO"}),
            frozenset({"ROBIN D  LOPEZ"}),
            frozenset({"NICK PILCH"}),
            frozenset({"JEREMIAH GARRETT PINGUELO"}),
        ),
        scores={
            "JOHN ANTHONY MIKI": 2152,
            "JENNIFER HANSEN ROMERO": 1814,
            "ROBIN D  LOPEZ": 1642,
            "NICK PILCH": 1237,
            "JEREMIAH GARRETT PINGUELO": 314,
        },
    ),
    ElectionState(
        round_number=1,
        remaining=(
            frozenset({"JOHN ANTHONY MIKI"}),
            frozenset({"JENNIFER HANSEN ROMERO"}),
            frozenset({"ROBIN D  LOPEZ"}),
            frozenset({"NICK PILCH"}),
        ),
        eliminated=(frozenset({"JEREMIAH GARRETT PINGUELO"}),),
        scores={
            "JOHN ANTHONY MIKI": 2219,
            "JENNIFER HANSEN ROMERO": 1881,
            "ROBIN D  LOPEZ": 1742,
            "NICK PILCH": 1284,
        },
    ),
    ElectionState(
        round_number=2,
        remaining=(
            frozenset({"JOHN ANTHONY MIKI"}),
            frozenset({"ROBIN D  LOPEZ"}),
            frozenset({"JENNIFER HANSEN ROMERO"}),
        ),
        eliminated=(frozenset({"NICK PILCH"}),),
        scores={
            "JOHN ANTHONY MIKI": 2876,
            "JENNIFER HANSEN ROMERO": 2076,
            "ROBIN D  LOPEZ": 2111,
        },
    ),
    ElectionState(
        round_number=3,
        remaining=(
            frozenset({"ROBIN D  LOPEZ"}),
            frozenset({"JENNIFER HANSEN ROMERO"}),
        ),
        elected=(frozenset({"JOHN ANTHONY MIKI"}),),
        scores={
            "JENNIFER HANSEN ROMERO": 2260.958234,
            "ROBIN D  LOPEZ": 2396.136396,
        },
    ),
    ElectionState(
        round_number=4,
        remaining=(frozenset({"JENNIFER HANSEN ROMERO"}),),
        elected=(frozenset({"ROBIN D  LOPEZ"}),),
        scores={
            "JENNIFER HANSEN ROMERO": 2303.409599
        },  # this number needs to be checked by hand
    ),
]

albany_quotas = [2387.0, 2376.0, 2355.0, 2338.0]

albany_not_same_as_wigm_profile = RankProfile(
    ballots=tuple(
        [
            RankBallot(ranking=tuple(map(frozenset, ["A", "B"])), weight=90),
            RankBallot(ranking=tuple(map(frozenset, ["B"])), weight=67),
            RankBallot(ranking=tuple(map(frozenset, ["C"])), weight=70),
            RankBallot(ranking=tuple(map(frozenset, ["D", "A"])), weight=40),
            RankBallot(ranking=tuple(map(frozenset, ["E"])), weight=30),
        ]
    ),
    max_ranking_length=2,
)

not_same_as_wigm_quotas = [100.0, 90.0, 90.0, 76.0, 54.0]

three_winners_with_sharp_quotas_profile = RankProfile(
    ballots=tuple(
        [
            RankBallot(ranking=tuple(map(frozenset, ["A"])), weight=200),
            RankBallot(ranking=tuple(map(frozenset, ["B"])), weight=90),
            RankBallot(ranking=tuple(map(frozenset, ["C"])), weight=86),
            RankBallot(ranking=tuple(map(frozenset, ["D"])), weight=60),
            RankBallot(ranking=tuple(map(frozenset, ["E"])), weight=40),
        ]
    ),
    max_ranking_length=1,
)

three_winners_with_sharp_quotas_states = [
    ElectionState(
        round_number=0,
        remaining=tuple(map(frozenset, ["A", "B", "C", "D", "E"])),
        scores={
            "A": 200,
            "B": 90,
            "C": 86,
            "D": 60,
            "E": 40,
        },
    ),
    ElectionState(
        round_number=1,
        remaining=tuple(map(frozenset, ["B", "C", "D", "E"])),
        elected=(frozenset({"A"}),),
        scores={
            "B": 90,
            "C": 86,
            "D": 60,
            "E": 40,
        },
    ),
    ElectionState(
        round_number=2,
        remaining=tuple(map(frozenset, ["B", "C", "D"])),
        eliminated=(frozenset({"E"}),),
        scores={
            "B": 90,
            "C": 86,
            "D": 60,
        },
    ),
    ElectionState(
        round_number=3,
        remaining=tuple(map(frozenset, ["C", "D"])),
        elected=(frozenset({"B"}),),
        scores={
            "C": 86,
            "D": 60,
        },
    ),
    ElectionState(
        round_number=4,
        remaining=(frozenset({"C"}),),
        eliminated=(frozenset({"D"}),),
        scores={"C": 86},
    ),
]

three_winners_with_sharp_quotas_quotas = [120.0, 100.0, 90.0, 90.0, 75.0]

interaction_with_simultaneous = RankProfile(
    ballots=tuple(
        [
            RankBallot(ranking=tuple(map(frozenset, ["A"])), weight=140),
            RankBallot(ranking=tuple(map(frozenset, ["B", "C"])), weight=100),
            RankBallot(ranking=tuple(map(frozenset, ["C"])), weight=76),
            RankBallot(ranking=tuple(map(frozenset, ["D"])), weight=80),
        ]
    ),
    max_ranking_length=2,
)


def looser_equality_for_election_states(state1, state2, precision):
    if state1.remaining != state2.remaining:
        return False
    if state1.eliminated != state2.eliminated:
        return False
    if state1.elected != state2.elected:
        return False
    for cand, score in state1.scores.items():
        if cand not in state2.scores:
            return False
        if abs(score - state2.scores[cand]) > precision:
            return False
    return True


def test_albany_state_list():
    albany_elec = AlbanySTV(albany_profile, m=2)
    for i in [0, 1, 2]:  # these states all have integer scores
        assert albany_elec.election_states[i] == albany_states[i]
    for i in [3, 4]:  # these do not
        assert looser_equality_for_election_states(
            albany_elec.election_states[i], albany_states[i], precision=1e-3
        )
    for i in [0, 1, 2, 3]:
        assert albany_elec._data.play_by_play[i]["threshold"] == albany_quotas[i]


def test_albany_not_same_as_wigm():
    elec = AlbanySTV(albany_not_same_as_wigm_profile, m=2)
    play_by_play = elec._data.play_by_play
    assert all(
        play_by_play[i]["threshold"] == not_same_as_wigm_quotas[i]
        for i in range(len(play_by_play))
    )
    assert elec.get_elected() == (frozenset({"A"}), frozenset({"C"}))


def test_three_winners_with_sharp_quotas():
    elec = AlbanySTV(three_winners_with_sharp_quotas_profile, m=3)
    play_by_play = elec._data.play_by_play
    assert all(
        play_by_play[i]["threshold"] == three_winners_with_sharp_quotas_quotas[i]
        for i in [0, 1, 2, 3, 4]
    )
    assert all(
        elec.election_states[i] == three_winners_with_sharp_quotas_states[i]
        for i in [0, 1, 2, 3, 4]
    )


def test_interaction_with_simultaneous():
    # could maybe also test quota values for both elections? idk I'm tired
    simultaneous_elec = AlbanySTV(interaction_with_simultaneous, m=3, simultaneous=True)
    non_simultaneous_elec = AlbanySTV(
        interaction_with_simultaneous, m=3, simultaneous=False
    )
    assert simultaneous_elec.get_elected() == (
        frozenset({"A"}),
        frozenset({"B"}),
        frozenset({"D"}),
    )
    assert non_simultaneous_elec.get_elected() == (
        frozenset({"A"}),
        frozenset({"B"}),
        frozenset({"C"}),
    )
