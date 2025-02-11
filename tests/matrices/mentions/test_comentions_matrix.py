from votekit.matrices import comentions_matrix
from votekit.ballot import Ballot
from votekit.pref_profile import PreferenceProfile

ballot_1 = Ballot(
    ranking=(frozenset({"Chris"}), frozenset({"Peter"}), frozenset({"Moon"}))
)
ballot_2 = Ballot(ranking=(frozenset({"Peter"}), frozenset({"Moon"})))
ballot_3 = Ballot(ranking=(frozenset({"Chris"}),))
ballot_4 = Ballot(
    ranking=(frozenset({"Chris"}), frozenset({"Peter"}), frozenset({"Moon"}))
)

pref_profile = PreferenceProfile(
    ballots=tuple(
        [ballot_1 for _ in range(5)]
        + [ballot_2 for _ in range(2)]
        + [ballot_3 for _ in range(1)]
        + [ballot_4 for _ in range(3)]
    )
)


def test_asym_comentions_matrix():
    asym_cmm = comentions_matrix(
        pref_profile, candidates=["Chris", "Peter", "Moon"], symmetric=False
    )

    assert asym_cmm[0][0] == 9
    assert asym_cmm[0][1] == 8
    assert asym_cmm[2][0] == 0

    assert asym_cmm[0][1] != asym_cmm[1][0]


def test_asym_comentions_matrix_cand_subset():
    asym_cmm = comentions_matrix(
        pref_profile, candidates=["Peter", "Moon"], symmetric=False
    )

    assert asym_cmm[0][0] == 10
    assert asym_cmm[0][1] == 10
    assert asym_cmm[1][0] == 0


def test_sym_comentions_matrix():
    sym_cmm = comentions_matrix(
        pref_profile, candidates=["Chris", "Peter", "Moon"], symmetric=True
    )

    assert sym_cmm[0][0] == 9
    assert sym_cmm[0][1] == 8

    for i in range(3):
        for j in range(i, 3):
            assert sym_cmm[i][j] == sym_cmm[j][i]


def test_sym_comentions_matrix_cand_subset():
    sym_cmm = comentions_matrix(
        pref_profile, candidates=["Peter", "Moon"], symmetric=True
    )

    assert sym_cmm[0][0] == 10
    assert sym_cmm[0][1] == 10
