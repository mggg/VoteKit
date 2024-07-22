from votekit.elections import ElectionState


def test_default_attributes():
    es = ElectionState()

    assert es.round_number == 0
    assert es.remaining == tuple([frozenset()])
    assert es.elected == tuple([frozenset()])
    assert es.eliminated == tuple([frozenset()])
    assert es.tiebreak_winners == {}
    assert es.scores == {}


def test_set_attributes():
    es = ElectionState(
        round_number=2,
        remaining=(
            frozenset({"Andrew", "Lauren"}),
            frozenset({"Chris", "Peter"}),
            frozenset({"Justin"}),
        ),
        elected=tuple([frozenset({"Moon"})]),
        eliminated=tuple([frozenset({"CJ"})]),
        tiebreak_winners={frozenset({"Moon", "Andrew", "Lauren"}): "Moon"},
        scores={
            "Chris": 12,
            "Peter": 12,
            "Moon": 25,
            "Andrew": 25,
            "Lauren": 25,
            "Justin": 4.0,
            "CJ": 0,
        },
    )

    assert es.round_number == 2
    assert es.remaining == ({"Andrew", "Lauren"}, {"Chris", "Peter"}, {"Justin"})
    assert es.elected == tuple([frozenset({"Moon"})])
    assert es.eliminated == tuple([frozenset({"CJ"})])
    assert es.tiebreak_winners == {frozenset({"Moon", "Andrew", "Lauren"}): "Moon"}
    assert es.scores == {
        "Chris": 12,
        "Peter": 12,
        "Moon": 25,
        "Andrew": 25,
        "Lauren": 25,
        "Justin": 4.0,
        "CJ": 0,
    }
