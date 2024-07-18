from votekit.elections import ElectionState


def test_default_attributes():
    es = ElectionState()

    assert es.round_number == 0
    assert es.remaining == frozenset()
    assert es.elected == frozenset()
    assert es.eliminated == frozenset()
    assert es.tiebreak_winners == tuple([frozenset()])
    assert es.scores == {}


def test_set_attributes():
    es = ElectionState(
        round_number=2,
        remaining={"Chris", "Peter"},
        elected={"Moon"},
        eliminated={"CJ"},
        tiebreak_winners=({"Andrew", "Lauren"}, {"Justin"}),
        scores={
            "Chris": 50,
            "Peter": 50,
            "Moon": 100,
            "Andrew": 25,
            "Lauren": 25,
            "Justin": 4.0,
        },
    )

    assert es.round_number == 2
    assert es.remaining == frozenset({"Chris", "Peter"})
    assert es.elected == frozenset({"Moon"})
    assert es.eliminated == frozenset({"CJ"})
    assert es.tiebreak_winners == (
        frozenset({"Andrew", "Lauren"}),
        frozenset({"Justin"}),
    )
    assert es.scores == {
        "Chris": 50,
        "Peter": 50,
        "Moon": 100,
        "Andrew": 25,
        "Lauren": 25,
        "Justin": 4.0,
    }
