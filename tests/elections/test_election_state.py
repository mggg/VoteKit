from votekit.elections import ElectionState


def test_default_attributes():
    es = ElectionState()

    assert es.round_number == 0
    assert es.remaining == tuple([frozenset()])
    assert es.elected == tuple([frozenset()])
    assert es.eliminated == tuple([frozenset()])
    assert es.tiebreaks == {}
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
        tiebreaks={
            frozenset({"Moon", "Andrew", "Lauren"}): (
                frozenset({"Moon"}),
                frozenset({"Andrew"}),
                frozenset({"Lauren"}),
            )
        },
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
    assert es.tiebreaks == {
        frozenset({"Moon", "Andrew", "Lauren"}): (
            frozenset({"Moon"}),
            frozenset({"Andrew"}),
            frozenset({"Lauren"}),
        )
    }
    assert es.scores == {
        "Chris": 12,
        "Peter": 12,
        "Moon": 25,
        "Andrew": 25,
        "Lauren": 25,
        "Justin": 4.0,
        "CJ": 0,
    }
