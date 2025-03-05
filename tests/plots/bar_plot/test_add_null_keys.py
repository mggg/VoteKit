from votekit.plots.bar_plot import add_null_keys


def test_add_null_keys():
    data = {
        "Profile 1": {"Peter": 6, "Chris": 4, "Moon": 2},
        "Profile 2": {"Chris": 4, "Peter": 3, "Mala": 1},
    }
    data = add_null_keys(data)

    assert data == {
        "Profile 1": {"Peter": 6, "Chris": 4, "Moon": 2, "Mala": 0},
        "Profile 2": {"Chris": 4, "Peter": 3, "Mala": 1, "Moon": 0},
    }
