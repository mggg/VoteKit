# import pytest
# from votekit.models import Outcome


# def test_outcome_promote_winners():
#     orig = Outcome(remaining=["A", "B", "C"])
#     new = orig.add_winners_and_losers({"B"}, {"A"})
#     assert new.elected == {"B"}
#     assert new.eliminated == {"A"}
#     assert new.remaining == {"C"}


# def test_outcome_promote_winners_error():
#     orig = Outcome(remaining=["A", "B", "C"])
#     with pytest.raises(ValueError) as exc:
#         orig.add_winners_and_losers({"D"}, {"E"})

#     # ensure error message mentions both bad candidates
#     assert "D" in str(exc.value)
#     assert "E" in str(exc.value)


# def test_outcome_immutable():
#     # not that important to test this since it is implemented by pydantic
#     # but provided as a check that we remembered to set allow_mutation = False
#     orig = Outcome(remaining={"A"})
#     with pytest.raises(TypeError):
#         orig.remaining = {"A", "B", "C"}
