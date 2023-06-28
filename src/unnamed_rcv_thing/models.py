from pydantic import BaseModel

# Example of immutable data model for results
class Outcome(BaseModel):
    """
    elected (a set of Candidate): candidates who pass a certain threshold to win an election
    eliminated (a set of Candidate): candidates who were eliminated (lost in the election)
    remaining (a set of Candidate): candidates who are still in the running
    rankings (a list of a set of Candidate): ranking of candidates with sets representing ties
    """

    # TODO: replace these with list/set[Candidate] once candidate is well-defined?
    remaining: set[str]
    elected: set[str] = set()
    eliminated: set[str] = set()
    # TODO: re-add this
    # rankings: list[set[str]]

    class Config:
        allow_mutation = False

    def add_winners_and_losers(self, winners: set[str], losers: set[str]):
        # example method, feel free to delete if not useful
        if not winners.issubset(self.remaining) or not losers.issubset(self.remaining):
            missing = (winners - self.remaining) | (losers - self.remaining)
            raise ValueError(f"Cannot promote winners, {missing} not in remaining")
        return Outcome(
            remaining=self.remaining - winners - losers,
            elected=self.elected | winners,
            eliminated=self.eliminated | losers,
        )
