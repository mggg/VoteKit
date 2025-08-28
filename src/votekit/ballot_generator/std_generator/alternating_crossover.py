from votekit.ballot_generator import BallotGenerator


class AlternatingCrossover(BallotGenerator):
    """
    Class for Alternating Crossover style of generating ballots.
    AC assumes that voters either rank all of their own blocs candidates above the other bloc,
    or the voters "crossover" and rank a candidate from the other bloc first, then alternate
    between candidates from their own bloc and the opposing.
    Should only be used when there are two blocs.

    Can be initialized with an interval or can be constructed with the Dirichlet distribution using
    the `from_params` method of `BallotGenerator`.

    Args:
        slate_to_candidates (dict): Dictionary whose keys are bloc names and whose
            values are lists of candidate strings that make up the slate.
        bloc_voter_prop (dict): Dictionary whose keys are bloc strings and values are floats
                denoting population share.
        pref_intervals_by_bloc (dict): Dictionary whose keys are bloc strings and values are
            dictionaries whose keys are bloc strings and values are ``PreferenceInterval`` objects.
        cohesion_parameters (dict): Dictionary mapping of bloc string to dictionary whose
            keys are bloc strings and values are cohesion parameters,
            eg. ``{'bloc_1': {'bloc_1': .7, 'bloc_2': .2, 'bloc_3':.1}}``

    Attributes:
        candidates (list): List of candidate strings.
        slate_to_candidates (dict): Dictionary whose keys are bloc names and whose
            values are lists of candidate strings that make up the slate.
        bloc_voter_prop (dict): Dictionary whose keys are bloc strings and values are floats
                denoting population share.
        pref_intervals_by_bloc (dict): Dictionary whose keys are bloc strings and values are
            dictionaries whose keys are bloc strings and values are ``PreferenceInterval`` objects.
        cohesion_parameters (dict): Dictionary mapping of bloc string to dictionary whose
            keys are bloc strings and values are cohesion parameters,
            eg. ``{'bloc_1': {'bloc_1': .7, 'bloc_2': .2, 'bloc_3':.1}}``
    """

    def __init__(
        self,
        cohesion_parameters: dict,
        **data,
    ):
        raise NotImplementedError(
            "This ballot generator type is no longer supported. "
            "The Plackett-Luce and Bradley-Terry models are preferred alternatives."
        )
