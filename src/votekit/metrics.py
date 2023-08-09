from votekit.cvr_loaders import rank_column_csv
from votekit.election_types import STV, fractional_transfer


fpath = "/Users/emariedelanuez/VoteKit/src/votekit/mn_clean_ballots.csv"


def loadin(fpath):
    """
    The goal of this file is to output the distance
    between the rankings produced by the STV
    results of a race and an unordered
    list of candidates.
    """
    data = rank_column_csv(fpath)
    return data


data = loadin(fpath)


def stv_mini(data):
    """ ""
    takes in: a csv of ddifferent ballots of an election
    which is an object of the ballot class.
    returns: ranked candidate list and unranked candidate list\n

    The function runs an stv election with fractional transfer
    and with magnitude 3 and saves the outcome as stv_outcome.
    Then the function uses list comprehension to make an
    ordered elected list and an ordered eliminated list.
    To make a ranking list, the function joins the elected and eliminated list.
    Lastly, the function makes an unordered candidate list
    by using the .get_candidates methed for stv elections.
    """
    testSTV = STV(data, fractional_transfer, seats=3)
    stv_outcome = testSTV.run_election()
    race_elected_list = [candidate for candidate in stv_outcome.get_all_winners()]
    race_eliminated_list = [candidate for candidate in stv_outcome.get_all_eliminated()]
    ranking_list = race_elected_list + race_eliminated_list
    candidate_list = data.get_candidates()
    return ranking_list, candidate_list


ranked_candidate_list, unranked_candidate_list = stv_mini(data)


def comparision(list1, list2):
    """
    takes in: two lists. one is an ordered list candidates
    and the other is an unordered list of candidates.
    returns: unranked values list\n

    This function first creates the model dictionary,
    which assigns indices to the ranked candidate list
    through dictionary comprehension.
    Then, we create a target dictionary, where the keys are candidates in the unordered list,
    and assigns its corresponding values from the model dictionary.
    Then, we make a list of the values in our target dictionary,
    which preserves the order of the "unranking."\n
    """
    n = len(list1)
    indices = list(range(n))
    model_dictionary = dict(zip(list1, indices))
    target_dictionary = {key: model_dictionary[key] for key in list2}
    unranked_values = list(target_dictionary.values())
    return unranked_values


def kendall_tau(unorderedlist):
    """ "
    kendall_tau:
    takes in: the unordered, but indexed candidate list from the comparision function
    returns: swap distance\n
    the function does the bubble sort on the indices of the unordered candidate list
    and returns a swap distance value.\n
    """
    swapcount = 0
    for index_i in range(len(unorderedlist)):
        for index_j in range(
            1, len(unorderedlist) - index_i
        ):  # look at all the other indices except for the one you originally chose
            if unorderedlist[index_j - 1] > unorderedlist[index_j]:
                swapcount += 1
                unorderedlist[index_j - 1], unorderedlist[index_j] = (
                    unorderedlist[index_j],
                    unorderedlist[index_j - 1],
                )
    return swapcount


rankinglist, candidates = stv_mini(data)
unordered_candidates = comparision(rankinglist, candidates)
bubble_distance = kendall_tau(unordered_candidates)
print(bubble_distance)
