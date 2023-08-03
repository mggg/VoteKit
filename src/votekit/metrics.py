from votekit.cvr_loaders import rank_column_csv
from votekit.election_types import STV, fractional_transfer

fpath = "/Users/emariedelanuez/VoteKit/src/votekit/mn_clean_ballots.csv"  # path to ballot here


def loadin(path):  # load in data
    minneapolis_data = rank_column_csv(fpath)
    return minneapolis_data


def stv_mini(data):  # get stv ranking list and regular list of candidates
    testSTV = STV(minneapolis_data, fractional_transfer, seats=3)
    stv_outcome = testSTV.run_election()
    race_elected_list = [candidate for candidate in stv_outcome.elected]
    race_eliminated_list = [candidate for candidate in stv_outcome.eliminated]
    ranking_list = race_elected_list + race_eliminated_list
    candidate_list = minneapolis_data.get_candidates()

    return ranking_list, candidate_list


def comparision(
    list1, list2
):  # compare the intial ranking to whatever order the candidates are listed as
    n = len(list1)
    indices = list(range(n))
    model_dictionary = dict(zip(list1, indices))
    target_dictionary = {key: model_dictionary[key] for key in list2}
    unranked_values = list(target_dictionary.values())
    return unranked_values


def perform_bubble_sort(
    unorderedlist,
):  # do the actual bubble sort and get the distance

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


#########################################################################

"""
returns
"""

minneapolis_data = loadin(fpath)
minneapolis_rankinglist, minneapolis_candidates = stv_mini(minneapolis_data)
unordered_candidates = comparision(minneapolis_rankinglist, minneapolis_candidates)
bubble_distance = perform_bubble_sort(unordered_candidates)


"""



test1 = perform_bubble_sort(minneapolis_rankinglist)
test2 = perform_bubble_sort(minneapolis_candidates)

print("here")
print(test1)
print(test2)

indices_min_rank = [index for index, candidates in enumerate(minneapolis_rankinglist)]

print (indices_min_rank)
print()

def election_state_mini(data_outcome, 3):
    testyy = (data_outcome)
    minnieapolis_remaining = testyy.remaining
    minnieapolis_elected =  testyy.elected
    minnieapolis_eliminated = testyy.eliminated
    return minnieapolis_elected, minnieapolis_remaining, minnieapolis_eliminated

test= outcome_mini(minneapolis_outcome) 

print(test)
random_seed = 42
random.seed(random_seed)
random_numbers = random.sample(range(1, 12), 11)

def perform_bubble_sort(random_numbers):
    cmpcount, swapcount = 0, 0
    for j in range(len(random_numbers)):
        for i in range(1, len(random_numbers)-j):
            cmpcount += 1  # Increment cmpcount for each comparison
            if random_numbers[i-1] > random_numbers[i]:
                print(random_numbers[i-1],random_numbers[i] )
                swapcount += 1
                random_numbers[i-1], random_numbers[i] = random_numbers[i], random_numbers[i-1]
                print(random_numbers[i-1], random_numbers[i])
  
    return cmpcount, swapcount

chill = perform_bubble_sort(random_numbers)

print(chill)



def load_csv_data_from_file(fpath):
    data = []
    with open(fpath, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row)
    return data

data = load_csv_data_from_file(file_path)





def bublr(ballot: Ballot) -> Ballot:
    rank_list= ballot.ranking 


    
    return none



def undervote(ballot: Ballot) -> Ballot:
        rank_list = ballot.ranking
        cleaned_rank_list = [rank for rank in rank_list if None not in rank]
        return Ballot(
            id=ballot.id,
            ranking=cleaned_rank_list,
            weight=Fraction(ballot.weight),
            voters=ballot.voters,
        )

"""
