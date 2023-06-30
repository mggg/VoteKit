from .candidate import Candidate, XCandidate
from .voter import Voter
import pandas as pd
import numpy as np

class CSVParser:
    """
    a class that takes in a csv and parses it into a list of Candidate and Voter objects
    """
    # def __init__(self):
    #     self.logger = logging.getLogger()
    
    # def deduplicate(self, ranking):
    #     """
    #     removes duplicates in a voter's ranking of candidates
    #     ex. ['c1', 'c1', 'c2] -> ['c1', '', 'c2']

    #     Args:
    #         ranking (list of string): the candidates ordered by voter's ranking

    #     Returns:
    #         a list of string: the ranking of candidates without duplicates
    #     """
    #     ranking_without_dups = []
    #     for cand in ranking:
    #         if cand in ranking and cand in ranking_without_dups:
    #             ranking_without_dups.append('')
    #         elif cand in ranking:
    #             ranking_without_dups.append(cand)
    #     return ranking_without_dups
    
    def reorder_cands(self, ranking):
        """
        reorder candidates so that real ...
        ex. ['c1', '', 'c2'] -> ['c1', 'c2', '']

        Args:
            ranking (list of string): the candidates ordered by voter's ranking

        Returns:
            a list of string: the ranking of candidates without duplicates
        """
        reordered = []
        for cand in ranking:
            if cand != "":
                reordered.append(cand)
        reordered += [''] * (len(ranking) - len(reordered))
        return reordered
    
    def make_cand(self, name, group=None):
        return Candidate(name=name, group=group)
    
    def make_voter(self, ranking, name):
        """
        creates a Voter object from ranking and name in the data

        Args:
            ranking (list of string): the candidates ordered by voter's ranking
            name (string): name of the voter

        Returns:
            Voter: a representation of the voter as an object
        """
        rankings = []

        for cand in ranking:
            if cand == '':
                rankings.append(XCandidate())
            else:
                cand = Candidate(name=cand)
                rankings.append(cand)
        
        voter = Voter(candidate_ranking=rankings, name=name)
        return voter
    
    # def format_check(self):
    #     assert os.path.isfile(self.path)
    #     assert os.path.getsize(self.path) != 0

    #     deduced_dialect = []

    #     with open(self.path, 'r') as f:
    #         sample = f.read(64)
    #         has_header = csv.Sniffer().has_header(sample)
    #         assert has_header
    #         deduced_dialect = csv.Sniffer().sniff(sample)
    #     f.close()
    #     return deduced_dialect

    def clean(self, ranking):
        """
        cleans the ranking according to a set of predefined rules

        Args:
            ranking (list of string): the candidates ordered by voter's ranking

        Returns:
            list of string: cleaned ranking
        """
        cleaned = self.deduplicate(ranking)
        cleaned = self.reorder_cands(cleaned)
        print(cleaned)
        return cleaned
            
    
    def parse_csv(self, path):
        voters = set()
        candidates = set()
    
        df = pd.read_csv(path)
        assert not df.empty

        for row in df.itertuples(index=False):
            voter_name = row[0]
            ranking = row[1:]
            cleaned_ranking = self.clean(ranking)
            new_voter = self.make_voter(cleaned_ranking, voter_name)
            voters.add(new_voter)

        df = df.to_string()
        unique_cands = np.unique(df)
        for cand in unique_cands:
            new_cand = self.make_cand(name=cand)
            candidates.add(new_cand)
        
        return voters, candidates

        # assert os.path.isfile(self.path)
        # assert os.path.getsize(self.path) != 0
        # with open(self.path, 'r') as f:
        #     reader = csv.reader(f)
        #     header = next(reader)
        #     assert header != []
        #     try:
        #         # assume that the first column is the voter's name
        #         for row in reader:
        #             voter_name = row[0]
        #             ranking = row[1:]
        #             cleaned_ranking = self.clean(ranking)
        #             new_voter = self.make_voter(cleaned_ranking, voter_name)
        #             self.voters.append(new_voter)

        #     except Exception as error:
        #         self.logger.error(error)
        #         raise
        #     f.close()


def main():
    csv = CSVParser()
    # print(csv.deduplicate(['c1', 'c1', 'c2', 'c1', 'c2']))
    # print(csv.deduplicate(['c1', 'c2', 'c1', 'c2']))
    # print(csv.deduplicate(['c1', 'c1', 'c2']))
    # print(csv.deduplicate(['c1', 'c1', 'c1']))

    # print(csv.reorder_cands(['c1', '', 'c2']))
    # print(csv.reorder_cands(['c1', '', '', 'c2', '']))
    # print(csv.reorder_cands(['', '', '', 'c2', '']))

    # csv.parse_csv('./data/test/empty.csv')
    # csv.parse_csv('./data/test/only_cols.csv')
    v, c = csv.parse_csv('./data/test/undervote.csv')
    for i in v:
        print([x for x in i.candidate_ranking])

if __name__ == "__main__":
    main()
