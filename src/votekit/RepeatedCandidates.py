import warnings
def contains_repeated_candidates(Ballot=None, df=None):
    if (Ballot is None and df is None) or (Ballot is not None and df is not None):
      raise TypeError("Sufficient Values weren't provided for the function to run. Please provide only one singular value, either a ballot or a dataframe.")

    elif Ballot is None and df is not None:


      for row_idx,(_,row) in enumerate(df.iterrows()):
        mylist=[]
        seen_cands=set()
        for col_idx,cand_set in enumerate(row):
          for candidate in cand_set:
            if candidate in seen_cands:
              for x,i in enumerate(row):
                if candidate in i:
                  mylist.append(x)
              warnings.warn(f'Duplicate Candidate spotted at these positions, {mylist} in ballot {row_idx+1}.')
              return True
            seen_cands.add(candidate)
      return False

    elif df is None and Ballot is not None:


      seen_cands = set()
      mylist=[]
      for cand_set in Ballot:
        for candidate in cand_set:
            if candidate in seen_cands:
                for x,i in enumerate(Ballot):
                    if candidate in i:
                        mylist.append(x)
                warnings.warn(f'Duplicate candidate spotted at these positions,{mylist}')
                return True
            seen_cands.add(candidate)
      return False

