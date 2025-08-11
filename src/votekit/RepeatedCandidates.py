import warnings
def repeat_rankings( ranking: Optional[tuple[frozenset[str], ...]]
                    ):
    seen_cands = set()
    for cand_set in ranking:
        for candidate in cand_set:
            if candidate in seen_cands:
                x=ranking.index(cand_set)
                for i in range(x+1,len(ranking)+1):
                    if candidate in ranking[i]:
                        y=i
                        break
                warnings.warn(f'Duplicate candidate spotted "{candidate}" at the number {x+1} choice and {y+1} choice  ,please fix.')
                return remove_repeated_candidates(ranking)
            seen_cands.add(candidate)
    return ranking
