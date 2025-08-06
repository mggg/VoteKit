"""
Utility functions for the VoteKit toolkit, supporting ballot conversion, profile preparation,
consensus metrics, election result processing, and visualization for election simulations.

Dependencies:
    - numpy
    - pandas
    - seaborn
    - matplotlib
    - scikit-learn
    - imgkit
    - votekit
"""

from functools import lru_cache
import imgkit
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import manifold
from votekit.ballot import Ballot
from votekit.elections import IRV, Borda
from votekit.pref_profile import PreferenceProfile
from votekit.agreement import agreement

STATE = "redstate"

cand_colors = {
    "dem.biden": "#3FA2F6",  # PREZ
    "harris": "#3FA2F6",  # PREZ
    "dem.williamson": "#3FA2F6",
    "dem.phillips": "#3FA2F6",
    "ind.kennedy": "#B1AFFF",
    "kennedy": "#B1AFFF",
    "romney": "#B1AFFF",
    "manchin": "#B1AFFF",
    "ind.west": "#B1AFFF",
    "west": "#B1AFFF",
    "lib.oliver": "#FF8A08",
    "oliver": "#FF8A08",
    "rep.trump": "#FF8B8B",
    "trump": "#FF8B8B",
    "rep.ramaswamy": "#FF8B8B",
    "rep.desantis": "#FF8B8B",
    "rep.haley": "#EB4747",
    "grn.stein": "#A1DD70",
    "stein": "#A1DD70",
    "dem.alexandra": "#0052cc",  # SENATE
    "dem.dale": "#66e0ff",
    "dem.kenneth": "#3385ff",
    "ind.peter": "#c6b600",
    "ind.randy": "#ccb3ff",
    "lib.timothy": "#e67300",
    "rep.bruce": "#990000",
    "rep.jim": "#ff0000",
    "rep.kelly": "#4d0000",
    "rep.laura": "#ffaaaa",
    "rep.mike": "#ff8080",
    "dem.brown": "#3FA2F6",  # Ohio Senate
    "rep.dolan": "#ffaaaa",
    "rep.larose": "#FF8B8B",
    "rep.moreno": "#4d0000",
    "lake": "#990000",  # Arizona
    "lamb": "#990000",
    "ducey": "#ffaaaa",
    "sinema": "#ccb3ff",
    "gallego": "#66e0ff",
    "ashcroft": "#990000",  # Missouri
    "eigel": "#990000",
    "kehoe": "#ffaaaa",
    "quade": "#66e0ff",
    "hamra": "#66e0ff",
    "slant": "#c6b600",
    "slantz": "#c6b600",
    "slotkin.dd": "#66e0ff",  # Michigan
    "rogers.rr": "#990000",
    "craig.rf": "#FF8A08",
    "amash.rl": "#c6b600",
    "meijer.rb": "#ccb3ff",
    "harper.dd": "#66e0ff",
    "tlaib.dp": "#0052cc",
}

def agrmt(x: pd.Series) -> float:
    """
    Compute agreement (bimodality) score for categorical or ordinal data.

    Args:
        x (pd.Series): Data column containing votes or ratings.

    Returns:
        float: Agreement score :math:`a \\in [0, 1]`, or np.nan if fewer than three unique values.
    """
    if x.nunique() > 2:
        return agreement(x.value_counts().sort_index().to_numpy())
    return np.nan

def one_three_count(x: pd.Series) -> float:
    """
    Calculate the proportion of non-zero, non-null votes in a data column.

    Args:
        x (pd.Series): Data column containing votes or ratings.

    Returns:
        float: Proportion of non-zero, non-null entries, :math:`\\in [0, 1]`.
    """
    return ((x > 0) & x.notna()).sum() / x.shape[0]

def score_non_zero(x: pd.Series) -> int:
    """
    Count voters giving a non-zero score or approval.

    Args:
        x (pd.Series): Data column containing scores or approvals.

    Returns:
        int: Number of non-zero, non-null entries.
    """
    return ((x > 0) & x.notna()).sum()

def mentions(x: pd.Series) -> int:
    """
    Count non-null entries (mentions) in a data column.

    Args:
        x (pd.Series): Data column containing votes or ratings.

    Returns:
        int: Number of non-null entries.
    """
    return x.notna().sum()

def zero_three_count(x: pd.Series) -> float:
    """
    Calculate the proportion of non-null votes in a data column.

    Args:
        x (pd.Series): Data column containing votes or ratings.

    Returns:
        float: Proportion of non-null entries, :math:`\\in [0, 1]`.
    """
    return x.notna().sum() / x.shape[0]

def zero_count(x: pd.Series) -> float:
    """
    Calculate the proportion of zero-valued votes in a data column.

    Args:
        x (pd.Series): Data column containing votes or ratings.

    Returns:
        float: Proportion of zero-valued entries, :math:`\\in [0, 1]`.
    """
    return (x == 0).sum() / x.shape[0]

def extract_voting_method(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """
    Extract voting method descriptors from candidate columns, returning a cleaned DataFrame and methods.

    Args:
        df (pd.DataFrame): DataFrame with voting data, method codes (e.g., 'M-pickone') in values.
        cols (list[str]): Candidate column names to process.

    Returns:
        tuple:
            - pd.DataFrame: Cleaned DataFrame with method strings removed and values converted to numeric.
            - pd.Series: Voting method descriptors for each row (e.g., 'M-pickone').

    Raises:
        ValueError: If no valid voting method codes are found.
    """
    voting_method_col = (
        df.loc[:, cols]
        .iloc[:, 0]
        .str.replace(r"(M-.*?)[_-](.*)", lambda m: m.group(1), regex=True)
    )
    df.loc[:, cols] = (
        df.loc[:, cols]
        .apply(lambda x: x.str.replace("M-.*?[_-]", "", regex=True))
        .replace("NOVOTE", np.nan)
        .replace({"": np.nan})
        .astype(float)
    )
    if voting_method_col.isna().all():
        raise ValueError("No valid voting method codes found in the data.")
    return df, voting_method_col

def volume_sweeper(test: 'SensitivityTestBaseClass', places: list[str], title: str, volumes: list[float] = None) -> None:
    """
    Sweep over different dropout or jitter volumes, aggregate sensitivity statistics, and plot as a heatmap.

    The heatmap shows the proportion of iterations where each candidate deviates from their original ranking
    for each volume, with rows as candidates and columns as volume fractions.

    Args:
        test (SensitivityTestBaseClass): Instance of DropoutTest or JitterTest from votekit.sensitivity.
        places (list[str]): Candidate names to track, in format 'party.candidate' or 'candidate'.
        title (str): Plot title and base filename for the output PNG.
        volumes (list[float], optional): Volume fractions to sweep, in :math:`[0, 1]`. Defaults to 10 values from 0.05 to 0.95.

    Returns:
        None: Saves a heatmap plot to 'outputs/{STATE}/sensitivity/{title}.png'.
    """
    if volumes is None:
        volumes = np.linspace(0.05, 0.95, num=10)
    sweeped_data = {}
    for volume in volumes:
        sweeped_data[volume] = []
        res = test.run(volume=volume)
        place_cols = [col for col in res.columns if "Place" in col]
        for place_col, place in zip(place_cols, places):
            place_res = res.proportion[res.loc[:, place_col] != place].agg(sum)
            sweeped_data[volume].append(place_res)

    sweeped_data = pd.DataFrame(sweeped_data)
    places = [place.split(".") for place in places]
    try:
        places = [f"{cand.title()} ({party.upper()})" for party, cand in places]
    except:
        places = [place[0] for place in places]
    sweeped_data.index = [
        "-".join([place_col, place]) for place_col, place in zip(place_cols, places)
    ]
    sweeped_data.columns = [f"{col:.0%}" for col in sweeped_data.columns]

    if "Jitter" in str(test.__class__):
        cm = sns.light_palette("green", as_cmap=True)
        xlabel = "Volume of votes jittered"
    else:
        cm = sns.light_palette("blue", as_cmap=True)
        xlabel = "Volume of votes dropped"

    if test.voting_method == "pickone":
        vol_calc = "Total volume is the number of ballots times the number of candidates."
    elif test.voting_method == "rate":
        vol_calc = "Total volume is the number of ballots times the number of candidates times the maximum score."
    elif test.voting_method == "accept_approve":
        vol_calc = "Total volume is the number of ballots times the number of candidates times 3."
    elif test.voting_method == "approval":
        vol_calc = "Total volume is the number of ballots times the number of candidates times 2."
    elif test.voting_method == "rank":
        vol_calc = "Total volume is the number of ballots times the sum of ranks from 1 to min(max_rank, candidates)."
    label = f"Percentage of runs where candidate moves from place\n\nNote: Total volume refers to the total information available.\n{vol_calc}"

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        sweeped_data,
        annot=True,
        fmt=".0%",
        cmap=cm,
        vmin=0,
        vmax=1,
        cbar_kws={
            "location": "bottom",
            "format": lambda x, y: f"{x:.0%}",
            "label": label,
            "shrink": 0.8,
        },
        ax=ax,
    )
    ax.set(xlabel=xlabel, ylabel="Original Result Places")
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    fig.subplots_adjust(top=0.9)
    plt.suptitle(title)
    ax.text(
        0,
        -1.5,
        (
            f"N candidates - {len(places)}, "
            f"N voters - {test.df.shape[0]}, "
            f"Total Volume - {test.total_volume}, "
            f"Voting Method - {test.voting_method}"
        ),
        ha="left",
    )
    plt.tight_layout()
    plt.savefig(f"outputs/{STATE}/sensitivity/{title}.png")
    plt.close()

def generate_accept_approve_ballots(df: pd.DataFrame) -> list[Ballot]:
    """
    Generate accept/approve-style Ballot objects from a DataFrame.

    Args:
        df (pd.DataFrame): Columns are candidates; values are 1=first-choice, 2=acceptable, other=reject.

    Returns:
        list[votekit.ballot.Ballot]: Ballots for accept/approve voting, with rankings as [first-choice, acceptable, reject].
    """
    ballots = [
        Ballot([
            frozenset(x[x == 1].index),
            frozenset(x[x == 2].index),
            frozenset(x[(x != 1) & (x != 2)].index),
        ])
        for _, x in df.iterrows()
    ]
    return ballots

def generate_accept_approve_profile(df: pd.DataFrame) -> PreferenceProfile:
    """
    Generate a PreferenceProfile from accept/approve ballots.

    Args:
        df (pd.DataFrame): DataFrame with accept/approve voting data.

    Returns:
        votekit.pref_profile.PreferenceProfile: Profile for accept/approve elections.
    """
    return PreferenceProfile(ballots=generate_accept_approve_ballots(df))

def process_row(row: pd.Series) -> list[set]:
    """
    Process a DataFrame row into a ranked ballot format, sorting by values.

    Args:
        row (pd.Series): Row of ranking data, with candidate names as index and ranks as values.

    Returns:
        list[set]: List of sets representing the ranking, with each set containing candidates at the same rank.
    """
    sorted_items = sorted(row.dropna().items(), key=lambda item: item[1])
    return [{k} for k, _ in sorted_items]

def generate_irv_ballots(df: pd.DataFrame, cols: list[str]) -> list[Ballot]:
    """
    Create Ballot objects for Instant Runoff Voting (IRV) from a DataFrame of ranked votes.

    Args:
        df (pd.DataFrame): DataFrame with ranking data, shape :math:`(n_{\\text{ballots}}, n_{\\text{candidates}})`.
        cols (list[str]): Candidate column names.

    Returns:
        list[votekit.ballot.Ballot]: Ballots suitable for IRV simulation.
    """
    ballots = df[cols].apply(process_row, axis=1)
    prepped_ballots = [Ballot(b) for b in ballots]
    return prepped_ballots

def run_irv_election(df: pd.DataFrame, cols: list[str]) -> tuple[list, IRV]:
    """
    Run an Instant Runoff Voting (IRV) election given a DataFrame of ballot rankings.

    Args:
        df (pd.DataFrame): DataFrame with ranking data, shape :math:`(n_{\\text{ballots}}, n_{\\text{candidates}})`.
        cols (list[str]): Columns to use as candidate names.

    Returns:
        tuple:
            - list: IRV ranking of candidates.
            - votekit.elections.IRV: IRV Election object.
    """
    irv_profile = PreferenceProfile(ballots=generate_irv_ballots(df, cols))
    irv_election = IRV(profile=irv_profile)
    irv_result = irv_election.run_election()
    irv_rankings = irv_result.rankings()
    irv_places = []
    for ranking in irv_rankings:
        irv_places += list(ranking)
    return irv_places, irv_result

@lru_cache
def ballots_to_array(x: Ballot) -> np.ndarray:
    """
    Convert a Ballot object with a ranking attribute to a numpy array of ordinal ranks.

    Args:
        x (votekit.ballot.Ballot): Ballot object with ranking attribute.

    Returns:
        np.ndarray: Array of ranks, resized to length 5.
    """
    r = np.array(x.ranking)
    r.resize(5)
    return r

@lru_cache
def array_to_ballots_hashable(x: np.ndarray) -> Ballot:
    """
    Convert a numpy array of ranks to a hashable Ballot object.

    Args:
        x (np.ndarray): Array of ranks.

    Returns:
        votekit.ballot.Ballot: Ballot object with ranking attribute.
    """
    return Ballot(x)

def array_to_ballots(arr: np.ndarray, labels: list[str] = None) -> list[Ballot]:
    """
    Convert a 2D numpy array of ballots to a list of Ballot objects for ranked voting.

    Each row is interpreted as a ranking: lowest value = highest rank. Equal values are grouped as tied.

    Args:
        arr (np.ndarray): Ballot array, shape :math:`(n_{\\text{ballots}}, n_{\\text{candidates}})`.
        labels (list[str], optional): List of candidate names. Defaults to numeric indices as strings.

    Returns:
        list[votekit.ballot.Ballot]: List of Ballot objects reflecting the ordering in each row.
    """
    n_candidates = arr.shape[1]
    if labels is None:
        labels = [str(i) for i in range(n_candidates)]
    ballots = []
    for row in arr:
        sorted_pairs = sorted(zip(row, labels))
        ranking = []
        prev_rank = None
        curr_level = set()
        for score, label in sorted_pairs:
            if prev_rank is None or score != prev_rank:
                if curr_level:
                    ranking.append(curr_level)
                curr_level = set()
            curr_level.add(label)
            prev_rank = score
        if curr_level:
            ranking.append(curr_level)
        ballots.append(Ballot(ranking))
    return ballots

def run_election(election_name: str, df: pd.DataFrame, cols: list[str], voting_method: str) -> pd.DataFrame:
    """
    Run an election with the specified voting method and compute summary statistics.

    Args:
        election_name (str): Name of the election for labeling results.
        df (pd.DataFrame): DataFrame with voting data, shape :math:`(n_{\\text{ballots}}, n_{\\text{candidates}})`.
        cols (list[str]): Candidate column names.
        voting_method (str): Voting method, one of ['M-pickone', 'M-irv', 'M-accept_approve', 'M-approval', 'M-sco', 'M-backup'].

    Returns:
        pd.DataFrame: Election results with columns for candidate, place, first-place votes, margins, and other metrics.
    """
    n = df.shape[0]
    df_temp = df.copy()

    if voting_method == "M-irv":
        irv_places, irv_result = run_irv_election(df_temp, cols)
        irv_result_df = pd.Series(irv_places).reset_index()
        irv_result_df.loc[:, "irv_result"] = irv_result_df.loc[:, "index"] + 1
        irv_result_df = irv_result_df.drop(columns="index")
        irv_result_df.columns = ["variable_0", "irv_result"]
        irv_scores_df = pd.DataFrame(
            [
                (cand, score)
                for score, cands in irv_result.election_scores.items()
                for cand in cands
            ],
            columns=["variable_0", "irv_score"],
        )
        irv_scores_df.loc[:, "irv_margin"] = (
            irv_scores_df.loc[:, "irv_score"] - irv_scores_df.loc[:, "irv_score"].max()
        )
        irv_result_df = pd.merge(
            irv_result_df,
            irv_scores_df,
            left_on="variable_0",
            right_on="variable_0",
        )
        borda = Borda(PreferenceProfile(ballots=generate_irv_ballots(df_temp, cols)))
        borda_result = borda.run_election()
        borda_ranks = pd.Series(borda_result.rankings()).reset_index()
        borda_ranks.loc[:, "index"] = borda_ranks.loc[:, "index"] + 1
        borda_ranks.columns = ["Borda Place", "variable_0"]
        borda_scores = pd.DataFrame(
            [
                (cand, score)
                for score, cands in borda_result.election_scores.items()
                for cand in cands
            ],
            columns=["variable_0", "Borda Score"],
        )
        irv_result_df = pd.merge(
            irv_result_df, borda_ranks, left_on="variable_0", right_on="variable_0"
        )
        irv_result_df = pd.merge(
            irv_result_df, borda_scores, left_on="variable_0", right_on="variable_0"
        )

    if "M-sco" in voting_method:
        sco_result = (
            df_temp.loc[:, cols]
            .sum()
            .reset_index()
            .rename(columns={"index": "variable_0", 0: "Total Score"})
        )
        sco_result.loc[:, "Score mean"] = sco_result.loc[:, "Total Score"] / n
        sco_result.loc[:, "Score margin"] = (
            sco_result.loc[:, "Total Score"] - sco_result.loc[:, "Total Score"].max()
        )
        sco_result.loc[:, "sco_result"] = (
            sco_result.loc[:, "Total Score"].rank(ascending=False).astype(int)
        )

    if "M-backup" == voting_method:
        merged_votes = (
            df_temp.loc[:, cols]
            .sum()
            .reset_index()
            .rename(columns={"index": "variable_0", 0: "backup_votes"})
        )
        merged_votes.loc[:, "backup share"] = (
            merged_votes.loc[:, "backup_votes"] / n
        )
        merged_votes.loc[:, "result"] = (
            merged_votes.loc[:, "backup_votes"].rank(ascending=False).astype(int)
        )
        final_votes = merged_votes.loc[:, ["variable_0", "backup_votes"]].rename(
            columns={"backup_votes": "final votes"}
        )
        final_votes.loc[:, "final share"] = final_votes.loc[:, "final votes"] / n
        merged_votes = pd.merge(
            merged_votes,
            final_votes,
            left_on="variable_0",
            right_on="variable_0",
        )
        merged_votes.loc[:, "Backup Added"] = (
            merged_votes.loc[:, "final votes"] - merged_votes.loc[:, "backup_votes"]
        )

    if "M-accept_approve" == voting_method:
        aa_first_choice = (
            (df_temp.loc[:, cols] == 1).sum().sort_values(ascending=False).index[0]
        )
        aa_alts = (df_temp.loc[:, cols] == 2).sum().sort_values(ascending=False)
        aa_rest_places = aa_alts[aa_alts.index != aa_first_choice].index.to_list()
        aa_result = [aa_first_choice] + aa_rest_places
        aa_result_df = pd.Series(aa_result).reset_index()
        aa_result_df.loc[:, "aa_result"] = aa_result_df.loc[:, "index"] + 1
        aa_result_df = aa_result_df.drop(columns="index")
        aa_result_df.columns = ["variable_0", "aa_result"]

        aa_fc_scores_df = (df_temp.loc[:, cols] == 1).sum().reset_index()
        aa_fc_scores_df.columns = ["variable_0", "aa_fc_score"]
        aa_fc_scores_df.loc[:, "aa_fc_score_mean"] = (
            aa_fc_scores_df.loc[:, "aa_fc_score"] / n
        )
        aa_fc_scores_df.loc[:, "aa_fc_margin"] = (
            aa_fc_scores_df.loc[:, "aa_fc_score"]
            - aa_fc_scores_df.loc[:, "aa_fc_score"].max()
        )

        aa_aa_scores_df = (df_temp.loc[:, cols] == 2).sum().reset_index()
        aa_aa_scores_df.columns = ["variable_0", "aa_aa_score"]
        aa_aa_scores_df.loc[:, "aa_aa_score_share"] = (
            aa_aa_scores_df.loc[:, "aa_aa_score"] / n
        )
        aa_aa_scores_df.loc[:, "aa_aa_volume"] = (
            aa_aa_scores_df.loc[:, "aa_aa_score"]
            / aa_aa_scores_df.loc[:, "aa_aa_score"].sum()
        )
        aa_aa_scores_df.loc[:, "aa_aa_margin"] = (
            aa_aa_scores_df.loc[:, "aa_aa_score"]
            - aa_aa_scores_df.loc[:, "aa_aa_score"].max()
        )

        aa_result_df = pd.merge(
            aa_result_df,
            aa_fc_scores_df,
            left_on="variable_0",
            right_on="variable_0",
        )
        aa_result_df = pd.merge(
            aa_result_df,
            aa_aa_scores_df,
            left_on="variable_0",
            right_on="variable_0",
        )

    res = df_temp.loc[:, cols].agg([lambda x: (x == 1).sum(), agrmt, score_non_zero, mentions])
    res = res.T
    res.columns = ["First Place Votes"] + res.columns.to_list()[1:]
    res.loc[:, "FPV Share"] = res.loc[:, "First Place Votes"] / n
    res.loc[:, "winner_score"] = res.loc[:, "First Place Votes"].max()
    res.loc[:, "FPV margin"] = res.loc[:, "First Place Votes"] - res.winner_score
    res.loc[:, "margin"] = res.loc[:, "First Place Votes"] - res.winner_score
    res.loc[:, "place"] = res["First Place Votes"].rank(ascending=False)
    res.loc[:, "mentions share"] = res["mentions"] / n
    res.loc[:, "score_non_zero share"] = res["score_non_zero"] / n
    res.loc[:, "election_name"] = election_name
    res.loc[:, "candidate"] = res.index

    extra_cols = []
    if "M-irv" == voting_method:
        res = pd.merge(res, irv_result_df, left_on="candidate", right_on="variable_0")
        res.loc[:, "place"] = res.loc[:, "irv_result"]
        res = res.drop(columns=["irv_result", "irv_score", "irv_margin"])
        res.loc[:, "Borda Place Diff"] = res.loc[:, "place"] - res.loc[:, "Borda Place"]
        extra_cols = ["Borda Place", "Borda Score", "Borda Place Diff"]
    if "M-accept_approve" == voting_method or "M-approval" == voting_method:
        res = pd.merge(res, aa_result_df, left_on="candidate", right_on="variable_0")
        res.loc[:, "place"] = res.loc[:, "aa_result"]
        res.loc[:, "Approval Votes"] = res.loc[:, "aa_aa_score"]
        res.loc[:, "Approval Votes share"] = res.loc[:, "aa_aa_score"] / n
        res.loc[:, "Approval Votes volume"] = res.loc[:, "aa_aa_volume"]
        res.loc[:, "Approval Votes margin"] = res.loc[:, "aa_aa_margin"]
        extra_cols = [
            "Approval Votes",
            "Approval Votes share",
            "Approval Votes volume",
            "Approval Votes margin",
        ]
    if "M-sco" in voting_method:
        res = pd.merge(res, sco_result, left_on="candidate", right_on="variable_0")
        res.loc[:, "place"] = res.loc[:, "sco_result"]
        extra_cols = [
            "Total Score",
            "Score margin",
            "Score mean",
            "score_non_zero",
            "score_non_zero share",
        ]
    if "M-backup" == voting_method:
        res = pd.merge(res, merged_votes, left_on="candidate", right_on="variable_0")
        res.loc[:, "place"] = res.loc[:, "result"]
        extra_cols = [
            "backup_votes",
            "backup share",
            "final votes",
            "final share",
            "Backup Added",
        ]
    res = (
        res.sort_values(["place"], ascending=[True])
        .round(3)
        .drop(columns="winner_score")
    )
    res.loc[:, "candidate"] = (
        res.loc[:, "candidate"].str.split("-", expand=True).iloc[:, -1]
    )
    res.loc[:, "voting_method"] = voting_method
    res.loc[:, "n"] = n
    res.loc[:, "election_name-voting_method-n"] = (
        res.election_name + " | " + res.voting_method + " | n=" + res.n.astype(str)
    )
    res = res.loc[
        :,
        [
            "election_name-voting_method-n",
            "candidate",
            "place",
            "First Place Votes",
            "FPV margin",
            "FPV Share",
            "agrmt",
            "mentions",
            "mentions share",
        ]
        + extra_cols,
    ]
    res = res.rename(columns={"agrmt": "Bimodal Score"})
    return res

def compute_mds(dist_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 2D Multidimensional Scaling (MDS) coordinates for visualization from a distance matrix.

    Args:
        dist_matrix (pd.DataFrame): Pairwise candidate distances, shape :math:`(n_{\\text{candidates}}, n_{\\text{candidates}})`.

    Returns:
        pd.DataFrame: DataFrame with columns ['candidate', 'x', 'y'] containing 2D coordinates.

    Raises:
        ValueError: If dist_matrix is not square.
    """
    if dist_matrix.shape[0] != dist_matrix.shape[1]:
        raise ValueError("Distance matrix must be square.")
    mds = manifold.MDS(
        n_components=2,
        max_iter=3000,
        eps=1e-9,
        dissimilarity="precomputed",
        n_jobs=1,
        normalized_stress="auto",
        random_state=42,
    )
    pos = mds.fit(dist_matrix).embedding_
    coord_dict = pd.DataFrame(pos, index=dist_matrix.index).reset_index()
    coord_dict.columns = ["candidate", "x", "y"]
    return coord_dict

def cand_dist(df: pd.DataFrame, title: str) -> None:
    """
    Plot candidate distance visualizations using rank difference and correlation metrics.

    Creates two subplots: a scatter plot of candidate rank differences and a scatter plot of candidate correlations,
    both using MDS coordinates.

    Args:
        df (pd.DataFrame): DataFrame with voting data, shape :math:`(n_{\\text{ballots}}, n_{\\text{candidates}})`.
        title (str): Plot title and base filename for the output PNG.

    Returns:
        None: Saves the plot to 'outputs/{STATE}/distance/{title}.png'.
    """
    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    cand_diff(df, "Candidate rank difference", ax[1])
    cand_corr(df, "Candidate rank correlation", ax[0])
    plt.suptitle(title, weight="bold", size=20)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    ax[0].get_xaxis().set_visible(False)
    ax[1].get_xaxis().set_visible(False)
    ax[0].get_yaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig(f"outputs/{STATE}/distance/{title}.png")
    plt.close()

def cand_diff(df: pd.DataFrame, title: str, ax: plt.Axes = None) -> pd.DataFrame:
    """
    Compute and visualize candidate rank differences using MDS coordinates.

    Args:
        df (pd.DataFrame): DataFrame with voting data, shape :math:`(n_{\\text{ballots}}, n_{\\text{candidates}})`.
        title (str): Title for the scatter plot.
        ax (plt.Axes, optional): Matplotlib axes object for plotting. If None, a new figure is created.

    Returns:
        pd.DataFrame: Pivot table of mean absolute rank differences between candidates.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 6))
    cand_ab_diffs = []
    for i, cand_a in enumerate(df.columns):
        for cand_b in df.columns:
            cand_ab = df.loc[:, cand_a] - df.loc[:, cand_b]
            cand_ab = np.abs(cand_ab)
            cand_ab = cand_ab.dropna()
            cand_ab_diff = cand_ab.mean()
            cand_ab_diffs.append((cand_a, cand_b, cand_ab_diff))

    cand_ab_diffs = pd.DataFrame(cand_ab_diffs)
    cand_ab_diffs.columns = ["cand_a", "cand_b", "dist"]
    order = cand_ab_diffs.loc[:, "cand_a"].drop_duplicates()
    cand_ab_diffs = cand_ab_diffs.pivot(index="cand_b", columns="cand_a")
    cand_ab_diffs.columns = cand_ab_diffs.columns.droplevel(0)
    cand_ab_diffs = cand_ab_diffs.loc[order, order]
    cand_ab_diffs.index.name = ""
    cand_ab_diffs.columns.name = ""
    sns.scatter
