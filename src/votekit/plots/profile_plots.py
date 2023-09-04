from matplotlib import pyplot as plt  # type: ignore
from votekit.pref_profile import PreferenceProfile
from votekit.utils import first_place_votes, mentions, COLOR_LIST, borda_scores
from matplotlib.figure import Figure  # type: ignore


def plot_summary_stats(
    profile: PreferenceProfile, stat: str, multi_color: bool = True, title: str = ""
) -> Figure:
    """
    Plots histogram of election summary statistics

    Args:
        profile (PreferenceProfile): a preference profile to visualize
        stat (str): 'first place votes', 'mentions', or 'borda'
        multi_color (bool, optional): if the bars should be multicolored. Defaults to True.
        title (str, optional): title for the figure. Defaults to None.

    Returns:
        Figure: a figure with the visualization
    """
    stats = {
        "first place votes": first_place_votes,
        "mentions": mentions,
        "borda": borda_scores,
    }

    stat_func = stats[stat]
    data: dict = stat_func(profile)  # type: ignore

    if stat == "first place votes":
        ylabel = "First Place Votes"
    elif stat == "mentions":
        ylabel = "Total Mentions"
    else:
        ylabel = "Borda Scores"

    if multi_color:
        colors = COLOR_LIST[: len(list(data.keys()))]
    else:
        colors = [COLOR_LIST[-1]]

    fig, ax = plt.subplots()

    ax.bar(data.keys(), data.values(), color=colors, width=0.35)
    ax.set_xlabel("Candidates")
    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title)

    return fig
