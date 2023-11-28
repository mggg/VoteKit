from matplotlib import pyplot as plt  # type: ignore
from votekit.pref_profile import PreferenceProfile
from votekit.utils import first_place_votes, mentions, COLOR_LIST, borda_scores
from matplotlib.figure import Figure  # type: ignore


def plot_summary_stats(
    profile: PreferenceProfile, stat: str, multi_color: bool = True, title: str = ""
) -> Figure:
    """
    Plots histogram of election summary statistics.

    Args:
        profile (PreferenceProfile): A PreferenceProfile to visualize.
        stat (str): 'first place votes', 'mentions', or 'borda'.
        multi_color (bool, optional): If the bars should be multicolored. Defaults to True.
        title (str, optional): Title for the figure. Defaults to None.

    Returns:
        (Figure): A figure with the visualization.
    """
    stats = {
        "first place votes": first_place_votes,
        "mentions": mentions,
        "borda": borda_scores,
    }

    stat_func = stats[stat]
    data: dict = stat_func(profile)  # type: ignore

    if multi_color:
        colors = COLOR_LIST[: len(list(data.keys()))]
    else:
        colors = [COLOR_LIST[-1]]

    fig, ax = plt.subplots()

    candidates = profile.get_candidates(received_votes = False)
    y_data = [data[c] for c in candidates]
    
    ax.bar(candidates, y_data, color=colors, width=0.35)
    ax.set_xlabel("Candidates")
    ax.set_ylabel("Frequency")

    if title:
        ax.set_title(title)

    return fig
