import matplotlib.pyplot as plt  # type: ignore
from ..pref_profile import PreferenceProfile
from ..utils import first_place_votes, mentions, COLOR_LIST, borda_scores
from matplotlib.figure import Figure  # type: ignore


def plot_summary_stats(
    profile: PreferenceProfile, stat: str, multi_color: bool = True
) -> Figure:
    """
    Plots histogram of election summary statistics
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

    return fig


def accumlation_chart():
    """
    TODO: After 0.0.0 version release
    """
