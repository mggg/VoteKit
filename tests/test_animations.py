from votekit.animations import STVAnimation
from votekit import Ballot, PreferenceProfile
from votekit.elections import STV
import pytest
import subprocess
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
import manim

# modified from STV wiki
# Election following the "happy path". One elimination or election per round. No ties. No exact quota matches. No funny business.


@pytest.fixture
def election_happy():
    profile_happy = PreferenceProfile(
        ballots=(
            Ballot(ranking=({"Orange"}, {"Pear"}), weight=3),
            Ballot(ranking=({"Pear"}, {"Strawberry"}, {"Cake"}), weight=8),
            Ballot(ranking=({"Strawberry"}, {"Orange"}, {"Pear"}), weight=1),
            Ballot(ranking=({"Cake"}, {"Chocolate"}), weight=3),
            Ballot(ranking=({"Chocolate"}, {"Cake"}, {"Burger"}), weight=1),
            Ballot(ranking=({"Burger"}, {"Chicken"}), weight=4),
            Ballot(ranking=({"Chicken"}, {"Chocolate"}, {"Burger"}), weight=3),
        ),
        max_ranking_length=3,
    )
    return STV(profile_happy, m=3)


def test_STVAnimation_init(election_happy):
    """
    Initialize an STVAnimation object and check some basic data.
    """
    animation = STVAnimation(election_happy)
    assert isinstance(animation.candidate_dict, dict)
    assert isinstance(animation.events, list)
    assert "Pear" in animation.candidate_dict.keys()
    assert animation.candidate_dict["Pear"]["support"] == 8


def images_match(img1_path: Path, img2_path: Path, tolerance: int = 2) -> bool:
    """
    Compare two images, return True if pixel differences are within tolerance.

    Args:
        img1_path (Path): Path to first image.
        img2_path (Path): Path to second image.
        tolerance (int): Maximum allowed average pixel difference (0-255 scale).

    Returns:
        True if images match within tolerance, False otherwise.
    """
    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")

    if img1.size != img2.size:
        return False

    arr1 = np.array(img1, dtype=np.float32)
    arr2 = np.array(img2, dtype=np.float32)

    diff = np.abs(arr1 - arr2)
    mean_diff = np.mean(diff)

    return mean_diff <= tolerance


@pytest.mark.slow
def test_stv_animation_video_screenshots(election_happy, tmp_path):
    """
    Render an STV animation video, extract frames at 3-second intervals,
    and compare them to saved snapshots.
    """

    # Configure manim to output to tmp_path to ensure media files are deleted after testing
    manim.config.media_dir = str(tmp_path / "media")

    animation = STVAnimation(election_happy, title="Test Election")
    animation.render()

    # Find the rendered video file
    video_dir = tmp_path / "media" / "videos" / "1080p60"
    video_files = list(video_dir.glob("*.mp4"))
    assert len(video_files) == 1, f"Expected 1 video file, found {len(video_files)}"
    video_path = video_files[0]

    # Get video duration using ffprobe
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",  # verbosity: errors only
            "-show_entries",
            "format=duration",  # Show duration
            "-of",
            "default=noprint_wrappers=1:nokey=1",  # Don't show extra info, just the number
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    duration = float(result.stdout.strip())

    # Extract frames every 3 seconds
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()

    timestamps = list(range(0, int(duration), 3))
    for i, timestamp in enumerate(timestamps):
        frame_path = frames_dir / f"frame_{i:02d}.png"
        subprocess.run(
            [
                "ffmpeg",
                "-y",  # Overwrite images
                "-ss",
                str(timestamp),  # Seek to timestamp
                "-i",
                str(video_path),  # Input file path
                "-frames:v",
                "1",  # One frame only
                str(frame_path),  # Output path
            ],
            check=True,
            capture_output=True,
        )

    # Compare to baseline images
    baseline_dir = Path(__file__).parent / "snapshots" / "animations"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    # If baselines don't exist or directory is empty, create them and fail the test
    existing_baselines = list(baseline_dir.glob("frame_*.png"))
    if not existing_baselines:
        for frame_file in sorted(frames_dir.glob("frame_*.png")):
            shutil.copy(frame_file, baseline_dir / frame_file.name)
        pytest.fail(
            f"Baseline images did not exist. Created {len(timestamps)} baseline images "
            f"in {baseline_dir}. Re-run the test to verify."
        )

    # Compare each frame to its baseline
    mismatched_frames = []
    for i in range(len(timestamps)):
        frame_path = frames_dir / f"frame_{i:02d}.png"
        baseline_path = baseline_dir / f"frame_{i:02d}.png"

        if not baseline_path.exists():
            mismatched_frames.append(f"frame_{i:02d}.png (baseline missing)")
            continue

        if not images_match(frame_path, baseline_path):
            mismatched_frames.append(f"frame_{i:02d}.png")

    assert not mismatched_frames, (
        f"Frame(s) did not match baseline: {', '.join(mismatched_frames)}. "
        f"Generated frames are in {frames_dir}"
    )
