from setuptools import find_packages, setup

setup(
    name="votekit",
    description="A Swiss army knife for computational social choice research",
    author="Metric Geometry and Gerrymandering Group",
    author_email="engineering@mggg.org",
    maintainer="Metric Geometry and Gerrymandering Group",
    maintainer_email="engineering@mggg.org",
    long_description="",
    url="https://github.com/mggg/VoteKit",
    packages=find_packages(exclude=("tests",)),
    keywords="VoteKit",
)
