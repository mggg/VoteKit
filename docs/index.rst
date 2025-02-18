.. VoteKit documentation master file, created by
   sphinx-quickstart on Wed May  8 08:33:07 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to VoteKit's documentation!
===================================

``VoteKit`` is a Swiss army knife for computational social choice research.

.. .. image:: https://circleci.com/gh/mggg/VoteKit.svg?style=svg
..     :target: https://circleci.com/gh/mggg/VoteKit
..     :alt: Build Status
.. image:: https://codecov.io/gh/mggg/VoteKit/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/mggg/VoteKit
    :alt: Code Coverage
.. image:: https://readthedocs.org/projects/votekit/badge/?version=latest
    :target: https://votekit.readthedocs.io/en/latest
    :alt: Documentation Status
.. image:: https://badge.fury.io/py/votekit.svg
    :target: https://pypi.org/project/votekit/
    :alt: PyPI Package

The project is in active development in the `mggg/VoteKit <https://github.com/mggg/VoteKit>`_ GitHub
repository, where :doc:`bug reports and feature requests <./package_info/issues>`, as well as 
:doc:`contributions <./package_info/contributing>`, are welcome.



.. include:: user/install.rst

.. toctree::
   :maxdepth: 1
   :caption: VoteKit Tutorial:

   user/tutorial/1_intro
   user/tutorial/2_real_and_simulated_profiles
   user/tutorial/3_viz
   user/tutorial/4_election_systems
   user/tutorial/5_score_based

.. toctree::
    :caption: Social Choice Reference
    :maxdepth: 1

    social_choice_docs/scr.rst
    

.. toctree::
    :caption: Package Information 
    :maxdepth: 1

    package_info/issues.rst
    package_info/contributing.rst
    package_info/related_resources.rst
    package_info/api.rst


