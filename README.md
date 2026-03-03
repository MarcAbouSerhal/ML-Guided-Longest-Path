# Machine Learning for Longest Paths

This repository implements tools to generate, analyze and (approximately) solve the Longest Simple Path problem on small graphs. 

Contents
- C++ code for generating data (extracting features and real longest path values):
  - [`config.hpp`](cpp_extraction/config.hpp) defines global configuration and shared data structures used throughout the extraction pipeline. This includes graph size limits, type aliases, bitset-based integer sets, adjacency list representations, and other common utilities.
  - [`algorithms.cpp`](cpp_extraction/algorithms.cpp) implements core graph algorithms and helper routines, including:
    - [`get_sccs`](cpp_extraction/algorithms.cpp) for SCC computation
    - [`fill_dp_grid`](cpp_extraction/algorithms.cpp) for LP DP over subsets
  - [`feature_calculation.cpp`](cpp_extraction/feature_calculation.cpp) extracts feature vectors from graphs.
- Python code for training the model, running the algorithm and evaluation:
  - [`algorithm.py`](algorithm.py) implements the pathfinding algorithm using beam search with specified width.
  - [`evaluate.py`](evaluation/evaluate.py) compares between the real longest path in a graph and the approximate longest path found by the algorithm.
  - [`model`](model/) contains files for training a neural network, and our trained model as a [`pkl`] file.
- Datasets & computed results:
  - [dataset.csv](datasets/dataset.csv) which was extracted from graphs in [graphs_adjlist_sparse.txt](datasets/graphs_adjlist_sparse.txt).
  - [longest_paths.txt](evaluation/longest_paths.txt), [approximate_paths.txt](evaluation/approximate_paths.txt), and [greedy_paths.txt](evaluation/greedy_paths.txt) which were ran on graphs in [graphs.txt](datasets/graphs.txt).
- Utilities in [graph_util](graph_util/).