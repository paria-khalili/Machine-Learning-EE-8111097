# Genetic Algorithms for Image Reconstruction

Uses a Genetic Algorithm (GA) to automatically reassemble a shuffled jigsaw puzzle image, built on top of the open-source [GAPS (Genetic Algorithm Puzzle Solver)](https://github.com/nemanja-m/gaps) library.

## What it covers

- **Fit score** — a similarity metric between two puzzle piece edges, computed from the negative sum of squared pixel differences (0 = perfect match, more negative = worse match)
- **Puzzle fitness** — aggregating edge fit scores to evaluate a full candidate arrangement of pieces
- **Selection** — survival-of-the-fittest selection of candidate arrangements between generations
- **Evolution engine** — the GA loop that mutates and recombines arrangements over generations to converge on the correct layout
- Running the GAPS solver end-to-end and visualizing the shuffled input alongside the reconstructed solution

## Contents

- `HW2-810801065.py` — solution notebook/script (exported as a `.py`), implementing the GA-based solver
- `HW2-810801065.pdf` — write-up of the solution
- `ML_Fall_2025_Homework_2 (2).pdf` — the assignment prompt
