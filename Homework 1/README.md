# Homework 1: Statistical Learning Foundations and Generative Classifiers

## What it covers

- **Decision theory and Gaussian boundary analysis**
  - Proved that a randomized decision rule based on posterior probabilities never outperforms the deterministic Bayes decision rule under zero-one loss.
  - Derived optimal decision thresholds for 1D and 2D Gaussian distributions; sampled and plotted 100 points per class in 2D to visualize how decision boundaries shift with changes in class priors and covariance matrices.

- **Parameter estimation and linear frameworks**
  - Derived Maximum Likelihood Estimates for a Bernoulli-based Naive Bayes model (empirical frequencies).
  - Modeled discrete event counts with a Poisson distribution, computing rolling MLE parameters and subsequent event probabilities.
  - Proved analytically that maximizing the log-likelihood of linear regression under zero-mean Gaussian noise is equivalent to minimizing Mean Squared Error.

- **Text classification**
  - Built a binary Naive Bayes spam filter from scratch on a subset (10,000 instances) of the TREC Public Spam Corpus.
  - Applied m-estimate smoothing for out-of-vocabulary terms and log-probability transforms to avoid floating-point underflow; evaluated accuracy across different smoothing scales.

- **Custom feature engineering**
  - Designed a rule-based classifier using raw pixel color configurations to distinguish cloud vs. clear-sky images.
  - Evaluated the pipeline with a confusion matrix, precision, and recall, plus a manual audit of misclassified edge cases.

## Contents

- `HW#1.pdf` — complete analytical report: mathematical proofs, derivations, and decision-boundary plots.
- `Images.zip` — dataset assets and generated plots for the decision-space and classifier performance analysis.
