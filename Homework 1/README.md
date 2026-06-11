Homework 1: Statistical Learning Foundations and Generative Classifiers

Technical Summary of Implementations: 

  1. Decision Theory and Gaussian Boundary Analysis

    Bayes Risk Optimization: Formulated a mathematical proof demonstrating that a randomized decision rule based on posterior probabilities underperforms or equals the deterministic Bayes decision rule under a zero-one loss function.

    Parametric Decision Boundaries: Derived optimal decision thresholds for 1D and 2D Gaussian distributions. Sampled and plotted 100 data points per class in a 2D feature space to analyze decision boundary shifts resulting from variations in prior probabilities and covariance matrices.

  2. Parameter Estimation and Linear Frameworks

    Maximum Likelihood Estimation: Derived Bernoulli parameter estimates as empirical frequencies for a Naive Bayes framework.

    Poisson Distribution Modeling: Modeled discrete event data using a Poisson distribution to calculate rolling maximum likelihood parameters and compute subsequent event probabilities.

    Least Squares Derivation: Proved analytically that maximizing the log-likelihood of a linear regression model under zero-mean Gaussian noise is mathematically equivalent to minimizing the Mean Squared Error loss function.

  3. Text Classification

    Spam Filtering Engine: Developed a binary Naive Bayes classifier from scratch using a subset of the TREC Public Spam Corpus (10,000 instances).

    Numerical Optimization: Implemented m-estimate smoothing to manage out-of-vocabulary terms and applied log-probability transformations to prevent floating-point underflow. Evaluated model accuracy across varying smoothing parameter scales.

  4. Custom Feature Engineering

    Image Classification Heuristic: Designed and evaluated a custom rule based on raw pixel color configurations to classify cloud and clear sky images.

    Performance Metrics: Evaluated the custom pipeline using a confusion matrix, precision, and recall, followed by a diagnostic audit of edge-case misclassifications.Technical Summary of Implementations and Derivations

Directory Contents:

  HW1.pdf: Complete analytical report containing step-by-step mathematical proofs, LaTeX equations, and generated decision boundary charts.
  Images.zip: Dataset assets and generated visual plots for the decision space and model performance analysis.
