# Dimensionality Reduction, Feature Selection, and Image Segmentation

Three separate exercises covering unsupervised and feature-engineering techniques.

## What it covers

- **Q7 — PCA & Kernel PCA** (`HW4_Q7.ipynb`): Standard PCA on the Iris dataset, compared against Kernel PCA (e.g. on non-linearly separable data such as `make_circles`), to illustrate when a non-linear projection is needed to reveal cluster structure.
- **Q8 — Feature selection** (`HW4_Q8.ipynb`): Applied to the heart-disease dataset (`heart.csv`) using mutual information, Recursive Feature Elimination (RFE/RFECV), and stratified k-fold cross-validation to select and evaluate the most predictive features for a logistic regression model.
- **Q9 — Image segmentation** (`HW4_Q9.ipynb`): Unsupervised segmentation of images (`image1.jpg`, `image2.jpg`) using K-Means and Gaussian Mixture Models across multiple values of *K*, comparing how each algorithm partitions pixel color space.

## Contents

- `HW4_Q7.ipynb`, `HW4_Q8.ipynb`, `HW4_Q9.ipynb` — notebooks for each question
- `Homework_4.pdf`, `تمرین__4_.pdf` — assignment prompt (English and Persian)
- `heart.csv`, `image1.jpg`, `image2.jpg` — datasets/images used in Q8 and Q9
