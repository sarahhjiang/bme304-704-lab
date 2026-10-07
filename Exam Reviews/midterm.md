# Midterm Review

Now you are a biomedical data scientist. You have just been assigned your very first task. Let's take a look at your data and what do you need to do!

## Data
### What kind of data do we have?
- Binary
- Continuous (Real-Valued)
- Categorical
- Ordinal

### What is the distribution of my data?
- Are there any unusual values or outliers?
- ...

### Does my data have labels?
- Is this a supervised or unsupervised learning problem?
- What is the label? Is the prediciton job
  - Regression
  - Classification
- Are the labels balanced?

### Do we need to transform any data? How to transform the data?
- Normalization
- Standardization
- Log transformations

### Do we have any missing data in our dataset?
- Are certain observations more likely to have missing data?
- What type of missingness might we have?
  - Missing Completely at Random (MCAR)
  - Missing at Random (MAR)
  - Missing Not at Random (MNAR)

### How should we handle missing data?
- Listwise deletion
- Single imputation
- Multiple imputation
- Model-based approaches

## Data Exploration

Now I know what kind of data I have. I would like to explore the relationships among my observations and see whether there are correlations, clusters, subgroups, or unusual patterns within the data.

When a dataset has many predictors, some predictors may contain overlapping information or add noise. Dimensionality reduction allows us to represent high-dimensional data using fewer variables while retaining important information, making the structure of the dataset easier to visualize and interpret.

### What is the curse of dimensionality?
- More predictors require more data
- Distances between observations become less informative
- The nearest and farthest observations may become increasingly similar
- Chance patterns become easier to find
- Model performance may decrease when too many irrelevant predictors are included

### Dimensionality Reduction

How can I represent my high-dimensional data in two or three dimensions?

- Multidimensional Scaling (MDS)
- Stochastic Neighbor Embedding (SNE)
- t-Distributed Stochastic Neighbor Embedding (t-SNE)
- Principal Component Analysis (PCA)
- Linear Discriminant Analysis (LDA)

### Which dimensionality reduction method should I use?
- Do I want to preserve global variance or local neighborhood structure?
- Do I want a linear or nonlinear projection?
- Is my goal visualization or class separation?
- Do I have class labels?
- Can the classes be separated by a linear boundary?
- Can the method identify which original features contribute most to the reduced dimensions?

### How similar are my observations?

After visualizing the data, I may also want to quantify how similar or different two observations are. The choice of distance or similarity measure depends on the type and structure of the data.

A distance metric should satisfy:
- The distance from an observation to itself is zero
- The distance from A to B is the same as the distance from B to A
- The distance obeys the triangle inequality
- Distance values are non-negative

Common distance / similarity measures include:

- Euclidean distance
- Manhattan distance
- Cosine similarity
- Hamming distance

### Which distance / similarity measure should I use?

- Are my features continuous and on comparable scales?
- Do I want less sensitivity to large individual differences or outliers?
- Is the pattern or direction more important than the overall magnitude?
- Am I comparing aligned sequences or categorical strings?

### Who Are the Nearest Neighbors?

Once a distance metric is defined, we can identify observations that are most similar to each other.

- Nearest Neighbor
- k-Nearest Neighbors (kNN)

kNN identifies the `k` observations that are closest to a given observation according to the selected distance metric.

## Data Splitting

Now that I understand my data and have explored the relationships among observations, I need to divide the dataset so that I can train, tune, and evaluate my model properly.

### What is each subset used for?
- Train set
- Test set
- Validation set

### Which splitting strategy should I use? How should I choose the splitting strategy?

- Random split
- Stratified split
- Group-based split
- Time-based split
- k-Fold Cross-Validation
- Leave-One-Out Cross-Validation (LOOCV)

### What should I avoid and how can I avoid it?
- Data leakage

### What if my data are imbalanced?
- Use stratified splitting to preserve class proportions across splits
- Consider undersampling the majority class
- Consider oversampling the minority class
- Be careful when using accuracy as the only evaluation metric

## Feature Selection

Now that my data have been split properly, I may want to reduce irrelevant or redundant features before building the model.

### Why do I need feature selection?
- Reduce the curse of dimensionality
- Remove irrelevant or redundant features
- Reduce overfitting
- Improve interpretability

### When should I perform feature selection?

- After separating the training data from the final test set
- Before fitting the final model, if using filter-based methods
- After fitting a model on the training data, if using model-based feature importance
- If using cross-validation, feature selection should be performed within each training fold

### Which feature selection method should I use?

- Remove low-variance features
- Remove highly correlated features
- Supervised feature selection
- Model-based feature selection

### If I use model-based feature importance

1. Fit the model using training data
2. Calculate feature importance
3. Select the most informative features
4. Retrain the model using the selected features
5. Evaluate on validation / test data

## Model Selection

Now that my data have been prepared and the relevant features have been selected, I need to choose an appropriate model for my prediction task.

### What kind of prediction problem do I have?
- Regression
- Classification

### Is the relationship approximately linear?
- Linear Regression
- Logistic Regression

### Do I need a more flexible nonlinear model?
- Nonlinear Regression
- k-Nearest Neighbors (kNN)
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest

### Do I need regularization?
- Ridge Regression
    - Shrinks coefficients toward zero to reduce model complexity and overfitting
- LASSO Regression
    - Can shrink some coefficients to zero, providing both regularization and feature selection

### What is overfitting/underfitting. How to avoid it?

- Overfitting
  - Model is too complex and fits noise in the training data
  - Performs well on training data but poorly on unseen data
  - High variance

- Underfitting
  - Model is too simple to capture the underlying pattern
  - Performs poorly on both training and test data
  - High bias

## Model Evaluation

Now that I have selected and trained my model, I need to evaluate how well it performs on unseen data.

### How should I evaluate a regression model?
- Residuals
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R²

### How should I evaluate a classification model?
- Confusion Matrix
  - True Positive (TP)
  - True Negative (TN)
  - False Positive (FP, Type I error)
  - False Negative (FN, Type II error)

### How should I evaluate a classification model?

- Accuracy
  - Overall proportion of correct predictions
  - `Accuracy = (TP + TN) / (TP + TN + FP + FN)`

- Precision
  - Among predicted positives, how many are actually positive?
  - Look at predicted positives
  - `Precision = TP / (TP + FP)`

- Recall / Sensitivity
  - Among actual positives, how many are correctly identified?
  - Look at actual positives
  - `Recall = Sensitivity = TP / (TP + FN)`

- Specificity
  - Among actual negatives, how many are correctly identified?
  - Look at actual negatives
  - `Specificity = TN / (TN + FP)`

- Balanced Accuracy
  - Average of sensitivity and specificity
  - `Balanced Accuracy = (Sensitivity + Specificity) / 2`

- Matthews Correlation Coefficient (MCC)
  - Considers TP, TN, FP, and FN together
  - `MCC = (TP × TN - FP × FN) / √[(TP + FP)(TP + FN)(TN + FP)(TN + FN)]`
  - Range: `-1` to `1`
    - `1` = perfect prediction
    - `0` = no correlation / approximately random
    - `-1` = completely opposite prediction
- AUROC
  - Area Under the Receiver Operating Characteristic (ROC) Curve
  - ROC curve plots:
    - `True Positive Rate (TPR) = TP / (TP + FN)`
    - `False Positive Rate (FPR) = FP / (FP + TN) = 1 - Specificity`
  - AUROC summarizes model discrimination across all classification thresholds
  - Range:
    - `1.0` = perfect discrimination
    - `0.5` = random guessing
    - `< 0.5` = worse than random / inverse prediction pattern

### Which classification metric should I use?
- What is your end goal?
- Is my dataset imbalanced?

### What if my classification threshold changes?
- Different thresholds change sensitivity and specificity
- Evaluate model performance across different thresholds
- ROC curve

## Model Interpretation / Feature Importance

Now that I know how well my model performs, I want to understand how the model makes its predictions and which features are most important.

### Which features are important to my model?
- Regression / logistic regression coefficients
- LASSO coefficients
- Decision tree feature importance
- Random Forest feature importance
- Linear SVM coefficients
- Permutation / perturbation importance
- PDP / ICE
- SHAP

### What do I want to interpret?
- Global feature importance
  - Which features are important across the entire dataset?
- Local feature importance
  - Which features contributed to a specific prediction?

### What should I be careful about?
- Feature importance does not necessarily imply causality
- Correlated features can share or distort importance
- Coefficient magnitude can depend on feature scaling
- Complex models may require model-agnostic interpretation methods

## Prediction Uncertainty

### How confident is my model in its prediction?
- Aleatoric uncertainty
  - Inherent / irreducible uncertainty
- Epistemic uncertainty
  - Uncertainty from lack of knowledge
  - Potentially reducible

### How can I estimate uncertainty?
- Distance to decision boundary
- Variation among ensemble models
- Prediction probabilities
- Bootstrapping

## Model Comparison

Now that I have evaluated my model, I need to determine whether differences in model performance are statistically meaningful.

### Is one model significantly better than another?
- Evaluate models multiple times
  - Cross-validation folds
  - Different train/test splits
  - Repeated training
- Compare model performance
  - ROC curves

### Which statistical test should I use?
- Student's t-test
- Paired t-test
- Welch's t-test
- Wilcoxon signed-rank test
- Mann-Whitney U test

### What should I consider when choosing a statistical test?
- Are the measurements paired?
- Are the data normally distributed?
- Do the groups have equal variance?