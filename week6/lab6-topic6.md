# Sampling, Ensembles, Random Forests, and Uncertainty

## Big Picture

This unit connects four closely related ideas:

1. **Which observations should a model learn from?** → sampling and class imbalance  
2. **How can multiple models be combined?** → bagging and boosting  
3. **How can unstable decision trees become a stronger model?** → random forests  
4. **How much should we trust an individual prediction?** → predictive uncertainty and calibration  

A useful theme is that good machine learning is not only about fitting a model. The **composition of the training data**, the **stability of the fitted model**, and the **reliability of its predictions** all matter.

---

# 1. Data Sampling

## 1.1 Why sample data?

More data is usually useful, but not every observation is equally informative. Sampling can be used to:

- address **class imbalance**
- reduce computational cost
- reduce biases caused by overrepresented populations
- focus training on difficult or informative observations
- remove observations that appear noisy or redundant

Sampling changes the **training distribution**, so it should be treated as part of the modeling pipeline rather than as a purely cosmetic preprocessing step.

---

## 1.2 Class imbalance

A classification dataset is **imbalanced** when one class occurs much more frequently than another.

For example:

- 99,000 negative cases
- 1,000 positive cases

A classifier that predicts the majority class for every observation would achieve

$\text{Accuracy} = \frac{99{,}000}{100{,}000}=99\%$

despite having no ability to detect the positive class.

### Consequence

**High accuracy does not necessarily imply a useful classifier.**

For imbalanced problems, useful evaluation measures often include:

- **Recall / sensitivity** – how many actual positives were detected?
- **Precision** – how many predicted positives were actually positive?
- **F1 score** – harmonic mean of precision and recall
- **Balanced accuracy** – gives equal importance to performance on each class
- **ROC AUC** – evaluates ranking across classification thresholds

The appropriate metric depends on the consequences of the different types of errors.

---

# 2. Strategies for Imbalanced Data

There are two broad approaches:

### Undersampling
Remove observations, usually from the majority class.

### Oversampling
Increase the representation of the minority class.

Different methods make different assumptions about which observations are useful.

---

## 2.1 Random undersampling

Randomly remove majority-class observations.

### Advantages

- simple
- fast
- reduces training cost
- directly improves class balance

### Disadvantages

- throws away real observations
- may remove important boundary cases
- different random samples can produce different fitted models

The last point introduces an additional source of variability into the pipeline.

---

## 2.2 Tomek links

A **Tomek link** consists of two observations from opposite classes that are each other's nearest neighbors.

Conceptually, these points often lie near a class boundary.

Removing selected observations from Tomek links can:

- make the boundary cleaner
- remove ambiguous or noisy observations
- modestly reduce imbalance

However, difficult boundary observations can also contain valuable information. Removing them may make the training problem easier without necessarily making the resulting model better on real-world data.

---

## 2.3 Edited Nearest Neighbors (ENN)

ENN examines each observation's local neighborhood.

An observation can be removed when its label disagrees with the labels of its nearest neighbors.

The intuition is:

> If a point strongly disagrees with its local neighborhood, it may be noise or an unusual observation.

ENN can smooth a decision boundary, but it can also remove genuinely informative difficult cases.

---

## 2.4 Condensed Nearest Neighbors (CNN)

CNN takes almost the opposite perspective: rather than deleting difficult points, it tries to **retain the observations needed to preserve the decision boundary**.

A simplified procedure:

1. Retain all minority observations.
2. Retain one or a few majority observations.
3. Fit a 1-nearest-neighbor classifier using the retained set.
4. Test the remaining majority observations.
5. If an observation is misclassified, add it to the retained set.
6. Repeat until the retained set can correctly classify the remaining observations.

The result is a smaller dataset containing:

- all minority observations
- majority observations that are important for defining the boundary

### Key intuition

Easy, redundant observations contribute relatively little new information. Boundary-defining observations contribute much more.

---

## 2.5 Active subsampling

**Active learning** attempts to identify observations that would be especially informative to a model.

The same principle can be used for subsampling: retain observations the model finds informative rather than choosing them randomly.

This connects sampling with uncertainty: observations about which the current model is uncertain are often candidates for additional attention.

---

## 2.6 Naive oversampling

A simple alternative is to duplicate minority-class observations.

This changes how strongly the minority class contributes during training, but **does not create new information**.

Conceptually, repeated observations behave similarly to assigning larger weights to those observations.

---

## 2.7 SMOTE

**SMOTE — Synthetic Minority Over-sampling Technique** creates synthetic minority examples by interpolating between existing minority observations.

Instead of copying an observation exactly, SMOTE generates new points between minority-class neighbors.

### Potential advantage

The model sees a more diverse minority class than it would under simple duplication.

### Potential problems

Synthetic observations are artificial. SMOTE can:

- reinforce biases already present in the data
- propagate errors or noise
- create unrealistic samples if interpolation is not meaningful in the feature space

Therefore, synthetic sampling should not automatically be assumed to improve generalization.

---

# 3. Bias, Variance, and Noise

Prediction error can be understood through three components:

$\text{Expected Error}
=
\text{Bias}^2
+
\text{Variance}
+
\text{Noise}$

### Bias

Systematic error caused by the model's assumptions.

A high-bias model tends to make similar mistakes across different training sets.

Often associated with **underfitting**.

### Variance

How much the fitted model changes when the training data changes.

A high-variance model may fit one dataset extremely well but behave very differently if trained on a slightly different sample.

Often associated with **overfitting**.

### Noise

Inherent variability in the data-generating process.

Noise is **irreducible error**: no model can eliminate uncertainty that is genuinely intrinsic to the process.

Bias and variance contribute to **reducible error**, meaning model design or additional information may improve them.

---

# 4. The Bias–Variance Tradeoff

Model complexity often trades bias for variance.

### Simple model

- higher bias
- lower variance
- more stable across datasets
- greater risk of underfitting

### Highly flexible model

- lower bias
- higher variance
- more sensitive to the training sample
- greater risk of overfitting

Decision trees illustrate this clearly.

### Fully grown tree

- highly flexible
- low training error
- low bias
- high variance

### Pruned / constrained tree

- less flexible
- higher bias
- lower variance

The goal is not simply to minimize bias or variance individually. It is to find a model whose **combined generalization error** is small.

---

# 5. Ensemble Learning

An **ensemble** combines predictions from multiple models.

The central idea is:

> Multiple imperfect models can sometimes produce a better predictor when their information is combined.

Two major ensemble strategies are **bagging** and **boosting**.

---

# 6. Bagging

**Bagging = Bootstrap Aggregating**

Bagging trains multiple models on different bootstrap samples of the training data and combines their predictions.

For each model:

1. Sample observations **with replacement** from the original training set.
2. Fit a base learner.
3. Repeat many times.
4. Aggregate predictions.

For regression, predictions are commonly averaged.

For classification, predictions may be combined by voting or averaged class probabilities.

---

## 6.1 Why bagging works

Bagging is particularly useful for **high-variance, low-bias models**, such as deep decision trees.

If several models make somewhat different errors, averaging them reduces the influence of any one model's instability.

Thus, bagging primarily aims to:

$\boxed{\text{reduce variance}}$

while preserving relatively low bias.

---

## 6.2 Why diversity matters

Suppose every tree in an ensemble were identical.

Averaging identical predictions would provide no benefit.

Bagging works best when the individual models are:

- reasonably accurate
- **not perfectly correlated**

This principle becomes especially important in random forests.

---

# 7. Bootstrap Samples and Out-of-Bag Observations

Suppose the original training set contains $n$ observations.

A bootstrap dataset is created by drawing $n$ times **with replacement**.

For a particular observation, the probability of not being selected on one draw is

$1-\frac{1}{n}.$

The probability of never being selected in $n$ draws is

$\left(1-\frac{1}{n}\right)^n.$

As $n\rightarrow\infty$,

$\left(1-\frac{1}{n}\right)^n \rightarrow e^{-1}\approx0.368.$

Therefore, approximately **36.8%** of observations are left out of a particular bootstrap sample.

Equivalently, approximately

$1-e^{-1}\approx0.632$

or **63.2% of the unique original observations** appear at least once in a typical bootstrap sample.

This does **not** mean the bootstrap dataset contains only $0.632n$ rows. It still contains $n$ draws; some original observations simply appear multiple times.

---

# 8. Out-of-Bag (OOB) Evaluation

The observations not selected for a particular tree are its **out-of-bag observations**.

For observation $i$:

1. identify every tree for which $i$ was OOB
2. predict $i$ using those trees
3. aggregate those predictions
4. compare the OOB prediction with the true outcome

Repeating this for all observations provides an internal estimate of predictive error.

### Why OOB evaluation is useful

Each observation is evaluated only by trees that **did not train on it**.

This creates a built-in validation mechanism without requiring a separate validation split solely for this purpose.

OOB performance should still be interpreted as an estimate of generalization, not as a replacement for every form of external evaluation.

---

# 9. Random Forests

A **random forest** is an ensemble of decision trees that combines two forms of randomness:

1. **random observations** through bootstrap sampling
2. **random features** considered when constructing splits

The final prediction aggregates predictions across the trees.

---

## 9.1 Why not simply bag ordinary trees?

Decision trees are deterministic given the same data and settings, and strong predictors can repeatedly dominate early splits.

Even with different bootstrap samples, trees may therefore remain strongly correlated.

Random forests add **feature subsampling** to make the trees more diverse.

At each split, only a random subset of features is considered.

This prevents the same dominant features from controlling every tree.

### Desired result

Individual trees remain strong enough to be useful, but different enough that averaging meaningfully reduces variance.

---

# 10. Important Random Forest Parameters

## `n_estimators`

Number of trees in the forest.

More trees generally:

- make ensemble estimates more stable
- increase computation
- eventually produce diminishing improvements

The forest benefits from averaging many trees, but the exact useful number is dataset dependent.

---

## `max_features` / `mtry`

Number of candidate features considered at each split.

This parameter controls an important tradeoff.

### Too few features

Trees may become weak because useful predictors are frequently unavailable.

$\rightarrow \text{higher bias}$

### Too many features

The same strong predictors repeatedly dominate.

Trees become more similar.

$\rightarrow \text{higher correlation among trees}$

and the benefit of ensembling decreases.

Thus, feature subsampling balances **individual tree strength** against **tree diversity**.

---

## Tree complexity parameters

Examples include:

- `max_depth`
- `min_samples_split`
- `min_samples_leaf`

For a random forest, individual trees are often allowed to be relatively complex because the ensemble is specifically intended to reduce their variance.

---

# 11. Random Forests and Feature Interactions

An **interaction** occurs when the effect of one feature depends on another feature.

For example, suppose a biomarker has little relationship with disease risk among younger patients but a strong relationship among older patients.

A tree can represent this naturally:

1. split on age
2. within each age group, split on biomarker level

The effect of the biomarker therefore changes depending on age.

Tree-based models can capture such interactions **without manually creating an explicit interaction term** such as

$\text{age}\times\text{biomarker}.$

Random forests retain this ability while stabilizing predictions through aggregation.

---

# 12. Random Forest Feature Importance

Feature importance attempts to quantify how useful each predictor was across the forest.

A common tree-based importance measure examines the improvement in the splitting criterion produced by a feature.

Across a forest, these contributions can be aggregated over many trees.

### Interpretation

A large importance indicates that a feature frequently contributed useful splits.

It does **not** automatically imply:

- causality
- clinical importance
- that the feature acts independently
- that correlated features would receive importance in an intuitive way

Feature importance is a property of the **fitted model**, not a universal property of the variable.

---

# 13. Random Forest Proximity

Random forests can also define a model-based measure of similarity between observations.

Two observations are considered similar when they repeatedly land in the same terminal leaf.

$\text{Proximity}(i,j)
=
\frac{\text{number of trees where }i\text{ and }j\text{ share a leaf}}
{\text{total number of trees}}$

Interpretation:

- proximity $=0$: never share a leaf
- proximity $=1$: always share a leaf

This is not ordinary Euclidean distance. It is a **learned similarity induced by the forest**.

Two observations can therefore be considered similar because they repeatedly follow the same predictive rules.

---

# 14. Boosting

Boosting also combines multiple models, but its logic differs fundamentally from bagging.

Models are trained **sequentially**.

A simplified idea is:

1. train a weak learner
2. identify errors made by the current model
3. focus the next learner on correcting those errors
4. repeat
5. combine the learners into an ensemble

Boosting therefore primarily attempts to:

$\boxed{\text{reduce bias}}$

by progressively improving a collection of relatively simple learners.

Examples include:

- AdaBoost
- Gradient Boosting
- XGBoost

---

# 15. Bagging vs. Boosting

| Property | Bagging | Boosting |
|---|---|---|
| Training | Parallel / independent learners | Sequential learners |
| Typical base learner | High variance, low bias | Weak / high-bias learner |
| Main goal | Reduce variance | Reduce bias |
| Data emphasis | Bootstrap samples | Increasing attention to errors |
| Major example | Random Forest | AdaBoost / Gradient Boosting / XGBoost |
| Noise sensitivity | Often relatively robust | Can focus heavily on difficult/noisy observations |

A useful memory aid:

> **Bagging stabilizes strong but unstable learners. Boosting combines weak learners that progressively correct errors.**

---

# 16. Gradient Boosting and XGBoost

Gradient boosting builds models sequentially so that each new learner improves the current ensemble.

Instead of merely asking whether an observation was classified incorrectly, gradient boosting uses the **gradient of the loss function** to determine the direction in which predictions should be corrected.

XGBoost is an optimized implementation of gradient-boosted decision trees and includes mechanisms such as:

- sequential tree construction
- gradient-based optimization
- regularization
- learning-rate control
- tree-complexity control

### Important parameters

#### `n_estimators`
Number of boosting rounds / trees.

#### `learning_rate`
Controls how strongly each new tree changes the ensemble.

Smaller learning rates usually require more trees.

#### `max_depth`
Controls the complexity of each individual tree.

These parameters interact: increasing the number or complexity of trees can improve fit but can also increase overfitting.

---

# 17. Predictive Uncertainty

A point prediction alone does not tell us how much confidence to place in that prediction.

For example, two patients could both receive a predicted disease probability of $0.60$, while the model has much stronger evidence for one prediction than the other.

Quantifying uncertainty can help:

- identify unreliable predictions
- decide when a model should abstain
- guide additional data collection
- diagnose weaknesses in the model or training data

---

# 18. Aleatoric vs. Epistemic Uncertainty

## Aleatoric uncertainty

**Inherent uncertainty in the process itself.**

Examples:

- measurement noise
- biological variability
- genuinely stochastic outcomes

This uncertainty is generally considered **irreducible**.

---

## Epistemic uncertainty

Uncertainty caused by **limited knowledge**.

Examples:

- insufficient training data
- few observations in a region of feature space
- uncertain parameter estimates
- an inappropriate model
- instability in the fitting procedure

Epistemic uncertainty is potentially **reducible** through better data, models, or training procedures.

### Memory aid

> **Aleatoric = uncertainty in the world.**  
> **Epistemic = uncertainty in what the model knows.**

---

# 19. More Specific Sources of Uncertainty

The lectures distinguish several practical sources.

### Experimental uncertainty
Noise or error in measurements and observations.

### Interpolation uncertainty
Insufficient data in a particular region of feature space.

### Systemic / model uncertainty
The chosen model cannot adequately represent the underlying relationship.

### Algorithmic uncertainty
The fitting procedure itself can produce different models across runs.

### Parameter uncertainty
Parameters are estimated from finite data, so their true values are not known exactly.

These categories can overlap, which makes fully disentangling uncertainty difficult.

In practice, **total predictive uncertainty** is often useful even when its individual sources cannot be perfectly separated.

---

# 20. Quantifying Predictive Uncertainty

There is no single universal uncertainty measure.

Several approaches are possible.

## 20.1 Distance from a decision boundary

Predictions close to a classification boundary may be less stable because a small change in the model could change the predicted class.

---

## 20.2 Ensemble disagreement

If many independently perturbed models agree, the prediction appears stable.

If they strongly disagree, the prediction is more sensitive to the modeling choices or sampled data.

Random forests provide a natural example: predictions from individual trees can be compared.

However:

> **Ensemble disagreement is a signal of uncertainty, not automatically a calibrated probability of being wrong.**

---

## 20.3 Model-derived probabilities

Some models directly produce probabilistic quantities that can be interpreted as predictive uncertainty, although their calibration must still be checked.

---

## 20.4 Bootstrapping

Repeatedly:

1. resample the training data
2. refit the model
3. predict the same target observation

The distribution of resulting predictions reveals sensitivity to the training sample.

---

## 20.5 Parameter perturbation

Slightly modify parameters or modeling choices and observe how much the prediction changes.

Large changes suggest that the prediction is not robust to those choices.

---

## 20.6 Monte Carlo dropout

For neural networks, dropout can be kept active during repeated prediction passes.

Each pass produces a slightly different network and therefore a slightly different prediction.

The resulting distribution can be used as an uncertainty estimate.

---

# 21. Conformal Prediction

Instead of forcing every model to output a single answer, uncertainty can be integrated into the prediction itself.

For classification, a model might effectively return:

- class A
- class B
- ambiguous / multiple plausible classes

For regression, it may return a **prediction interval** rather than only a point estimate.

The central idea is to communicate the range of predictions compatible with the model's uncertainty rather than hiding uncertainty behind a single number.

---

# 22. Calibration

A model can rank observations correctly while still producing misleading probabilities.

**Calibration** asks whether the numerical uncertainty estimates correspond to observed frequencies.

For classification:

> Among predictions assigned probability $p$, does the outcome occur approximately $p$ of the time?

For example, among many predictions near $0.8$, a well-calibrated model should have the positive outcome occur roughly 80% of the time.

---

## 22.1 Reliability diagrams

A calibration curve / reliability diagram compares:

- predicted probabilities
- observed event frequencies

Perfect calibration follows the diagonal relationship

$\text{predicted probability}
=
\text{observed frequency}.$

A model may be:

- **overconfident** – probabilities are more extreme than observed performance justifies
- **underconfident** – probabilities are less extreme than observed performance would justify

Calibration is distinct from discrimination: a model can separate classes well and still have poorly calibrated probabilities.

---

# 23. Recalibration

A poorly calibrated model does not necessarily need to be retrained from scratch.

Instead, a mapping can be learned from the model's original scores to better-calibrated probabilities.

This mapping should be learned using **held-out data**, not the same observations used to train the original predictive model.

Common approaches include:

### Histogram binning
Group predictions into bins and replace scores with the observed frequency in each bin.

### Platt scaling
Fit a logistic regression that maps model scores to probabilities.

### Isotonic regression
Learn a flexible monotonic mapping between scores and calibrated probabilities.

Recalibration changes the interpretation of the output without necessarily changing the underlying ranking learned by the original model.

---

# 24. Connecting the Unit

The major ideas fit together:

$\boxed{\text{Sampling}}
\rightarrow
\boxed{\text{Different training sets}}
\rightarrow
\boxed{\text{Different fitted models}}$

Decision trees respond strongly to these changes:

$\boxed{\text{Tree flexibility}}
\rightarrow
\boxed{\text{Low bias + high variance}}$

Bagging deliberately exploits sampling variation:

$\boxed{\text{Bootstrap samples}}
\rightarrow
\boxed{\text{Many diverse trees}}
\rightarrow
\boxed{\text{Average predictions}}
\rightarrow
\boxed{\text{Lower variance}}$

Random forests add feature randomness:

$\boxed{\text{Bagging + random feature subsets}}
\rightarrow
\boxed{\text{Less correlated trees}}
\rightarrow
\boxed{\text{More effective variance reduction}}$

Variation across models can then provide information about prediction stability:

$\boxed{\text{Model disagreement}}
\rightarrow
\boxed{\text{Signal of predictive uncertainty}}$

Finally:

$\boxed{\text{Uncertainty estimate}}
eq
\boxed{\text{Automatically trustworthy probability}}$

so we evaluate **calibration**.

---

# 25. High-Yield Distinctions

| Concept A | Concept B | Key Difference |
|---|---|---|
| Bias | Variance | systematic error vs. sensitivity to training data |
| Reducible error | Irreducible error | model/data limitations vs. inherent noise |
| Undersampling | Oversampling | remove observations vs. increase minority representation |
| Tomek / ENN | CNN | remove difficult/local-inconsistent points vs. retain boundary-defining points |
| Naive oversampling | SMOTE | duplicate observations vs. synthesize interpolated observations |
| Bagging | Boosting | stabilize models in parallel vs. correct errors sequentially |
| Decision tree | Random forest | one unstable learner vs. ensemble of randomized trees |
| Aleatoric | Epistemic uncertainty | inherent variability vs. lack of knowledge |
| Uncertainty | Calibration | how unsure the model is vs. whether numerical confidence matches observed frequencies |
| Accuracy | Balanced accuracy | overall fraction correct vs. equal emphasis on classes |

---

# 26. Core Takeaways

1. **The training data distribution matters.** Sampling can substantially alter what a model learns.
2. **Accuracy can be misleading under class imbalance.**
3. **Sampling methods encode assumptions about which observations are informative.**
4. **Flexible decision trees tend to have low bias but high variance.**
5. **Bagging primarily reduces variance by averaging diverse models.**
6. **Boosting primarily reduces bias by sequentially correcting errors.**
7. **Random forests combine bootstrap sampling with random feature selection to decorrelate trees.**
8. **About 63.2% of unique observations appear in a typical size-$n$ bootstrap sample; about 36.8% are OOB for that tree.**
9. **OOB observations provide a built-in way to estimate predictive performance.**
10. **Random forests can capture nonlinearities and interactions without explicitly specifying interaction terms.**
11. **Feature importance and forest proximity are properties of the fitted forest and should be interpreted accordingly.**
12. **Predictive uncertainty is not the same as predictive error.**
13. **Aleatoric uncertainty is inherent; epistemic uncertainty is potentially reducible.**
14. **Ensemble disagreement can indicate instability, but it is not automatically calibrated.**
15. **Calibration asks whether confidence estimates correspond to observed frequencies.**
16. **A useful model should not only predict well—it should also communicate when its predictions are less reliable.**
