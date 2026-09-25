## 1. Why Feature Selection?

Real datasets can contain many features, but **more features do not automatically mean a better model**. As dimensionality increases:

- More data is required to adequately represent the feature space.

- Irrelevant features can add noise and increase overfitting.

- Redundant features can complicate interpretation and model fitting.

- Training becomes more computationally expensive.

**Feature selection** reduces the original feature set to a smaller set of useful predictors. The goal is generally to improve the **signal-to-noise ratio** while making models simpler, faster, and easier to interpret.

> **Feature selection vs. dimensionality reduction:** Feature selection keeps a subset of the original features. Methods such as PCA instead construct new features from the originals.

---

## 2. Unsupervised vs. Supervised Feature Selection

### Unsupervised Feature Selection

Unsupervised methods examine only the structure of the predictors \(X\). They do **not** require knowledge of the target \(y\).

#### Low-variance filtering

A feature that barely changes across observations has low resolution for distinguishing observations and may contribute little useful information.

```python

from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.0)

X_selected = selector.fit_transform(X)

```

Zero-variance features are common candidates for removal, and a higher threshold can be used to remove near-zero-variance features.

#### Removing highly correlated features

Highly correlated predictors contain redundant information. Keeping both can unnecessarily increase dimensionality and can make some models, particularly linear models, harder to interpret because their individual effects become difficult to separate.

One of a pair of strongly correlated predictors can sometimes be removed with little loss of information. Domain knowledge can also guide which predictor to retain.

However, low variance or correlation alone does **not** guarantee that a feature is useless. A low-variance feature may still be valuable for a particular task, such as detecting a rare event.

### Supervised Feature Selection

Supervised methods evaluate the relationship between the predictors \(X\) and the target \(y\) to identify useful features.

Examples include:

- correlation with the target

- statistical tests

- mutual information

- wrapper methods

- embedded/model-based selection

> **Important:** Because supervised feature selection uses the target, it must be learned from the **training data only**. Otherwise, information from the validation/test data can leak into model development.

---

## 3. Filter, Wrapper, and Embedded Methods

### Filter Methods

Filter methods evaluate features using properties of the data rather than repeatedly fitting the final predictive model.

For example, a feature may be selected because it has a strong association with the target.

A major limitation is that evaluating predictors individually can miss **interactions**. A feature that appears uninformative by itself may become highly predictive when combined with another feature. Simple association measures may also miss nonlinear relationships.

### Mutual Information

**Mutual information (MI)** measures dependence between two random variables and can capture both linear and nonlinear relationships.

Conceptually,

\[

I(X;Y) = H(Y) - H(Y|X)

\]

so mutual information asks:

> **How much does knowing \(X\) reduce our uncertainty about \(Y\)?**

Higher mutual information indicates that the feature provides more information about the target.

Feature-selection strategies can select features with high mutual information with the target while also considering redundancy with features that have already been selected.

### Wrapper Methods

Wrapper methods directly evaluate subsets of features by repeatedly:

1. selecting a feature subset,

2. training a model,

3. evaluating performance,

4. modifying the subset.

This can produce a feature set tailored to a particular predictive task and model.

The drawback is computational cost. Evaluating every possible combination of features quickly becomes infeasible, so practical wrapper methods often use heuristic or greedy strategies such as progressively adding or removing features.

### Embedded Methods

Embedded methods perform feature selection **during model fitting**.

Examples include:

- LASSO

- decision trees

- tree-based models such as random forests

Rather than having a separate feature-selection step, the structure or optimization of the fitted model determines which features are emphasized or retained.

---

# Decision Trees

## 4. Why Trees?

Linear models require us to specify a model structure in advance. They can represent nonlinear effects through transformations, but interactions between predictors must generally be explicitly included.

As the number of predictors increases, the number of possible interactions grows rapidly.

Decision trees instead learn a set of **conditional rules directly from the data**.

They are:

- non-parametric

- nonlinear

- capable of discovering interactions without explicitly specifying interaction terms

- inherently feature-selective

Rather than fitting one global relationship across the entire dataset, a tree divides the predictor space into simpler regions and makes predictions within those regions.

---

## 5. Anatomy of a Decision Tree

A decision tree recursively segments the predictor space.

A tree consists of:

- **Decision/internal nodes:** apply a splitting rule such as $X_j < t$

- **Branches:** represent the outcomes of the rule

- **Leaves/terminal nodes:** contain the final prediction

For a new observation, we begin at the root and follow the appropriate branches until reaching a leaf.

### Classification trees

At a terminal node, predictions are based on the class distribution of the training observations in that leaf, commonly using the majority class.

### Regression trees

At a terminal node, predictions are based on the target values of the training observations in that leaf, commonly their average.

The resulting prediction function is **piecewise**: different regions of feature space receive different predictions.

---

## 6. Recursive Binary Splitting

It is generally computationally infeasible to examine every possible partition of feature space.

Decision trees therefore use a **top-down, greedy approach** called recursive binary splitting.

At each node:

1. Consider candidate features and split points.

2. Divide the observations into candidate child nodes.

3. Measure the quality of each possible split.

4. Choose the split producing the greatest improvement.

5. Repeat the process within the resulting nodes.

The method is **top-down** because it starts at the root and successively divides the data.

It is **greedy** because it chooses the best split at the current step rather than looking ahead to determine which split would eventually produce the globally best tree.

---

## 7. Purity, Impurity ($I$), and Gain

A useful split creates child nodes that are more homogeneous than their parent.

### Classification

If all observations in a node belong to the same class, the node is **pure** and has zero impurity.

If multiple classes are mixed together, the node is impure.

Common classification impurity measures include:

#### Gini impurity/index ($G$)

$$
\text{G} = \sum_{i=1}^{K} p_i(1-p_i) = 1 - \sum_{i=1}^{K} p_i^2
$$

#### Entropy ($H$)

$$
H = -\sum_{i=1}^{K} p_i \log(p_i)
$$



#### Classification error ($E$)

$$
E = 1-\max_{{i=1,2,3,...,K}}(p_i) = 1 - p_{max}
$$

where:
- $K$: the number of classes
- $p_i$: the proportion (probability) of samples belonging to class $i$

### Regression

For regression trees, impurity represents the spread of target values within a node.

Common measures include:

- Mean Squared Error (MSE)

- Mean Absolute Error (MAE)

### Gain

A candidate split is evaluated by comparing the impurity of the parent (before split) with the weighted impurity of its children (after split):

$$
\text{Gain}
=
I(P)
-
\left(
\frac{n_L}{n_P} I(L)
+
\frac{n_R}{n_P} I(R)
\right)
$$

where:

- $P$ is the parent node

- $L$ and $R$ are the child nodes

- $I$ is the chosen impurity measure

The child impurities are weighted by node size.

**The tree chooses the split with the greatest gain.**

---

## 8. Trees Naturally Learn Interactions

One major advantage of trees is that different features can become relevant along different branches.

For example:

```text

             X₁ < 5?

             /     \

           yes      no

           /         \

       X₂ < 3?      X₃ < 8?

```

The effect of `X₂` only matters for observations that first satisfy `X₁ < 5`.

This allows a tree to represent **interactions between predictors** without explicitly creating interaction terms such as `X₁ * X₂`.

### Local Models

Many datasets may contain mixtures of different underlying relationships. A tree can first separate observations into different groups and then use different predictors within each group.

In this sense, trees can construct **local models** rather than forcing one global relationship to describe every observation.

---

# Controlling Tree Complexity

## 9. Why Trees Overfit

Remember when we learned [KNN](../week3/lab03-topic3.md): a small $K$ can lead to overfitting because the model becomes too sensitive to individual training observations and noise.

A similar idea applies to decision trees. We can think of **tree depth as playing a role similar to $K$ in KNN**: it controls the flexibility and complexity of the model.

- **KNN:** smaller $K$ → more flexible → higher risk of overfitting
- **Decision Tree:** greater depth → more flexible → higher risk of overfitting

If a tree is allowed to continue splitting and grow deeper, it can create increasingly small regions until it closely memorizes the training observations.

A very large tree can therefore have:

- extremely low training error
- high model complexity
- high variance
- poor generalization to new observations

A smaller tree may introduce some additional bias while reducing variance and improving interpretability.

This illustrates the **bias-variance tradeoff**: increasing model complexity can reduce bias but increase variance and the risk of overfitting.

### Early Stopping

**Early stopping** prevents the tree from growing beyond specified constraints.

Examples include:

- setting a maximum tree depth → `max_depth`
- requiring a minimum number of observations per node before splitting → `min_samples_split`
- requiring a minimum number of observations in each leaf after splitting → `min_samples_leaf`
- requiring a minimum impurity improvement before splitting → `min_impurity_decrease`

### Pruning

**Pruning** takes the opposite approach:

1. grow a larger tree,

2. remove unnecessary branches afterward.

Possible strategies include reduced-error pruning and error-complexity pruning.

The goal is to simplify the tree enough to improve generalization without discarding useful predictive structure.

---

## 10. Advantages and Limitations of Decision Trees

### Advantages

Decision trees are:

- nonlinear

- relatively easy to interpret, especially when small

- computationally efficient

- capable of discovering interactions

- capable of performing inherent feature selection

- able to model different relationships in different regions of feature space

Because individual features are considered at split points, trees also generally do not require features to be normalized or standardized simply because their numerical ranges differ.

### Limitations

Individual decision trees can be:

- unstable

- prone to overfitting

- difficult to optimize with respect to depth, pruning, and stopping criteria

- less accurate than stronger supervised-learning methods on complex prediction problems

Methods such as bagging, random forests, and boosting combine multiple trees to improve predictive performance, generally at the cost of some interpretability.

---

# Feature Importance and Model Interpretation

## 11. Why Interpret a Model?

After fitting a model, we may want to understand:

- how the model is making predictions

- whether it is making reasonable choices

- which predictors it relies on

- whether the data contain unexpected patterns or biases

**Feature importance** is an umbrella term for methods that quantify how relevant a feature is to making predictions.

Feature-importance methods can broadly be divided into:

- **model-based methods**

- **model-agnostic methods**

---

## 12. Feature Selection vs. Feature Importance

These ideas are closely related but answer different questions.

### Feature Selection

> **Which features should the model use?**

Feature selection changes or reduces the predictors available to the model.

### Feature Importance

> **How relevant is each feature to the fitted model's predictions?**

Feature importance helps us interpret the behavior of a model after or during fitting.

---

## 13. Model-Based Feature Importance

Some models provide importance information directly from their fitted structure.

### Regression Coefficients

For multiple linear regression,

$$
\hat y = \beta_0 + \beta_1X_1+\cdots+\beta_pX_p
$$

the coefficient $β_j$ describes how much the predicted dependent variable is expected to change when $X_j$ increases by one unit, **holding the other predictors constant**.

Coefficient magnitude must be interpreted in the context of the units in which each predictor is measured.

We can also evaluate whether an estimated coefficient is significantly different from zero. A common statistic is

$$
t = \frac{\hat{\beta}_j}{SE(\hat{\beta}_j)}
$$

which compares the estimated coefficient with its standard error.

### Decision Tree Feature Importance

Tree structure can also be used to quantify feature importance.

Possible signals include:

- how high in the tree a feature is used

- how frequently it is used

- how much impurity reduction its splits produce

A common variable-importance measure sums the reduction in impurity associated with splits using a particular feature.

For regression trees, this can involve reductions in MSE.

For classification trees, this can involve reductions in Gini impurity.

A feature that repeatedly creates large reductions in impurity will receive greater importance.

---

## 14. Model-Agnostic Feature Importance

Some models do not have easily interpretable coefficients or tree structures.

A model-agnostic alternative is to ask:

> **What happens to predictive performance when the information from this feature is disrupted?**

### Perturbation / Permutation Importance

A perturbation-based approach:

1. Measure the model's baseline performance.

2. Perturb one feature, for example by shuffling its values or adding noise.

3. Measure performance again.

4. Compare the perturbed performance with baseline performance.

Conceptually,

$$

\text{Importance}(X_j)

\approx

\text{Performance}_{baseline}

-

\text{Performance}_{perturbed}

$$

If performance changes very little, the feature may not be important to the model.

If performance decreases substantially, the model was relying strongly on that feature.

#### Advantages

- Applicable to many different model types.

- Does not require interpretable model parameters.

#### Limitations

- Can be computationally expensive.

- May miss complex relationships.

- Can underestimate the importance of correlated predictors.

If two features contain similar information, disrupting one may have little effect because the other feature can still provide similar information.

---

## 15. Feature Ablation

**Feature ablation** measures importance by removing a feature and evaluating model performance without it.

```text
Full feature set

       ↓

remove Xj

       ↓

fit/evaluate model

       ↓

compare performance
```

If performance remains similar, the feature may not contribute much unique predictive information.

If performance drops substantially, the removed feature was important for accurate predictions.

Ablation can be used with models that do or do not have built-in feature-importance measures.

---

# Putting the Concepts Together

Feature selection, decision trees, and feature importance address different parts of the same modeling problem:

```text
Many possible predictors

        ↓

FEATURE SELECTION

Which information should we keep?

        ↓

MODEL FITTING

How do we map predictors → outcome?

        ↓

FEATURE IMPORTANCE

What information does the fitted model rely on?
```

Decision trees connect these ideas particularly well because they:

1. perform **embedded feature selection** while fitting,

2. model nonlinear relationships and interactions through recursive splitting,

3. provide feature-importance information through their split structure and impurity reduction.

---

# Key Takeaways

- More features are not always better; irrelevant and redundant predictors can make modeling more difficult.

- **Unsupervised feature selection** examines the predictors without using the target.

- **Supervised feature selection** uses the relationship between predictors and the target and must be performed using training data only.

- **Filter methods** evaluate features directly, **wrapper methods** evaluate subsets through model performance, and **embedded methods** select features during fitting.

- Mutual information measures how much knowing one variable reduces uncertainty about another and can capture nonlinear dependence.

- Decision trees recursively partition feature space using conditional rules.

- Trees use **impurity reduction / gain** to determine useful splits.

- Recursive binary splitting is a **greedy** algorithm.

- Trees naturally represent nonlinear relationships and interactions without explicitly specifying interaction terms.

- Large trees can overfit; **early stopping and pruning** control model complexity.

- **Feature selection** asks which predictors to use, while **feature importance** asks how relevant predictors are to a fitted model's predictions.

- Regression coefficients and tree impurity reductions provide model-based measures of importance.

- Perturbation/permutation and ablation provide model-agnostic approaches to feature importance.

- Correlated features can complicate both model interpretation and feature-importance estimates.

- Model interpretation is only meaningful when the underlying model itself is performing adequately.
