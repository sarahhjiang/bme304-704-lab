# Week 3 Lab Recap: Similarity, kNN, t-SNE, and Evaluation

This week is most useful for the lab as a bridge from **defining similarity** to **using learned or measured representations responsibly**. The key questions are not just "which algorithm do I use?" but:

-   What does *nearby* mean in this representation?
-   Do local neighborhoods correspond to something biologically or clinically meaningful?
-   Does a visualization preserve the structure I think it does?
-   Is model performance real, or is it an artifact of how the data were split?

## 1. Similarity is a modeling choice

Many ML methods depend on the assumption that observations that are similar in the chosen feature space should behave similarly. The important part is that **similarity is not intrinsic to the data**: it depends on the representation and the distance/similarity function.

### Distance choices

-   **Euclidean distance** measures straight-line distance across features.
    -   Appropriate when features are continuous and on comparable scales.
    -   Sensitive to feature scale and large differences, so standardization matters.
-   **Manhattan distance** sums absolute feature-wise differences.
    -   Less dominated by a single large feature difference than Euclidean distance.
    -   Can be useful for sparse/count-like data.
-   **Cosine similarity/distance** emphasizes the *direction* of feature vectors rather than their magnitude.
    -   Useful when relative composition matters more than absolute size.
    -   Can fail badly when magnitude or absolute location carries the signal.

For embeddings, "distance" determines what we mean when we say two participants, windows, cells, or recordings have similar representations. Before interpreting nearest neighbors or clusters, ask whether the chosen metric reflects the type of similarity we actually care about.

Feature scaling is especially important for distances computed on hand-engineered or heterogeneous features. A high-variance or large-unit feature can otherwise dominate the neighborhood structure.

## 2. k-Nearest Neighbors as a representation sanity check

kNN predicts from the labels or values of nearby training observations:

-   classification → majority vote among the `k` neighbors
-   regression → average among the `k` neighbors

The main tuning parameter is `k`.

-   Small `k` → highly local/flexible; more sensitive to noise and outliers.
-   Large `k` → smoother/more stable; can wash out real local structure and drift toward the majority class.
-   `k = 1` can essentially memorize the training set, so training accuracy is not evidence of generalization.

### Why this matters beyond using kNN as a classifier

kNN can be used as a **probe of representation quality**. If embeddings are useful, observations that should be related may occupy similar neighborhoods.

Examples for lab work:

-   cohort-label purity among nearest neighbors
-   disease/phenotype enrichment in local neighborhoods
-   whether repeated observations from a participant remain locally related
-   whether device/source labels dominate neighborhoods when we would prefer physiological structure

A strong kNN result therefore does not automatically mean the representation is "good." It means the representation contains local structure predictive of the label being tested. Whether that structure is desirable depends on the question.

## 3. Dimensionality reduction

Dimensionality reduction maps high-dimensional observations into a small number of dimensions so that humans can inspect structure.

### SNE / t-SNE

SNE instead emphasizes preservation of **local neighborhoods**. t-SNE modifies SNE with a heavy-tailed Student-t distribution in the low-dimensional space, which helps reduce the crowding problem and visually separate local groups.

The important interpretation is:

> Points that are close in a t-SNE plot are intended to represent local similarity, but the 2-D map is not a faithful picture of every relationship in the original high-dimensional space.

So:

-   visible clusters can be useful hypotheses;
-   distances between far-away clusters should not be overinterpreted;
-   apparent cluster size/spacing is not necessarily meaningful;
-   conclusions should be checked against labels, marker features, neighbor statistics, or downstream tasks.


For learned wearable embeddings, a UMAP/t-SNE figure is useful for seeing whether cohorts or phenotypes appear structured, but it should be paired with quantitative measures such as neighbor purity, silhouette score, or downstream probes. The visualization alone is not evidence that the representation learned the desired biology.

## 4. Splitting strategy is part of the experiment

A train/test split is not merely bookkeeping. **The split defines the generalization problem.**

A random split asks roughly:

> Can the model predict another observation drawn from a similar mixture of data?

Other splits can ask much harder and more useful questions.

### Group-based splitting

If related observations occur multiple times, randomly splitting individual observations can put highly similar data in both train and test sets. This can inflate performance.

For biomedical data, groups might be:

-   participant/patient
-   site/provider
-   demographic or clinical subgroup
-   device/source
-   acquisition batch

For longitudinal health data, **participant-level splitting is critical** whenever multiple days/windows come from the same person. Otherwise the model can partially exploit participant-specific information rather than generalizing to unseen people.

### Time-based splitting

Training on earlier data and testing on later data approximates a prospective deployment question:

> Can a model trained on what was available then generalize to what appears later?

This is especially relevant when datasets, devices, clinical practices, or populations change over time.

### Stratified splitting

Stratification preserves class proportions across splits and is useful when labels are imbalanced. It solves a different problem from grouping: a split can need to be both **group-aware and stratified**.

## 5. Cross-validation estimates stability

A single held-out split can be unusually easy or difficult. In `K`-fold cross-validation:

1.  divide the data into `K` folds;
2.  train on `K-1`;
3.  evaluate on the remaining fold;
4.  rotate the held-out fold;
5.  combine performance across folds.

Typical `K` values are around 5--10, with the choice depending partly on dataset size and computational cost.

Cross-validation is useful both for:

-   estimating expected performance on unseen data;
-   selecting hyperparameters such as `k` in kNN.

When reporting CV results, variability across folds matters. The assignment's accuracy-vs-`k` plot with standard-error bars is useful because it shows whether an apparent "best" value is meaningfully better or just within split-to-split noise.

## 6. Avoid leakage

One of the most important lessons from the split lecture is that **anything that learns from the data must happen inside the validation procedure**.

If feature selection uses all labels first and cross-validation is applied only afterward, information from the held-out folds has already influenced the model. The resulting performance can be dramatically
optimistic.

The same principle applies to:

-   feature selection
-   normalization/statistics estimated from data
-   dimensionality reduction used as model input
-   imputation
-   hyperparameter selection
-   representation learning, when the evaluation is supposed to measure generalization to unseen participants/data

A useful rule:

> The test set should not influence any decision that affects the fitted model.

## 7. Practical takeaways for our lab

When evaluating biomedical or wearable representations:

1.  **Define the unit of independence first.** If multiple observations belong to one participant, split by participant unless the scientific question explicitly concerns within-participan prediction.
2.  **Ask what generalization you care about.** New participant? New cohort? New device? New time period? New site? The split should reflect that question.
3.  **Treat neighborhood metrics as representation diagnostics.** kNN purity or local label agreement tells us what information is organized locally in the embedding.
4.  **Do not rely on embedding plots alone.** Pair t-SNE/UMAP with quantitative neighborhood, separation, and downstream-task metrics.
5.  **Keep preprocessing leakage-free.** Fit normalization, feature selection, and other learned preprocessing only on the appropriate training data.
6.  **Report repeated/CV performance when feasible.** A mean plus variability is more informative than a single favorable split.
7.  **Interpret high performance carefully.** If cohort/device identity is easily predicted, that may indicate meaningful distribution shift rather than universally "better" representations.

The central idea of topic 3 is that **representation, similarity, and evaluation are inseparable**. A representation determines which observations become neighbors; the distance metric determines how those neighbors are defined; and the splitting strategy determines whether apparent predictive structure actually generalizes to the population or setting we care about.
