# Lab 01–02 Recap — Lectures 1–6 (Lines & Curves)


## 1. Correlation and the simple linear model

**Pearson r** measures strength and direction of a *linear* association, $r \in [-1, 1]$:

$$r = \frac{\sum (x_i - \bar x)(y_i - \bar y)}{\sqrt{\sum (x_i - \bar x)^2 \sum (y_i - \bar y)^2}}$$

If $r$ is close to $\pm 1$, fitting $f(x) = mx + b$ is reasonable. For one predictor you can compute the parameters directly:

$$m = \frac{\sum (x_i - \bar x)(y_i - \bar y)}{\sum (x_i - \bar x)^2}, \qquad b = \bar y - m\bar x$$

**Interpretation** (drill this — it shows up in 2.4):
- $\beta_0$ = expected $y$ when $x = 0$ (baseline).
- $\beta_1$ = expected change in $y$ per one-unit change in $x$.
- Interpolating inside the observed range of $x$ is fine; extrapolating outside it is a modeling claim you have no data to support.

**Four assumptions**:
1. Linearity between $x$ and $y$
2. Independence of observations / residuals
3. Normality of residuals
4. Homoscedasticity — residual variance constant across $x$

*Remember the above by **LINE** (Linearity, Independence, Normality, and Equal Variance/Homoscedasticity)*

---

## 2. OLS: from calculus to matrices

The whole derivation is: **assume a model, write the error, minimize it.** Important note: with $p$ predictors, your model will have $p+1$ parameters. 

$$E(\beta_0, \beta_1) = \frac{1}{n}\sum_{i=1}^{n}(y_i - \beta_0 - \beta_1 x_i)^2$$

**In lab today**, we'll derive the values of $S_{XX}$ and $S_{XY}$. In your homework, you'll derive the values of the parameters $\beta_0, \beta_1$.

Remember: 
$S_{XX} = \sum_{i=1}^{n} (x_i - \bar{x})^2$

$S_{XY} = \sum_{i=1}^{n} (x_i - \bar{x})(y_i-\bar{y})$

Helpful equivalences:
- $\sum y_i = n\bar y$
- $\sum x_i y_i - n\bar x \bar y = \sum (x_i - \bar x)(y_i - \bar y)$
- $\sum x_i^2 - n\bar x^2 = \sum (x_i - \bar x)^2$

**Linear Regression Matrix Form:** Imagine you now have 25 parameters that you're interested in using as predictors. Writing out $y_i = \beta_0 + \beta_1 * x_{i,1} + ... + \beta_25 * x_{i,25}$ repeatedly becomes impractical. This is where matrix notation becomes helpful. We can express the same relationship via 
$$\hat Y = X\vec\beta$$
where $X$ is the **design matrix**: one row per observation, one column per parameter.

$$
X = \begin{pmatrix}
1 & x_{1,1} & x_{1,2} & \cdots & x_{1,25} \\
1 & x_{2,1} & x_{2,2} & \cdots & x_{2,25} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_{n,1} & x_{n,2} & \cdots & x_{n,25}
\end{pmatrix},
\qquad
\vec\beta = \begin{pmatrix}
\beta_0 \\ \beta_1 \\ \beta_2 \\ \vdots \\ \beta_{25}
\end{pmatrix}
$$

For $n$ observations and $p$ predictors, $X$ is $n \times (p+1)$ and $\vec\beta$ is $(p+1) \times 1$, so $X\vec\beta$ is $n \times 1$ — one fitted value per observation, exactly as desired. Adding a 26th predictor means adding a column to $X$ and an entry to $\vec\beta$; nothing else about the notation changes.

This lets us write the residuals compactly as a function of the parameter vector:

$$\vec e(\vec\beta) = Y - \hat Y = Y - X\vec\beta$$

Note that $\vec e$ is a *vector* of $n$ residuals, not the single scalar $E$ from the univariate case. To recover that scalar sum of squared residuals, take the dot product with itself:

$$E = \vec e \cdot \vec e = \vec e^{\,T}\vec e = (Y - X\vec\beta)^T(Y - X\vec\beta)$$

Minimizing $E$ with respect to $\vec\beta$ gives the **normal equations**, $X^TX\hat\beta = X^TY$, and therefore

$$\hat\beta = (X^TX)^{-1}X^TY$$

This is the same optimization you performed with calculus in Section 1.1 — propose parameters, evaluate the error, minimize, but it now solves for all parameters at once instead of two.

---

## 3. Transformations and scaling

**When to transform:**
- *Log rule* — variable spans more than one order of magnitude and is strictly positive → try a log.
- *Range rule* — variable spans much less than one order of magnitude → transforming probably won't help.
- Always re-plot after transforming.

**Normalization vs. standardization (4.1):**

$$X_{\text{norm}} = \frac{X - X_{\min}}{X_{\max} - X_{\min}}, \qquad X' = \frac{X - \mu}{\sigma}$$

** Remember, *normalizations* scales the data to values in [0,1] while *standardization* transforms the data such that it is centered around 0 with standard deviation 1 (i.e. N(0,1)).

Two reasons this matters:
1. Features with wildly different variances get weighted differently by the fitting procedure — LASSO will zero out a large-variance feature that carries real signal purely because of its scale.
2. Once features are standardized, coefficient magnitudes become comparable and can be read as feature importance.

---

## 4. Overfitting, Ridge, and LASSO

More flexible model terms (higher-degree polynomials, exponentials) fit training data better and generalize worse. Regularization adds a penalty on coefficient size to the loss:

$$\text{Loss}_{\text{Ridge}} = \underbrace{\sum_{i=1}^{m}(y_i - x_i^T\beta)^2}_{\text{least squares}} + \underbrace{\lambda\sum_{j=1}^{n}\beta_j^2}_{L2}$$

$$\text{Loss}_{\text{LASSO}} = \sum_{i=1}^{m}(y_i - x_i^T\beta)^2 + \lambda\sum_{j=1}^{n}|\beta_j|$$


- **LASSO (L1)** drives coefficients exactly to zero → performs *feature selection*. Use when you suspect many features are irrelevant, or you want a sparse, interpretable model.
- **Ridge (L2)** shrinks coefficients toward zero but (except under collinearity) not to zero → keeps all features. Use when features are correlated and you believe most carry some signal.

---

## 5. Classification and evaluation

**Regression → classification.** Threshold a continuous prediction: classify positive if $f(x) > T$. Setting $T = 0$ and absorbing the offset into $b$ gives a linear classifier via $\text{sign}(w^Tx + b)$. But sign outputs only $\pm 1$ — no confidence. Applying the logistic (sigmoid) function instead gives an output in $(0,1)$ you can read as a probability:

$$p(X) = \frac{e^{\beta_0 + \beta_1 X}}{1 + e^{\beta_0 + \beta_1 X}} \quad \Longleftrightarrow \quad \log\!\left(\frac{p(X)}{1-p(X)}\right) = \beta_0 + \beta_1 X$$

Coefficients are interpreted on the **log-odds** scale, not the probability scale. Linear regression fits by least squares; logistic regression fits by maximum likelihood.

**Confusion matrix and the metrics:**

| Metric | Formula |
|---|---|
| Precision (PPV) | TP / (TP + FP) |
| Recall (sensitivity, TPR) | TP / (TP + FN) |
| Specificity (TNR) | TN / (FP + TN) |
| Accuracy | (TP + TN) / (P + N) |
| Error rate | 1 − Accuracy |
| Balanced accuracy | (TPR + TNR) / 2 |

- **Type I error = false positive**; **Type II error = false negative**. Which is worse is application-dependent (screening tolerates FPs; confirmatory testing before an invasive procedure does not).
- **Class imbalance:** on a 90/10 dataset, always predicting the majority class gives 90% accuracy and is useless. Use balanced accuracy or MCC. MCC also distinguishes a random model (0) from a perfectly inverted one (−1), which balanced accuracy cannot.

**ROC / AUROC:** sweep the decision threshold from 0 to 1 and plot TPR vs. FPR at each. The most conservative threshold sits at (0,0); the most permissive at (1,1); the perfect classifier at (0,1). AUC = 1.0 perfect, 0.5 random, 0.0 perfectly wrong. AUC is the probability the model ranks a random positive above a random negative.

---

## 6. From lecture 3: Why R² goes up — and what it can't tell you

$$R^2 = 1 - \frac{\text{SSE}}{\text{SST}}, \qquad \text{SST} = \sum(y_i - \bar y)^2$$

$R^2$ is a ratio, so it moves when *either* term moves. Two distinct things inflate it, and neither one means the model got better at predicting.

### Mechanism 1: adding a predictor (fixed dataset, SST fixed)

Here the $y$ values never change, so SST is fixed and every change in $R^2$ is a change in SSE.

Fit a model with $p$ predictors, obtaining $\text{SSE}_p$. Now add a $(p+1)$-th column to $X$ and refit. The old solution is still available in the new parameter space — just set $\beta_{p+1} = 0$. OLS minimizes SSE over a strictly larger set that *contains* the previous solution, so:

$$\text{SSE}_{p+1} \le \text{SSE}_p \quad \Longrightarrow \quad R^2_{p+1} \ge R^2_p$$

The added feature can be pure noise and $R^2$ still won't go down. In the limit, with $n$ observations and $n-1$ predictors plus an intercept, you can fit the data exactly: SSE = 0, $R^2$ = 1, and the model has learned nothing.

### Mechanism 2: widening the range of x (model fixed, SST changes)

Here the model is identical and the *sample* changes, so SST is no longer fixed. Decompose it:

$$\text{SST} = \underbrace{\beta_1^2 S_{xx}}_{\text{signal}} + \underbrace{\text{SSE}}_{\text{noise}}
\qquad\Longrightarrow\qquad
R^2 = \frac{\beta_1^2 s_x^2}{\beta_1^2 s_x^2 + \sigma^2}$$

**$R^2$ increases when the spread of the data in x outgrows the noise** — that is, when the ratio $\beta_1^2 s_x^2 / \sigma^2$ increases. Sampling the same line over a wider range of x raises $R^2$ with an unchanged slope and unchanged prediction error.

Note that this is *not* a restatement of Mechanism 1. Adding a predictor doesn't change $s_x^2$ or $\sigma^2$; it lowers SSE by letting the fit bend toward the noise. Same symptom, different cause.
