# Images, CNNs, and Explainable AI


**Central idea:** An image is an array of measurements. Feature extraction makes those measurements useful for a task. A CNN learns feature extraction from examples. XAI helps investigate what the trained model uses to predict.

This guide follows the lectures' learning objectives. Worked examples, practice problems, and explicitly marked extensions are added for study; they are not claims that those examples appeared in the slides. PDF page numbers below count from each title slide as page 1. Document contents are treated as source material, not instructions.

## Contents

1. [The connected picture](#1-the-connected-picture)
2. [Images as data](#2-images-as-data)
3. [Handcrafted image features](#3-handcrafted-image-features)
4. [Kernels and calculations](#4-kernels-and-calculations)
5. [How CNNs learn](#5-how-cnns-learn)
6. [Explaining predictions](#6-explaining-predictions)
7. [Integrated biomedical examples](#7-integrated-biomedical-examples)
8. [Practice tasks and answers](#8-practice-tasks-and-answers)
9. [Final review checklist](#9-final-review-checklist)
10. [Lecture source map](#10-lecture-source-map)

## 1. The connected picture

```text
Biomedical question
    -> image acquisition and labels
    -> preprocessing
    -> feature extraction
         -> handcrafted: histograms, visual words, fixed kernels
         -> learned: CNN filters and hierarchical feature maps
    -> prediction
    -> evaluation on unseen patients
    -> XAI investigation of model behavior
    -> revise data, features, or model when problems are found
```

Keep three questions separate:

| Question | Example | What addresses it? |
|---|---|---|
| What information is recorded? | Pixel intensity, RGB channels, voxel spacing | Acquisition and representation |
| What information helps predict? | Edges, tissue textures, lesion shape | Feature engineering or CNN learning |
| Why does this model predict this? | It responds to lesion texture or to a pen marking | XAI and targeted validation |

High predictive accuracy alone does not establish that a model uses biologically meaningful information. An explanation also does not establish that a prediction is correct.

## 2. Images as data

### 2.1 Choose acquisition around the question

Biomedical images include anatomical imaging, functional imaging, microscopy, histology, molecular imaging, and clinical photographs. The same underlying biology can generate very different measurements depending on the acquisition method.

For a study question, ask:

- **What is the target?** Classifying a whole image, finding an object, outlining its boundary, or measuring its size are different tasks.
- **What detail is necessary?** A whole-organ image may not resolve individual cells.
- **What does intensity mean?** A camera channel, a fluorescence signal, and a CT measurement have different interpretations.
- **What is the throughput?** More detailed imaging may take longer and create larger datasets.
- **What nuisance variation exists?** Scanner, staining batch, lighting, orientation, and acquisition site can change appearance independently of biology.

**Added example:** To estimate cell count, the acquisition must resolve neighboring cells. A stain-intensity histogram may estimate overall staining but cannot reliably distinguish ten large cells from many small cells with the same overall intensity distribution.

### 2.2 Raster, vector, channels, and voxels

| Representation | What is stored? | Key intuition |
|---|---|---|
| Raster | A grid of pixel values | A sampled measurement at each location |
| Vector | Shapes, coordinates, drawing operations | A recipe for drawing an image |
| Grayscale image | Usually an `H × W` array | One intensity per pixel |
| RGB image | Often `H × W × 3` | Three values per spatial location |
| 3D image | A volume such as `D × H × W` | A voxel is a volume element |

A PDF can contain vector graphics, raster images, and text; the extension alone does not tell you that everything inside is vector-based.

**Added calculation:** A `256 × 256` RGB image contains

```text
256 × 256 × 3 = 196,608 channel values.
```

At 8 bits per channel, its raw pixel payload is 196,608 bytes, or 192 KiB, before headers and compression. Converting those values to 32-bit floating point makes that array four times larger.

The lecture uses 8-bit intensities from 0 to 255. Biomedical images can use other bit depths, floating-point values, or calibrated units. Do not assume every image should be interpreted on a 0-255 scale.

**Voxel intuition:** With spacing `1 × 1 × 3 mm`, one voxel occupies `3 mm³`. Ten voxels span different physical distances along different axes. Pixel or voxel counts become physical sizes only after accounting for spacing.

### 2.3 Compression

- **Lossless:** Reconstructs the original encoded pixel values exactly.
- **Lossy:** Discards some information to reduce storage further.

Lossless storage does not undo information already lost during acquisition, resizing, or earlier compression. A visually subtle compression artifact can still change fine textures or boundaries that a model uses.

### 2.4 Why flattened pixels can be difficult features

Flattening rearranges a matrix into a vector; it does not delete its values. However, a generic model no longer receives an explicit spatial neighborhood structure. It has to learn useful relationships without the local weight-sharing structure of a CNN.

A small shift moves an object into different vector positions. Rotation, resizing, lighting, and background changes can alter many entries while preserving the object's identity.

**Intuition:** A flattened representation asks, “What is the value at coordinate 4,271?” A local feature detector asks, “Is this pattern present here?”

Raw pixels can still work, especially on aligned, simple data. Their limitations motivate better structure and preprocessing rather than a blanket rule that pixels are unusable.

### 2.5 Preprocessing choices

| Operation | Why use it? | What can go wrong? |
|---|---|---|
| Resize or resample | Standardize input dimensions | Distort shape or physical scale |
| Intensity normalization | Reduce nuisance scale differences | Remove useful absolute intensity information |
| Color/stain adjustment | Reduce acquisition variation | Alter meaningful biological differences |
| Denoising | Suppress irrelevant fluctuations | Erase small structures |
| Augmentation | Teach tolerance to plausible variation | Create transformations that change the label |

**Added evaluation principle:** Split by patient before generating patches or augmentations. Fit dataset-level preprocessing and learned feature dictionaries on training data only. Otherwise, information can leak into evaluation.

## 3. Handcrafted image features

### 3.1 Color or intensity histograms

A histogram counts how many pixel values fall in each bin. Dividing each count by the number of pixels produces a normalized histogram.

**Added worked example:** For grayscale values

```text
0, 0, 1, 1, 1, 2, 3, 3
```

with one bin per value:

```text
Counts:      [2,    3,     1,     2]
Frequencies: [0.25, 0.375, 0.125, 0.25]
```

For color, one option is a histogram per channel. Another is a joint histogram over combinations of channel values. These preserve different information.

**Intuition:** A histogram is an inventory of intensities or colors. It says how much of each exists, without saying where it occurs.

**Useful for:** Overall staining, color composition, or intensity distributions when location is less important.

**Weak for:** Shape, boundaries, spatial arrangements, and tasks where identical colors arranged differently imply different labels.

Two images can have identical histograms and completely different anatomy.

### 3.2 Bag of visual words (BoVW)

A visual word is usually a cluster center in a numerical descriptor space, rather than a literal object with an obvious name.

Typical pipeline:

1. Extract local patches or keypoints from training images.
2. Convert each into a numerical descriptor.
3. Cluster training descriptors into `K` groups; centers form the visual vocabulary.
4. Assign each image's descriptors to visual words.
5. Count or normalize the assignments to obtain a length-`K` feature vector.
6. Train a classifier or regressor on those vectors.

**Added example:** With four words, an image might become `[12, 8, 15, 5]`. It contains 40 assigned patches, so its normalized representation is `[0.30, 0.20, 0.375, 0.125]`.

**Intuition:** Describe a biopsy as a mixture of recurring tissue appearances. A pathologist may later inspect representative patches and give some clusters biological interpretations.

**Advantages:** Fixed-length representation, reduced dimensionality, local texture information, some tolerance to patch rearrangement.

**Limitations:** A basic bag discards patch locations; vocabulary choice matters; cluster centers need not correspond to biological concepts. Counting patches also weights tissue by the sampling scheme.

In the kidney-biopsy example, the lecture describes **57 patients and 107,471 tissue patches**. The patches are not 107,471 independent patients. The lecture ranks nine visual words by random-forest importance and interprets examples as glomerular and interstitial/tubular patterns. This is a model association, not evidence that a visual word causes kidney dysfunction. [Lecture 28, p. 17.]

### 3.3 Edges

An edge is a rapid spatial change in intensity or color. Edges can approximate object boundaries, but lighting, noise, artifacts, and internal texture can also create edges.

A discrete derivative compares nearby values. For

```text
I = [1, 1, 1, 2, 2]
```

the forward differences are

```text
I[i+1] - I[i] = [0, 0, 1, 0].
```

This produces four values unless a boundary convention is added. Centered differences use a different comparison and can respond at two neighboring locations around a step. Always state the convention instead of memorizing a derivative vector without it.

**Intuition:** An edge detector responds to change, not brightness alone. A uniformly bright patch has no interior edge.

## 4. Kernels and calculations

### 4.1 The operation

Place a small matrix over an image patch. Multiply corresponding entries, sum the products, record one output value, and move to the next location.

For a grayscale patch under the common CNN convention:

$$
y_{r,c}=\sum_{u=0}^{k_h-1}\sum_{v=0}^{k_w-1}K_{u,v}I_{r+u,c+v}.
$$

This expression uses stride 1 and no padding. **Mathematical convolution flips the kernel in both spatial directions; cross-correlation does not.** CNN libraries commonly call cross-correlation “convolution.” Use the convention specified in a problem.

### 4.2 Common kernels

```text
Box blur B = (1/9) ×     Prewitt Px =           Prewitt Py =
[ 1  1  1 ]             [ 1  0 -1 ]           [ 1  1  1 ]
[ 1  1  1 ]             [ 1  0 -1 ]           [ 0  0  0 ]
[ 1  1  1 ]             [ 1  0 -1 ]           [-1 -1 -1 ]

Laplacian L =           Sharpen S =            Identity delta =
[ 0  1  0 ]             [ 0 -1  0 ]           [ 0  0  0 ]
[ 1 -4  1 ]             [-1  5 -1 ]           [ 0  1  0 ]
[ 0  1  0 ]             [ 0 -1  0 ]           [ 0  0  0 ]
```

The lecture shows this `Px` sign convention. Opposite signs are also common for derivative filters. `Px` measures left-right intensity differences and therefore responds to **vertical boundaries**; `Py` measures top-bottom differences and responds to **horizontal boundaries**.

| Kernel | Operation and intuition | Likely appearance |
|---|---|---|
| Box blur | Average neighbors; suppress rapid variation | Smoother detail and softer boundaries |
| Prewitt | Approximate directional first derivatives | Directional edges |
| Laplacian | Approximate a sum of second derivatives | Strong responses around intensity transitions; also noise-sensitive |
| Sharpen | Original image minus this Laplacian | Enhanced local contrast with original structure retained |

The sharpening kernel boosts high-frequency detail while retaining a constant background. It is not a pure zero-sum high-pass filter.

### 4.3 Worked edge response

Apply `Px` without flipping to

```text
Patch = [1 1 2]
        [1 1 2]
        [1 1 2]
```

Each row contributes `1×1 + 1×0 + 2×(-1) = -1`. The total is **-3**.

Flipping the kernel gives **+3**. Taking the magnitude gives **3** in either case. This explains why an edge visualization can display positive values even when a direct signed calculation gives a negative response.

A common combined edge-strength measure is

$$G=\sqrt{G_x^2+G_y^2}.$$

The sign carries direction information; magnitude measures strength.

### 4.4 Worked blur, Laplacian, and sharpening

Use a bright center surrounded by zeros:

```text
Patch = [0 0 0]
        [0 9 0]
        [0 0 0]
```

At the center:

- Box blur: `9/9 = 1`.
- Laplacian: `-4×9 = -36`.
- Sharpen: `5×9 = 45`.

A filter's output can be negative or exceed the input's original range. Clipping or rescaling is a separate display/storage operation. Clipping intermediate calculations can change later results.

**Constant-patch check:** On a patch filled with `c`, blur returns `c`, Prewitt and Laplacian return `0`, and sharpening returns `c`. Kernel sums provide a quick sanity check: blur and sharpening sum to 1; these derivative kernels sum to 0.

### 4.5 Combining kernels

Let `B` be a blur and `delta` the identity kernel.

```text
Detail = I - blur(I)
       = (delta - B) applied to I

Sharpened = I + alpha × Detail
          = [(1 + alpha)delta - alpha B] applied to I
```

This is an added example of unsharp masking: isolate rapid variation, then add some back. It can enhance noise too.

With the Laplacian defined above, the displayed sharpen kernel satisfies `S = delta - L`.

Sequential linear filters can be combined into one effective kernel, subject to matching boundary handling. This does not generally work across nonlinear activations or max pooling.

## 5. How CNNs learn

### 5.1 Fixed versus learned filters

A handcrafted filter is chosen before training. A CNN learns filter weights to reduce a prediction loss.

```text
Image -> convolution -> activation -> optional pooling
      -> convolution -> activation -> ... -> prediction head
      -> loss against labels -> backpropagation -> update weights
```

Weights across layers are usually trained jointly. The loss supervises the final task; it does not necessarily tell each filter to become a named biological detector.

**Intuition:** A CNN repeatedly uses small pattern detectors across an image. Later detectors combine earlier responses over larger regions.

Typical hierarchy:

| Level | Possible learned pattern | Biomedical intuition |
|---|---|---|
| Early | Edge, color transition, small texture | A nuclear boundary or stain transition |
| Middle | Combinations of local features | A nucleus-like shape or gland fragment |
| Deep | Larger configurations | Tissue organization associated with the label |

These are useful hypotheses, not guarantees. Learned channels may encode mixed or unintuitive properties.

### 5.2 Local connectivity and weight sharing

- **Local connectivity:** Each output location uses a small neighborhood.
- **Weight sharing:** The same filter weights are reused at all spatial positions.

A nucleus-like pattern may be informative wherever it appears. Reusing a detector reduces parameters compared with learning an independent connection for every pixel-output pair.

Convolution tends toward translation **equivariance**: shifting an input shifts its feature responses, ignoring boundary and sampling complications. Classification-level invariance requires additional architecture or aggregation and is not perfect. CNNs are not automatically rotation-invariant.

### 5.3 Channels and parameters — added quantitative extension

An RGB filter spans all three input channels. A `3 × 3` filter for RGB therefore has `3 × 3 × 3 = 27` weights, plus an optional bias. Summing over channels produces one output map per output filter.

For a standard dense convolutional layer with one bias per output channel:

$$\text{parameters}=(k_h k_w C_{in}+1)C_{out}.$$

**Example:** Sixteen `3 × 3` filters applied to RGB input have

```text
(3 × 3 × 3 + 1) × 16 = 448 parameters.
```

Each filter's weights are reused spatially. Increasing image size increases computation and activations, but does not change this layer's parameter count.

### 5.4 Output dimensions — added quantitative extension

For either spatial axis, with dilation 1 and symmetric padding:

$$N_{out}=\left\lfloor\frac{N_{in}+2P-K}{S}\right\rfloor+1.$$

Here `K` is kernel size, `P` padding on each side, and `S` stride.

**Worked example:** `32 × 32 × 3` input, sixteen `3 × 3` filters, stride 1, no padding:

```text
Spatial output: floor((32 - 3)/1) + 1 = 30
Output shape: 30 × 30 × 16
```

With padding 1, the output is `32 × 32 × 16`. Padding specifies how to handle boundaries; zeros are one option.

### 5.5 Hyperparameter intuition

| Choice | Effect | Tradeoff |
|---|---|---|
| Larger kernel | Examines a larger neighborhood per layer | More weights and computation at fixed channels |
| More filters | Produces more feature channels | Greater capacity, computation, and potential overfitting |
| Larger stride | Samples output locations farther apart | Smaller maps and less spatial detail |
| More layers | Composes features into larger patterns | Greater training difficulty and complexity |
| Pooling | Summarizes neighborhoods within each channel | Can lose precise location and small structures |

A small kernel can contribute to a large effective receptive field after stacking. Two `3 × 3`, stride-1 layers have a nominal `5 × 5` receptive field away from boundaries. Activations between them make this different from a single linear `5 × 5` filter.

ReLU is `max(0, z)`. Nonlinearities let stacked layers learn richer functions than a composition of purely linear filters.

### 5.6 Pooling by hand

Using the lecture's example, with a `2 × 2` window, stride 1, no padding:

```text
Input:             Max pool:        Mean pool:
[0 1 2]            [4 5]            [2 3]
[3 4 5]            [7 8]            [5 6]
[6 7 8]
```

The first max is `max(0,1,3,4)=4`; the first mean is `(0+1+3+4)/4=2`.

**Max intuition:** “Was there a strong response anywhere in this neighborhood?”

**Mean intuition:** “What was the average response in this neighborhood?”

Pooling can reduce sensitivity to small shifts within a window, but does not guarantee invariance. Its output size depends on window, stride, and padding. Pooling does not itself remove convolution weights; smaller maps can reduce the parameters of a later flattened dense layer.

**Clarification of Lecture 29, p. 26:** Neither strided convolution nor pooling universally preserves more information, and pooling is not universally preferred. Both can discard spatial detail. A strided convolution uses learned filters; standard max/mean pooling uses a fixed rule. Task and architecture determine the choice.

### 5.7 Other ordered data

- **ECG/time series:** 1D filters detect local waveform patterns across time.
- **Peptides:** Sequence filters can detect local residue motifs that recur at different positions.
- **Text:** Local token patterns can be informative across positions.
- **Graphs:** Neighborhood aggregation extends the local-pattern idea, but graph operations are not simply rectangular image kernels.

The benefit depends on whether locality and reuse are meaningful. Convolution is not automatically appropriate for arbitrarily ordered tabular columns.

## 6. Explaining predictions

### 6.1 Scope and purpose

XAI investigates model behavior to reveal useful associations, identify shortcuts, support human interpretation, and guide improvement.

| Scope | Question |
|---|---|
| Single prediction | Why this output for this patient/image? |
| Local region | How does behavior change near this example? |
| Global | What patterns characterize behavior across a population? |

**Intrinsic interpretation** uses the model's own structure, such as a small tree. **Post-hoc interpretation** analyzes an already-trained model. Some methods are model-agnostic; others require specific model internals.

An accuracy-interpretability tradeoff can occur, but complexity does not guarantee higher accuracy. A small tree can be understandable, while a very large tree may not be. Some models, such as nonlinear-kernel SVMs, are also difficult to interpret despite being less elaborate than deep networks.

### 6.2 Regression and trees

For `y_hat = b0 + b1*x1 + b2*x2`, increasing `x1` by one unit changes the prediction by `b1`, holding other predictors fixed. Interpretation depends on units, coding, transformations, and the model's assumptions.

**Added example:** For `y_hat = 10 + 2*age - 3*treatment`, a five-unit increase in age adds 10 prediction units at fixed treatment. Changing treatment from 0 to 1 subtracts 3 at fixed age. This describes the fitted model, not a causal treatment effect.

Large coefficients do not automatically mean greater importance: age measured in years and a concentration measured in tiny units cannot be compared by raw coefficient magnitude alone.

A decision-tree explanation follows the actual path from root to leaf. Splits describe prediction rules rather than proving mechanisms.

### 6.3 Perturbation and permutation importance

Basic procedure:

1. Evaluate the trained model on an appropriate held-out set.
2. Perturb one feature, such as by shuffling it across samples.
3. Re-evaluate without retraining.
4. Repeat to assess variability.

For a higher-is-better score:

$$Importance_j=Score_{baseline}-Score_{perturbed,j}.$$

**Added example:** Baseline accuracy is 0.90. Shuffling texture drops it to 0.73, yielding importance 0.17. Shuffling color drops it to 0.89, yielding 0.01. This model relies more on the texture feature under this evaluation and perturbation scheme.

If the metric is a loss, use the increase in loss instead. A negative importance can occur because of sampling variability or harmful reliance on a feature.

**Limitations:** Correlated predictors may substitute for one another, making each appear less important individually. Perturbation can also create unrealistic combinations. A drop measures model reliance under the chosen intervention, not biological causation. Basic individual-feature scores do not explain an interaction's structure; grouped perturbations can probe related features together.

### 6.4 Anchors versus counterfactuals

| Method | Main question | Added hypothetical example |
|---|---|---|
| Anchor | Which conditions keep the prediction stable with high probability under specified perturbations? | Fix two measured features; varying the rest usually preserves the class |
| Counterfactual | What feasible change would produce a different prediction? | Change selected image characteristics until the predicted class flips |

An anchor is a sufficient local rule under a sampling distribution, not necessarily a globally valid rule. A counterfactual is often sought with minimal change, but “minimal” depends on a distance definition and feasibility constraints.

For biomedical images, an edited image can flip a prediction while being biologically implausible. A model counterfactual is not automatically a treatment recommendation or a possible disease trajectory.

### 6.5 Surrogate models

A surrogate is an interpretable model trained to imitate the original model's outputs:

```text
Inputs X -> complex model f -> predictions f(X)
Inputs X and targets f(X) -> train interpretable surrogate g
```

The surrogate's targets are the original model's predictions, rather than ground-truth labels. Inputs may be observed examples or intentionally generated samples.

**Critical distinction:** Original-model accuracy measures agreement with real labels; surrogate **fidelity** measures agreement with the original model. Evaluate fidelity on relevant held-out inputs. An easy-to-read tree is misleading if it poorly reproduces the behavior it claims to explain.

### 6.6 LIME

LIME stands for **Local Interpretable Model-agnostic Explanations**.

1. Choose the instance to explain.
2. Generate perturbed examples around it.
3. Obtain the original model's predictions.
4. Weight examples by proximity to the chosen instance.
5. Fit a simple local surrogate.
6. Interpret its coefficients or selected regions.

**Image intuition:** Toggle groups of neighboring pixels (superpixels) and ask which groups support the prediction locally.

A schematic objective is:

$$\min_g \sum_i \pi_x(z_i)[f(z_i)-g(z_i)]^2+\Omega(g).$$

`pi` weights locality and `Omega` penalizes complexity. This is intuition for the approach, not a requirement that every implementation use exactly this loss.

**Limitations:** Results depend on neighborhood sampling, segmentation, distance, masking, and randomness. A strong local explanation need not describe global behavior.

**Clarification of Lecture 30, p. 12:** Original LIME includes a feature-selection approach called K-LASSO, but that is not the definition of every LIME implementation. Aggregating many local explanation magnitudes can summarize patterns, but there is no universal LIME global-importance formula that is valid independently of the sampling and aggregation choices.

### 6.7 SHAP

SHAP assigns contributions to features relative to a reference or background, using Shapley-value ideas from cooperative games.

For an additive explanation:

$$f(x)=\phi_0+\sum_j\phi_j.$$

Here `phi_0` is the baseline and `phi_j` is a feature's contribution for this instance, on the explained output scale.

**Added arithmetic example:** If the explained output is a probability:

```text
Baseline:             0.20
Texture contribution: +0.30
Border contribution:  +0.15
Color contribution:   -0.05
Prediction:           0.60
```

Positive contributions push above the baseline; negative contributions push below it. These are illustrative values. Some explainers operate in a score or log-odds space; those contributions cannot be added directly as probability changes.

A common global summary is:

$$GlobalImportance_j=\frac{1}{n}\sum_{i=1}^n|\phi_j(x_i)|.$$

Use absolute values for this magnitude summary because positive and negative contributions can cancel in a signed average. A SHAP distribution plot additionally reveals direction and variability.

**Limitations:** Background choice and feature-dependence assumptions affect attribution. Correlation can complicate how credit is assigned. Similar attribution patterns can identify similar model behavior without proving biological equivalence. SHAP values do not establish causality.

### 6.8 Self-explainable AI

These models build interpretable structure or explanation outputs into the model, for example by organizing computation around biological concepts or pathways.

**Opportunity:** Explanations can align better with domain concepts.

**Challenge:** Concept definitions, architecture constraints, and incomplete biological knowledge can limit the model. Human-readable output is useful only if it faithfully reflects the computation; evaluate that faithfulness rather than assuming it.

### 6.9 Choose a method by the question

| Need | Useful approach | Check |
|---|---|---|
| Which features support held-out performance? | Permutation or grouped perturbation | Correlations, metric, realistic perturbations |
| Why this one image? | LIME or suitable local attribution | Masking/segmentation and stability |
| How does one prediction differ from baseline? | SHAP | Background, output scale, dependence assumptions |
| What simple rules approximate behavior overall? | Surrogate tree | Coverage and held-out fidelity |
| What preserves the prediction? | Anchor | Rule precision and coverage under its perturbation distribution |
| What change flips the prediction? | Counterfactual | Feasibility and plausibility |
| Can architecture reflect biological concepts? | Self-explainable model | Concept quality, accuracy, faithfulness |

## 7. Integrated biomedical examples

### 7.1 Skin-lesion classification and shortcuts

**Lecture connection:** Lecture 28, p. 26 reports that adding markings to 107 benign lesions changed mean model melanoma score from **0.16 to 0.54**, with **P < .001**. This illustrates reliance on an image artifact; it is not a statement about diagnostic performance of all CNNs.

**Added analysis workflow:**

1. Define the label, patient population, acquisition conditions, and intended prediction task.
2. Use patient-separated evaluation and examine performance across acquisition sources.
3. Inspect local image explanations for attention to markings, rulers, borders, or background.
4. Compare predictions for carefully controlled marked/unmarked versions.
5. Check whether such edits introduce their own artifacts.
6. Retrain or revise data collection if a shortcut is found, then re-evaluate on held-out data.

**Intuition:** A model is rewarded for predicting the label. Without additional constraints, it can use any predictive cue available in training data.

### 7.2 Kidney-biopsy texture

A histogram captures the amount of each intensity or color. BoVW captures frequencies of local appearances. A CNN can learn local appearances and their hierarchical combinations.

**Added task choice:** If tissue composition drives prediction, BoVW may provide a useful baseline. If arrangement of tubules and surrounding tissue matters, a simple bag may discard necessary information. Compare approaches on the same patient-level evaluation.

For interpretation, inspect representative visual-word patches, test whether importance is stable across resampling, and account for correlated tissue patterns. Feature importance alone does not turn a cluster into a validated biomarker.

### 7.3 Boundary-sensitive tasks

**Added example:** For outlining a small lesion, edges can help identify rapid intensity changes, but noise and neighboring tissue boundaries also create edges. A classifier answers whether a lesion is present; segmentation estimates its spatial extent.

Aggressive downsampling may remove the small structure needed for segmentation. Kernel size, physical image resolution, and receptive field should be considered together. A `3 × 3` kernel sees different physical areas at different voxel spacings.

## 8. Practice tasks and answers

Try each task before reading the answer.

### Task 1: Storage

A `128 × 128` grayscale image uses 16 bits per pixel. How large is the raw pixel payload?

**Answer:** `128 × 128 × 2 = 32,768 bytes = 32 KiB`, excluding metadata and compression.

### Task 2: Representation

Two images have the same normalized color histogram. Must they depict the same tissue structure?

**Answer:** No. The histogram preserves color frequencies but discards arrangement. Rearranging pixels preserves the histogram while changing structure.

### Task 3: BoVW

A vocabulary has 20 words, and one image yields 500 patch descriptors. What is its basic BoVW feature-vector length?

**Answer:** 20, one count or frequency per visual word. The number of patches changes the counts, not vocabulary length.

### Task 4: Edge response

Apply `Px = [[1,0,-1],[1,0,-1],[1,0,-1]]` without flipping to `[[2,2,5],[2,2,5],[2,2,5]]`.

**Answer:** Each row contributes `2-5=-3`; total `-9`. Magnitude is 9. Flipping the kernel changes the sign.

### Task 5: Recognize a filter

A `3 × 3` filter has every entry equal to `1/9`. What happens to a uniform region and to an isolated bright pixel?

**Answer:** A uniform interior region stays unchanged. The isolated bright pixel is averaged into neighboring outputs, spreading and reducing its peak. Boundary behavior depends on padding.

### Task 6: Sharpening

For a patch with center 10 and its four direct neighbors all 8, compute the displayed sharpening kernel's response.

**Answer:** `5×10 - 8 - 8 - 8 - 8 = 18`. The center-to-neighbor contrast is enhanced.

### Task 7: CNN output shape

A `28 × 28 × 1` image enters a layer with eight `5 × 5` filters, stride 1, no padding. What are output shape and parameter count, including biases?

**Answer:** Spatial size `28-5+1=24`; output `24 × 24 × 8`. Parameters: `(5×5×1+1)×8=208`.

### Task 8: Stride

A spatial dimension is 31; kernel size is 3, padding is 0, stride is 2. What is the output dimension?

**Answer:** `floor((31-3)/2)+1=15`.

### Task 9: Pooling

Apply `2 × 2` max pooling, stride 2, no padding:

```text
[1 3 2 0]
[4 2 1 5]
[0 1 8 2]
[2 3 4 6]
```

**Answer:** `[[4,5],[3,8]]`. Mean pooling gives `[[2.5,2],[1.5,5]]`.

### Task 10: Why nonlinear activations?

Why not simply stack many linear convolutions?

**Answer:** Compositions of linear operations remain linear, and biases make the overall map affine. Nonlinearities enable richer input-dependent behavior and more expressive feature combinations.

### Task 11: Permutation importance

Held-out accuracy is 0.86. After shuffling feature A, accuracy is 0.70. After shuffling B, it is 0.85. Is A biologically more causal?

**Answer:** The estimated importance drops are 0.16 and 0.01. A has greater measured model reliance under this procedure. Neither score establishes biological causation; correlation and unrealistic perturbations can affect the result.

### Task 12: Surrogate fidelity

A small tree matches the CNN on only 65% of relevant held-out inputs. Can its rules be presented as a faithful global explanation?

**Answer:** The low agreement is a major limitation. Readable rules do not compensate for poor fidelity. Restrict the claimed scope or improve the surrogate and evaluate again.

### Task 13: SHAP arithmetic

On an explained score scale, baseline is 1.0 and contributions are `+0.8`, `-0.4`, and `+0.2`. What is the explained prediction?

**Answer:** `1.0+0.8-0.4+0.2=1.6` on that score scale. Do not label it a probability unless that is the explained output scale.

### Task 14: Signed cancellation

A feature has SHAP values `+2` and `-2` for two examples. Is its global importance zero?

**Answer:** Its signed mean is zero, but mean absolute SHAP is `(2+2)/2=2`. It matters strongly in opposite directions for these examples.

### Task 15: Anchor or counterfactual?

A researcher asks (a) which conditions preserve a predicted class and (b) which feasible changes flip it. Which method corresponds to each?

**Answer:** (a) Anchor; (b) counterfactual. Both depend on the perturbation or feasibility assumptions used.

### Task 16: Design a complete analysis

You have 100 patients and 200 histology patches per patient. Propose a CNN evaluation and explanation plan.

**Answer:** Split patients across training, validation, and test sets before patch processing; keep all patches from a patient together. Fit learned preprocessing only on training data, tune on validation data, and evaluate the intended patient-level outcome with an appropriate aggregation rule. Inspect explanations across patients and acquisition groups; test suspected artifacts through controlled comparisons. Twenty thousand patches do not imply twenty thousand independent patient observations.

## 9. Final review

You should be able to:

- Explain raster versus vector storage, channels, voxels, bit depth, and compression.
- Connect acquisition resolution and throughput to a biomedical question.
- Explain why pixel shifts, lighting, and dimensionality challenge simple pixel-based models.
- Construct and interpret normalized histograms and BoVW features.
- State what spatial information each feature representation preserves or discards.
- Calculate a filter response and specify whether the kernel is flipped.
- Write box-blur, Prewitt, Laplacian, and sharpening kernels and recognize their effects.
- Explain how a detail map or sharpened image can be formed by combining filters.
- Describe CNN learning, locality, weight sharing, feature maps, activations, and hierarchy.
- Calculate convolution dimensions, parameter counts, and pooling outputs for the added quantitative examples.
- Predict effects of kernel size, stride, filter count, pooling, and depth.
- Explain how locality can apply to sequences and how graphs require adapted operations.
- Distinguish local and global explanations, intrinsic and post-hoc methods, and accuracy and fidelity.
- Interpret coefficients, perturbation scores, anchors, counterfactuals, LIME, and SHAP.
- Explain why an attribution, plausible explanation, or accurate prediction does not by itself establish causality.
- Design a test for whether a biomedical model uses an artifact instead of the intended signal.

**Five ideas to retain:** Representation determines what information a model can use. Kernels turn neighborhoods into responses. CNNs learn reusable local detectors and compose them. Evaluation must reflect genuinely unseen patients and settings. XAI provides evidence about model behavior that must itself be checked.
