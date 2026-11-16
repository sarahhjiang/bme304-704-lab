# Lecture Recap: Paired Data, Autoencoders, and Transformers

**Central idea:** Useful representations depend on what we ask a model to preserve: similarity between observations, information needed to reconstruct an input, or relationships among parts of an input.

## 1. The conceptual map

| Approach | What is related or compared? | What does the model learn? |
|---|---|---|
| Paired statistical analysis | Two linked measurements | The average within-pair difference |
| Siamese / metric learning | Two observations | A representation whose distances reflect task-specific similarity |
| Autoencoder | Input and its reconstruction | A representation that retains information needed for reconstruction |
| Variational autoencoder (VAE) | Input and reconstruction, plus latent distribution and prior | A regularized latent space that supports sampling |
| Attention | A query and candidate keys, followed by weighted values | How to combine relevant information for each position |

An **embedding** is a numerical representation of an observation. Its usefulness depends on the training objective: good reconstruction alone does not guarantee good class separation.

## 2. Paired data: what makes two observations a pair?

A pair consists of two observations considered together to answer a question. The relationship may come from **study design** or from an **ML task**.

- **Homogeneous pairs:** Same data type, such as patient–patient, ECG–ECG, or drug–drug.
- **Heterogeneous pairs:** Different data types, such as image–report, drug–protein, or gene–disease phenotype.

The target can describe a change, similarity, a match, or an interaction. A drug–protein interaction model, for example, need not interpret the two inputs as “similar.” It may combine their representations to predict binding or activity.

### Paired clinical designs

| Design | What is paired? | Typical difference | Main consideration |
|---|---|---|---|
| Single-group pre–post | The same patient before and after treatment | After − before | Change alone does not isolate a treatment effect from other changes over time |
| Matched-pair randomized trial | Two participants matched on relevant baseline characteristics | Study treatment − control/standard treatment | Randomize treatment within pairs; matching quality matters |
| Crossover trial | The same participant under both treatments | Treatment A − treatment B | Randomize order and account for period and carryover effects |

**Example:** Blood pressure falls from 146 to 130 mmHg. With “after − before,” the within-patient difference is **−16 mmHg**. Choose a sign convention and use it consistently.

Matching can reduce variability when matching characteristics predict outcomes. In crossover studies, each patient acts as their own control. A washout period aims to reduce carryover; it does not automatically remove every order or period effect.

### Paired t-test

The paired t-test asks whether the **population mean of the within-pair differences** equals a specified value, usually zero. It analyzes one difference per pair.

**Study supplement:** For (n) pairs,

$$
d_i=y_i-x_i,\qquad t=\frac{\bar d-\mu_{d,0}}{s_d/\sqrt n},\qquad df=n-1.
$$

Here, \(\bar d\) is the mean difference and \(s_d\) is the standard deviation of differences. The independent units are the pairs; differences should be independent across pairs. For exact small-sample inference, the differences are assumed normally distributed. Complex crossover designs may need models that explicitly account for period and carryover effects.

**ML example:** Evaluate two models on the same five datasets and calculate one performance difference per dataset. Those differences form the paired comparison, provided the datasets can reasonably be treated as independent units.

## 3. Metric learning and Siamese networks

### Why learn distances?

A conventional classifier has an output score for each class it was trained to predict. A classifier trained on rhythms A, B, and C does not automatically have a class D output.

An encoder instead maps each ECG into a feature space:

```text
ECG → encoder → embedding → compare with labeled references → predicted label
```

If distances reflect clinically meaningful similarity, a new query can be classified by its nearest labeled reference, including a reference from a new class.

### Positive and negative pairs

- **Positive pair:** Shares the property the task cares about, such as the same rhythm.
- **Negative pair:** Does not share that property.

Pair labels are **task-dependent**. Two ECGs from the same patient could be positive for patient identity but negative for rhythm if their rhythms differ.

### Siamese architecture

```text
Input 1 → shared encoder fθ → embedding 1 ┐
                                        ├→ distance → pair loss
Input 2 → shared encoder fθ → embedding 2 ┘
```

The two branches use **the same weights**, placing both inputs in a consistent representation space. Training updates the encoder using both inputs.

### Contrastive loss and its margin

The slides emphasize two behaviors:

- Positive pairs are penalized for being far apart.
- Negative pairs are penalized for being closer than a margin \(m\).

**Study supplement:** One common loss, defining \(y=1\) for a positive pair, is

$$
L=yD^2+(1-y)\max(0,m-D)^2,
\qquad D=\|f_\theta(x_1)-f_\theta(x_2)\|_2.
$$

For a negative pair, the loss becomes zero once \(D\ge m\). The objective therefore does **not** require negative pairs to move infinitely far apart. Label conventions vary across implementations; check the definition of \(y\).

## 4. One-shot learning versus kNN

**One-shot learning** means having one labeled reference, or support example, per class in the new task. It does not mean the encoder was trained using only one example overall.

Typical workflow:

1. Train an encoder on many labeled examples from base classes A, B, and C.
2. Obtain one labeled support example for each new class D, E, and F.
3. Encode the query and support examples with the same encoder.
4. Assign the nearest support example’s label.

**Slide example:** Query-to-reference distances are D: 3.6, E: 1.0, and F: 4.5. The nearest-reference prediction is **E**.

| Term | What it counts |
|---|---|
| **k** in kNN | Neighbors consulted when predicting |
| **Shot** | Labeled support examples available per class |

The lecture’s one-shot example uses **1-NN in a learned embedding space**. kNN can use learned features too; the concepts overlap rather than describe mutually exclusive algorithms.

### Opportunities and limits

The slides highlight cancer classification from gene expression and drug-development assays. A learned similarity may help when class-specific labels are scarce, but success depends on how well new examples resemble the training domain. The drug-development example warns that performance can collapse for very different molecules or assays.

**Evaluation checklist:**

- Split by **patient before constructing pairs** to prevent patient leakage.
- Hold out entire **classes** when testing novel-class recognition.
- Check whether support examples represent their classes.
- Consider changes in patients, devices, assays, and acquisition conditions.
- Many generated pairs are dependent or redundant; more pairs do not equal more independent observations.
- Pair construction can increase computation substantially.

## 5. Autoencoders: learning through reconstruction

An autoencoder learns to encode an input and reconstruct it:

$$
z=f_\theta(x),\qquad \hat x=g_\phi(z).
$$

```text
Input x → encoder → latent vector z → decoder → reconstruction x̂
  └──────────────── compare x and x̂ ──────────────────────────┘
```

- **Encoder:** Maps the input into a representation.
- **Latent vector / embedding / encoded representation:** The encoder’s output in this context.
- **Latent space:** The space containing these vectors.
- **Decoder:** Maps the latent representation to the output.

Training compares the reconstruction with its target and updates both networks.

### Reconstruction loss

Mean squared error is a common reconstruction loss. Averaging over \(N\) scalar signal values,

$$
\operatorname{MSE}=\frac{1}{N}\sum_{j=1}^{N}(x_j-\hat x_j)^2.
$$

**Worked example from the slides:**

| Signal sample | Target | Reconstruction | Squared error |
|---|---:|---:|---:|
| 1 | 0 | 0.1 | 0.01 |
| 2 | 1 | 0.8 | 0.04 |
| 3 | 0 | 0.1 | 0.01 |

MSE = (0.01 + 0.04 + 0.01) / 3 = **0.02**.

### Undercomplete and overcomplete representations

- **Undercomplete:** Latent dimension is smaller than input dimension, creating a compression bottleneck.
- **Overcomplete:** Latent dimension is larger than input dimension. Constraints such as sparsity can encourage useful features instead of simple copying.

### The target defines the task

| Task | Input | Training target |
|---|---|---|
| Standard reconstruction | ECG | Original ECG |
| Denoising | Noisy ECG or image | Clean signal or image |
| Sequence-to-sequence mapping | Clinical sentence | Summary |
| Image reconstruction | 2D measurements plus learned prior information | 3D volume |

An **encoder–decoder** is a broad design pattern. Not every encoder–decoder is an autoencoder: a report generator predicts a different target rather than reconstructing its input. CNNs, RNNs, and attention can be components within encoder–decoder systems.

## 6. Variational autoencoders: making latent sampling useful

A standard autoencoder learns to reconstruct observed inputs, but it provides no guarantee that an arbitrary latent vector—or a small perturbation of one—will decode to a plausible example.

A VAE encodes an input into a **distribution of latent vectors**, commonly represented by a mean \(\mu\) and standard deviation \(\sigma\):

$$
z=\mu+\sigma\odot\epsilon,\qquad \epsilon\sim\mathcal N(0,I).
$$

The symbol \(\odot\) denotes elementwise multiplication. This sampling expression is the reparameterization used during training.

### Two training objectives

1. **Reconstruction:** Preserve enough information to reproduce the observed input.
2. **KL regularization:** Encourage the input-conditioned latent distribution to resemble a chosen prior, commonly \(\mathcal N(0,I)\).

**Study supplement:** The usual minimization objective can be written

$$
\mathcal L(x)=\mathbb E_{q_\phi(z\mid x)}[-\log p_\theta(x\mid z)]
+D_{\mathrm{KL}}\big(q_\phi(z\mid x)\,\|\,p(z)\big).
$$

The first term is a reconstruction term; its precise form depends on the output model. MSE is one possible choice, not a universal requirement.

**KL divergence** compares probability distributions and is generally asymmetric:

$$
D_{\mathrm{KL}}(P\|Q)\ne D_{\mathrm{KL}}(Q\|P).
$$

### Reconstruction versus generation

- **Reconstruct:** Encode an observed input, sample its latent distribution, and decode.
- **Generate:** Sample \(z\) from the prior and decode, without supplying an observed input to the encoder.

Regularization encourages a latent space suitable for sampling. It does not guarantee that every generated molecule or signal is valid or useful.

## 7. Attention: combining information using context

Attention combines information with **learned, input-dependent weights**. It provides a direct way to connect distant positions; recurrence and convolution can also capture long-range relationships, but may require more steps or layers.

**Example:** In “No evidence of pneumonia,” representing “pneumonia” requires context, especially the negation.

The slides give illustrative weights:

| Source word | Weight |
|---|---:|
| No | 0.55 |
| evidence | 0.10 |
| of | 0.05 |
| pneumonia | 0.30 |

The output is the weighted sum of the words’ **value vectors**, not a weighted sum of their text labels. These numbers illustrate the mechanism; they are not reported clinical model findings.

### Queries, keys, and values

| Component | Role |
|---|---|
| **Query (Q)** | What information does this position seek? |
| **Key (K)** | How should another position be matched to that query? |
| **Value (V)** | What information does that position contribute? |

For input representation matrix \(X\), learned projections create

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.
$$

**Study supplement: scaled dot-product attention**

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

Read this in four steps:

1. Compare each query with candidate keys using dot products.
2. Scale scores by \(\sqrt{d_k}\), where \(d_k\) is the key dimension.
3. Apply softmax across candidate keys to obtain normalized weights.
4. Use those weights to combine value vectors.

Each row of the attention-weight matrix describes what information one query position combines. A mask can exclude positions from that combination.

## 8. Transformers: architectures built around attention

**Attention is an operation; a transformer is an architecture that uses it.** Transformer layers combine attention with position-wise feed-forward processing, and stacked layers repeatedly refine representations.

### Core distinctions

- **Self-attention:** Queries, keys, and values come from the same sequence of representations.
- **Cross-attention:** Queries come from one representation stream; keys and values come from another. In an encoder–decoder model, decoder positions can attend to encoder outputs.
- **Multi-head attention:** Several attention heads operate in parallel, allowing different learned relationships. Their outputs are combined. Multiple heads can be used in either self-attention or cross-attention.
- **Positional information:** Supplies order or position information that attention alone does not inherently provide. The slides describe sinusoidal encoding; learned positional encodings are another option.
- **Skip / residual connections:** Pass a layer’s input around a sublayer and add it to that sublayer’s output, helping preserve information and train deeper stacks. This expands a term listed in the lecture objectives.

**Study supplement:** Standard transformer blocks also use normalization. Autoregressive decoder self-attention uses a causal mask so a position cannot attend to future tokens during next-token prediction. Different transformer variants arrange these components differently.

Biomedical task connections include contextual clinical-text representations and image–report mappings. The lectures also present denoising, molecular representations, and reconstruction as encoder–decoder applications; these tasks do not inherently require a transformer.

## 9. Common confusions to avoid

| Confusion | Clarification |
|---|---|
| “Paired data always means similar data.” | Pairs can encode differences, matches, or interactions. |
| “A paired t-test compares two unrelated groups.” | It analyzes linked observations through within-pair differences. |
| “One-shot means one total training example.” | It means one support example per class in the new task; earlier training can be extensive. |
| “One-shot and 1-NN mean the same thing.” | Shot counts support examples per class; k counts neighbors used for prediction. |
| “Siamese branches learn independently.” | The lecture’s Siamese branches share encoder weights. |
| “Negative pairs must be pushed arbitrarily far apart.” | Margin-based loss stops penalizing them beyond the margin. |
| “Any autoencoder supports reliable random generation.” | Reconstruction training alone does not organize latent space for arbitrary sampling. |
| “A VAE just adds noise.” | It learns latent distributions and regularizes them toward a prior. |
| “Attention and transformers are interchangeable terms.” | Attention is a component used within transformer architectures and other models. |
| “Self versus cross is the same distinction as single versus multi-head.” | The former describes information sources; the latter describes parallel heads. |
