# Week 10 Lab Recap: Sequence Data, Generators, and Evaluation


## Big picture

This week, we moved from **representing biomedical sequences**, to **learning from and generating sequences**, to **evaluating whether generated samples are useful**.

---

## Sequence Data

### 1. What counts as biomedical sequence data?

A sequence is an ordered collection of observations. The index may represent **token/residue position**, **measurement time**, or **clinical event order**.

| Sequence type | Examples | What the index means |
|---|---|---|
| Symbolic sequences | Clinical notes, DNA, RNA, proteins | Token or residue position |
| Physiological time series | ECG, EEG, PPG, respiration | Measurement time |
| Longitudinal measurements | Glucose, blood pressure, lab results | Measurement time |
| Clinical event sequences | Diagnoses, visits, medications | Event order and time |
| Movement/image sequences | Gait, motion sensors, echo video | Time or frame number |

For time series, preserve **timestamps, units, and missing values**. Regularly sampled data and irregular clinical events need not be treated identically.

### 2. Representations versus descriptors

- **Representation:** A way of encoding information (e.g., clinical note, FASTA, SMILES, raw time series).
- **Descriptor/feature:** A numerical measurement derived from a representation (e.g., word counts, k-mer frequencies, molecular fingerprint bits, ECG shape features).
- **Key tradeoff:** Descriptors make modeling easier but may discard ordering, context, or structural information.

**Molecular representations:**

- **FASTA** represents biological sequences as one-letter residue strings, typically with a `>` header containing identifiers and metadata.
- **SMILES** represents a molecular graph as text. For aspirin: `CC(=O)Oc1ccccc1C(=O)O`. Symbols indicate atoms, bonds, branches, aromatic atoms, and ring closures. Different graph traversals can yield different valid strings for the same molecule.
- **Adjacency matrices** store pairwise bond information but require roughly **O(n²)** entries for *n* atoms, which is inefficient for sparse molecular graphs.

### 3. Bag-of-Words (BoW)

**BoW** converts each document into a vector of word counts over a fixed vocabulary.

**Process:** Define vocabulary → normalize/tokenize text → count terms → form a document–term matrix.

**Strengths:** Simple, interpretable, and useful for document classification and extracting features from clinical notes.

**Limitations:**

- Ignores word order, grammar, and much of the semantic context.
- Sensitive to the vocabulary; new terms can require updating feature definitions.
- Rare terms and common stopwords may create misleading or uninformative features.
- Cannot reliably distinguish context such as negation without additional features.

**Improvements:** Normalize case/punctuation, stem or lemmatize where appropriate, use **n-grams** to capture short phrases (e.g., “no fever”), consider stopwords carefully, and apply **TF-IDF**.

### 4. TF-IDF

**Term Frequency–Inverse Document Frequency** gives higher weight to words that are frequent in a document but relatively rare across the corpus.

One form used in lecture:

\[
w_{i,j}=\frac{tf_{i,j}}{\sum_{i'}tf_{i',j}}\log\left(\frac{N}{df_i}\right)
\]

where `tfᵢⱼ` is the count of term *i* in document *j*, `dfᵢ` is the number of documents containing term *i*, and *N* is the total number of documents. Other variants use smoothing or different scaling.

### 5. K-mers and sliding windows

A **k-mer** is a contiguous subsequence of length *k*, extracted by sliding a window one position at a time.

Example: `ACGTACGT`, with `k = 2`:

`AC, CG, GT, TA, AC, CG, GT`

- Number of windows for a sequence of length *L*: **L − k + 1** (when *L ≥ k*).
- Counts: `AC: 2, CG: 2, GT: 2, TA: 1`.
- Relative frequency of `AC`: **2/7**.
- K-mers capture **local patterns**, but counts lose absolute position and long-range ordering.

Related descriptor strategies include identifying molecular functional groups (prefer graph matching to naive SMILES substring matching) and extracting ECG waveform characteristics using sliding windows.

### 6. Why sequence order matters

Two glucose trajectories can contain exactly the same values but convey different information:

- A: `80 → 100 → 140 → 180` (rising)
- B: `180 → 140 → 100 → 80` (falling)

Both have mean **125 mg/dL**. A mean or bag of values loses the direction of change; an order-aware model can retain it.

**Takeaway:** Feature engineering is useful, but compressing sequences into counts or summary statistics can remove the very temporal or positional relationships that matter biologically.

---

## Generators

### 1. Why model sequences directly?

BoW and k-mer frequencies provide useful summaries but cannot fully encode long-range dependencies or complex positional relationships. Neural networks can learn from encoded characters or tokens directly.

- **Feed-forward networks:** Encode characters (e.g., one-hot vectors or embeddings), use fixed-length inputs with padding/truncation, and learn to predict attributes.
- **Independent word-level networks:** Process words separately and aggregate predictions, but do not naturally share contextual information between positions.
- **Recurrent networks:** Pass information between sequence positions so predictions depend on earlier context.

### 2. Recurrent Neural Networks (RNNs)

A feed-forward state depends on the current input, while an RNN hidden state also depends on its previous hidden state:

\[
h_t=f(h_{t-1},x_t)
\]

This recurrence allows an RNN to incorporate earlier observations when processing the current token/time point.

**Challenge — vanishing gradients:** During backpropagation through time, repeated multiplication of small derivatives can make gradients shrink substantially. This makes it difficult to learn dependencies across many time steps.

### 3. Long Short-Term Memory (LSTM)

An **LSTM** extends the RNN with mechanisms for managing information over time:

- **Forget gate:** Controls which previous information is retained or discarded.
- **Memory/input update:** Incorporates new information into the memory state.
- **Output gate:** Controls what information is passed into the output/hidden state.

**Why it matters:** LSTMs help mitigate vanishing-gradient difficulties and are better suited than simple RNNs for many long-range sequence relationships.

### 4. From prediction to generation

RNNs and LSTMs can support **supervised prediction** (e.g., process a sequence and feed its final representation to a prediction network) or **autoregressive generation**:

1. Start with a seed token/character.
2. Predict a distribution over the next token.
3. Select/sample a token.
4. Feed that token back into the model.
5. Repeat to generate a new sequence.

This is the basic next-token prediction idea behind language modeling. Biomedical examples include **antimicrobial peptide generation** and **small-molecule discovery**.

### 5. Generative Adversarial Networks (GANs)

A **GAN** trains two competing models:

- **Generator:** Produces synthetic samples intended to resemble real data.
- **Discriminator:** Attempts to distinguish generated samples from real samples.

As the discriminator improves, the generator receives pressure to create more convincing samples; both models learn through adversarial training.

**GAN training challenges:**

- **Cheating:** The generator exploits a weakness of the discriminator rather than learning the desired distribution.
- **Mode collapse:** The generator repeatedly produces the same or very similar samples, losing diversity.
- **Instability:** Optimizing two models with opposing objectives is difficult.

**Takeaway:** RNNs/LSTMs use sequential context for prediction and generation; GANs use adversarial feedback to train generators. Producing plausible samples is not the same as producing diverse, novel, and useful samples.

---

## Evaluating Generators

### 1. What makes a good generator?

There is **no single perfect metric**. Evaluation depends on whether the goal is realistic samples, diversity, novel discoveries, or a particular desired property.

Two central dimensions are:

- **Quality:** Are samples valid, realistic, and potentially useful?
- **Diversity:** Does the model produce a broad variety of samples rather than repeatedly reproducing a few?


### 2. Simple checks: validity, uniqueness, novelty

| Metric | Question | Example interpretation |
|---|---|---|
| **Validity** | Does the sample obey basic domain rules? | A peptide uses permitted amino acids; a molecule follows chemistry constraints. |
| **Uniqueness** | What fraction of generated samples are distinct from one another? | 75 distinct samples out of 100 generated → 75%. |
| **Novelty** | What fraction of generated samples are absent from the training set? | A generated sequence not seen during training counts as novel under exact matching. |

These checks are necessary but not sufficient: a valid, unique, novel molecule might still be useless.

### 3. More informative checks

- **Rule filters:** Check domain-specific constraints or alerts, such as peptide design alerts or PAINS filters for small molecules.
- **Internal diversity:** Measure *how different* generated samples are from each other, not merely whether they are exact duplicates.
- **Fragment overlap:** Check whether generated samples reuse substantial substructures or subsequences from training examples, even when the complete sample is technically novel.

### 4. Comparing distributions

If a generator aims to reproduce a training/reference distribution, compare generated samples with reference samples in either:

- **Property space:** Molecular weight, solubility, amino-acid composition, charge, hydrophobicity, etc.
- **Representation/design space:** Learned latent embeddings that reflect more complex sample characteristics.


#### Kullback–Leibler (KL) divergence

\[
D_{KL}(P\|Q)=\sum_x P(x)\log\frac{P(x)}{Q(x)}
\]

- Measures how much one probability distribution differs from another.
- **Asymmetric:** `KL(P || Q) ≠ KL(Q || P)`.
- Here, *P* can represent the reference distribution and *Q* an approximate/generated distribution.
- Smaller values indicate closer agreement under the chosen direction of comparison.

#### Fréchet-based distances

Embed real and generated samples using a pretrained neural network, then compare their **distributions in latent space**.

- **Fréchet Inception Distance (FID):** Used for generated images.
- **Fréchet ChemNet Distance (FCD):** Used for generated small molecules.
- **Interpretation:** Lower distance means the generated feature distribution more closely resembles the reference feature distribution.

### 5. Inception Score (IS)

The **Inception Score** uses a pretrained image classifier to assess two properties of generated images:

1. **Confident individual predictions:** Each image ideally produces a concentrated class distribution `p(y | x)` (a recognizable sample).
2. **Diverse predictions across images:** The aggregate/marginal class distribution `p(y)` should cover multiple classes (a varied collection).

A common mathematical definition is:

\[
IS=\exp\left(\mathbb{E}_{x\sim p_g}\left[D_{KL}\big(p(y\mid x)\|p(y)\big)\right]\right),
\qquad p(y)=\mathbb{E}_{x\sim p_g}[p(y\mid x)].
\]

**Important distinction:** `p(y | x)` is the classifier's label distribution for **one image**, whereas `p(y)` is the **marginal distribution averaged over generated images**. Higher IS generally reflects greater classifier confidence together with greater predicted class diversity. The score depends on the pretrained classifier and does not directly compare generated samples against real ones.

### 6. Goal-directed generation

Sometimes the goal is **not** to reproduce the entire training distribution, but to bias generation toward a desired property or structure.

- Example from lecture: Optimize generated molecules toward **higher or lower predicted melting temperature**.
- Evaluate whether the property distribution shifts in the intended direction, while also checking validity, novelty, and diversity.
- A high predicted objective score alone does not establish experimental success.

### 7. Rediscovery

**Rediscovery** asks whether a generator can produce **known useful samples deliberately excluded from training**.

- Hide selected known samples during training.
- Train the generator on the remaining data.
- Generate/rank candidates and check whether the withheld samples reappear.
- **Time-based or group-based splits** can provide a more informative test of prospective discovery ability than random withholding alone.

Rediscovery rates can remain very low, making this a challenging but meaningful evaluation.

**Takeaway:** Evaluate generators on multiple complementary dimensions. Distributional similarity, novelty, goal achievement, and rediscovery each answer different questions.

---

## Across-lecture connections

| Core question | Lecture 25 | Lecture 26 | Lecture 27 |
|---|---|---|---|
| How is information encoded? | FASTA, SMILES, time series, BoW, k-mers | Token embeddings and learned hidden states | Sample properties and learned latent embeddings |
| How is order/context handled? | Count-based features often discard it | RNN/LSTM explicitly carry sequential context | Evaluate whether generated samples retain meaningful patterns |
| What is the objective? | Build useful representations/features | Predict or generate sequences | Assess realism, diversity, novelty, and utility |
| What can go wrong? | Lost order/meaning; poor feature choice | Vanishing gradients, GAN cheating/mode collapse | Valid but useless samples; memorization; low diversity |

### Key distinctions to remember

- **Representation vs. descriptor:** Raw/structured encoding versus derived numerical feature.
- **BoW vs. k-mers:** Unordered word counts versus counts of contiguous subsequences; both can lose longer-range order.
- **RNN vs. LSTM:** Basic recurrent state versus gated memory for better handling longer dependencies.
- **RNN/LSTM vs. GAN:** Sequential modeling architectures versus an adversarial generator–discriminator training framework.
- **Uniqueness vs. novelty:** Different from *other generated samples* versus different from *training samples*.
- **Internal diversity vs. uniqueness:** Degree of dissimilarity versus exact nonduplication.
- **KL divergence vs. Fréchet-based metrics:** Comparison of probability distributions versus comparison of distributions in pretrained feature space.
- **Inception Score vs. FID:** Classifier confidence/class diversity for generated images versus distance between generated and real feature distributions.
- **Distribution matching vs. goal-directed evaluation:** Reproduce reference characteristics versus intentionally optimize a chosen property.

## Quick self-check

1. Why do two glucose sequences with the same mean potentially require different predictions?
2. What does a k-mer representation retain, and what does it lose?
3. Why can BoW fail to distinguish “fever” from “no fever” without n-grams or other context?
4. What information does an RNN hidden state carry that an independent feed-forward network does not?
5. Why do LSTMs help with long-term dependencies?
6. How does an autoregressive generator produce a new sequence?
7. What roles do the generator and discriminator play in a GAN? What is mode collapse?
8. Can a generated molecule be valid and unique but not novel? Explain.
9. Why should generator evaluation include both quality and diversity?
10. What is the difference between `p(y | x)` and `p(y)` in Inception Score?
11. When would goal-directed evaluation be more appropriate than matching the training distribution?
12. Why can rediscovery with a time-based holdout be informative for prospective biomedical discovery?

---
