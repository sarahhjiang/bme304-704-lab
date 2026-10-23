# Week 9 Lab Recap: Graphs, Neural Networks, and Optimization

## The big picture

This week connects three questions:

1. **How can we represent relationships?** Graphs store entities and the connections between them.
2. **How can we learn a nonlinear prediction function?** Neural networks combine weighted inputs with nonlinear activations.
3. **How do we find and evaluate a useful model?** Backpropagation computes gradients, an optimizer updates parameters, and validation guides model selection.

The progression is:

```text
Represent relationships → build a prediction function → fit parameters
                                                        ↓
                           evaluate on new data ← select model settings
```

## 1. Graphs: relationships become data

A graph is a pair

$$
G=(V,E),
$$

where $V$ is a set of **vertices/nodes** and $E$ is a set of **edges**. A graph is different from a plot: a plot usually encodes values through coordinates; a graph encodes relationships through connections. Moving a node in a drawing does not change its connections.

### Basic vocabulary

- **Order:** number of nodes, $|V|$.
- **Size:** number of edges, $|E|$.
- **Degree:** number of edges incident to a node in a simple undirected graph.
- **Directed graph:** an edge from $u$ to $v$ need not imply an edge from $v$ to $u$; distinguish in-degree and out-degree.
- **Undirected graph:** an edge represents a symmetric connection.
- **Node attributes/colors:** encode properties such as atom type or disease category.
- **Edge weights:** encode quantities such as distance, cost, bond order, or interaction strength.

The meaning of a weight matters. A large interaction strength might mean “closely related,” whereas a large path cost means “far apart.” Shortest-path algorithms need a cost/distance interpretation; they do not automatically convert strength into distance.

## 2. Storing a graph

For a simple unweighted graph, the **adjacency matrix** has entries

$$
A_{ij}=\begin{cases}1,&\text{if an edge connects }i\text{ to }j,\\0,&\text{otherwise.}\end{cases}
$$

For an undirected graph, $A=A^T$. Without self-loops, the diagonal is zero. In this setting, a row sum gives a node's degree, and the sum of all entries is $2|E|$ because each edge appears twice.

An **adjacency list** stores the neighbors of each node. 

| Representation | Useful property | Main tradeoff |
|---|---|---|
| Dense adjacency matrix | Direct edge lookup and convenient matrix operations | Uses $O(\lvert V\rvert^2)$ space, even when most entries are zero |
| Adjacency list | Stores only existing neighbors; useful for sparse networks | Edge lookup depends on the neighbor data structure |

An adjacency list uses $O(|V|+|E|)$ space. Sparse matrices are another option for large sparse graphs.

## 3. Centrality: what does “important” mean?

Different centrality measures answer different questions. 

### Degree

Degree counts immediate connections. Normalized degree centrality divides by the largest possible degree, $n-1$, in a simple graph with $n$ nodes.

A highly connected protein may be worth investigating, but degree alone does not establish biological importance; it can also reflect which proteins have been studied most often.

### Closeness

For a connected, unweighted, undirected graph:

$$
C_{\mathrm{close}}(v)=\frac{n-1}{\sum_{u\ne v}d(v,u)},
$$

where $d(v,u)$ is the shortest-path distance. This is the inverse of the **average** distance to the other nodes. Larger values indicate that a node can reach others in fewer steps.

For node $D$ above, the distances to $(A,B,C,E)$ are $(1,1,2,1)$:

$$
C_{\mathrm{close}}(D)=\frac{4}{1+1+2+1}=0.8.
$$

This normalized convention matches the numerical example in Lecture 22. Some definitions instead use the inverse of the sum, so state your convention. Disconnected graphs require a modified definition or a component-specific calculation because some distances are infinite.

### PageRank: conceptual comparison

PageRank considers the importance of the nodes supplying connections, not just how many connections exist. It can be understood through a random walk with occasional restarts. Degree asks “how many neighbors?”; PageRank also asks “which neighbors?” The lecture introduces it as another centrality measure; hand calculations here focus on degree and closeness.

**Analogy help you to remember:** 

Think about you pubished a paper.  
**Degree** would be your citation number.  
**Closness** is about how connected you are (to other fields).  
**PageRank** is who (the big names) cites you.

## 4. Shortest paths and Dijkstra's algorithm

A path is a sequence of connected nodes. In an unweighted graph, path length is the number of edges. In a weighted graph, path cost is the sum of edge costs.

Dijkstra's algorithm:

1. Set the source distance to 0 and all others to infinity.
2. Select the unvisited node with the smallest tentative distance.
3. Try improving each neighbor's distance using the route through that node.
4. Mark the selected node visited and repeat until the target is settled or no reachable unvisited nodes remain.

The update, or **relaxation**, is

$$
d(v)\leftarrow\min\{d(v),d(u)+w(u,v)\}.
$$

Dijkstra requires **nonnegative edge costs**. For unweighted graphs, breadth-first search is sufficient.

## 5. Link prediction with Jaccard similarity

An incomplete network may be missing real connections. **Link prediction**, or edge imputation, ranks candidate connections for further investigation. 

Let $N(u)$ be the set of neighbors of $u$. For two nodes:

$$
J(u,v)=\frac{|N(u)\cap N(v)|}{|N(u)\cup N(v)|}.
$$

It compares shared neighbors with all distinct neighbors. A larger score suggests stronger neighborhood overlap, not a calibrated probability that the edge exists.

## 6. From a perceptron to a neural network

A perceptron first calculates a weighted sum and then applies a threshold:

$$
z=\mathbf{w}^T\mathbf{x}+b,\qquad
\hat y=\begin{cases}1,&z\ge0,\\0,&z<0.\end{cases}
$$

The weights control contributions from the inputs; the bias shifts the boundary. A single perceptron has a linear decision boundary. It can represent OR, but not XOR. 

For OR, take $w_1=w_2=1$ and $b=-0.5$. Only $(0,0)$ produces a negative weighted sum.

For XOR, the labels are:

| $x_1$ | $x_2$ | OR | XOR |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 |
| 0 | 1 | 1 | 1 |
| 1 | 0 | 1 | 1 |
| 1 | 1 | 1 | 0 |

No straight line separates the two XOR classes. A hidden layer can construct a representation where separation is possible.

## 7. Activations, layers, and parameters

A typical layer computes

$$
\mathbf{z}^{(\ell)}=W^{(\ell)}\mathbf{a}^{(\ell-1)}+\mathbf{b}^{(\ell)},
\qquad
\mathbf{a}^{(\ell)}=f(\mathbf{z}^{(\ell)}),
$$

with $\mathbf{a}^{(0)}=\mathbf{x}$. The activation is applied elementwise. 

| Activation | Formula | Useful property / limitation |
|---|---|---|
| Sigmoid | $\sigma(z)=1/(1+e^{-z})$ | Output in $(0,1)$; can saturate and produce small gradients |
| ReLU | $\max(0,z)$ | Simple nonlinearity; zero derivative for negative inputs |
| Linear | $f(z)=z$ | Useful for an unconstrained regression output |

For binary classification, a sigmoid output is commonly paired with binary cross-entropy. Multiple mutually exclusive classes commonly use softmax and multiclass cross-entropy. The output activation and loss should match the task.

**Why nonlinearity matters:** Two affine layers without a nonlinear activation collapse into one:

$$
W_2(W_1x+b_1)+b_2=(W_2W_1)x+(W_2b_1+b_2).
$$

Adding layers alone does not create a nonlinear function.

### Count the parameters

A fully connected layer with $p$ inputs and $h$ outputs has $ph$ weights and $h$ biases. A network with widths $n_0,n_1,\ldots,n_L$ has

$$
\sum_{\ell=1}^{L}(n_{\ell-1}+1)n_\ell
$$

trainable parameters when every non-input neuron has one bias.

**Practice:** A $3\to4\to2\to1$ network has

$$
(3+1)4+(4+1)2+(2+1)1=16+10+3=29
$$

parameters. Inputs do not have learned biases of their own. Activations are values produced by a particular input, not extra learned parameters.

## 8. Depth and universal approximation

A feedforward network passes information from inputs toward outputs without recurrent loops. The examples here are fully connected networks. Using the lecture's convention, a deep network has more than one hidden layer.

The **universal approximation theorem** says that a network with a sufficiently wide hidden layer, suitable nonlinear activations and biases, and a linear output can approximate continuous functions arbitrarily well on a closed, bounded input region.

It is an existence statement. It does not guarantee:

- that gradient descent finds the required weights;
- that the network is small or computationally efficient;
- that predictions generalize beyond the training examples.

Depth can represent some hierarchical functions more efficiently than a shallow network, but adding layers does not automatically reduce parameter count or improve performance. Count the parameters and evaluate the model.

## 9. Loss: define what training should improve

Training searches for weights and biases, collected in $\theta$, that minimize a loss:

$$
L(\theta)=\frac{1}{N}\sum_{i=1}^{N}\ell(y_i,\hat y_i(\theta)),
\qquad
\theta^*=\arg\min_\theta L(\theta).
$$

For regression, a common choice is mean squared error:

$$
L_{\mathrm{MSE}}=\frac{1}{N}\sum_i(\hat y_i-y_i)^2.
$$

For binary classification with predicted probabilities $p_i$, a common choice is binary cross-entropy:

$$
L_{\mathrm{BCE}}=-\frac{1}{N}\sum_i[y_i\log p_i+(1-y_i)\log(1-p_i)].
$$

**Loss and evaluation metric are related but distinct.** A model may train with cross-entropy and be evaluated with recall, balanced accuracy, or MCC. Metrics based on hard class predictions are generally unsuitable as direct differentiable objectives. In particular, the MCC mentioned in Lecture 23 should ordinarily be treated as an evaluation metric unless a differentiable surrogate is explicitly defined. 

## 10. Gradient descent, batches, and epochs

The gradient contains partial derivatives with respect to the model parameters. Gradient descent moves in the negative-gradient direction:

$$
\theta_{t+1}=\theta_t-\eta\nabla_\theta L(\theta_t).
$$

The learning rate $\eta$ controls the step size. A small value can make progress slow; a large value can overshoot or cause divergence. A neural network's loss is generally nonconvex, so convergence to a global minimum is not guaranteed. 

| Method | Examples used for one update | Typical tradeoff |
|---|---|---|
| Full-batch gradient descent | All training observations | Stable gradient, potentially expensive updates |
| Stochastic gradient descent, strict definition | One observation | Cheap, noisy updates |
| Mini-batch gradient descent | A subset, such as 32 observations | Compromise between cost and gradient variability |

An **epoch** is one complete pass through the training data. An **update/iteration** is one parameter update. “SGD” is also often used informally for mini-batch training.

Changing batch size changes both gradient noise and the number of updates per epoch. Equal epoch counts therefore do not imply equal numbers of updates.

## 11. Backpropagation: compute the gradient before updating

Backpropagation applies the chain rule backward through the computation to find how each parameter affects the loss. The optimizer then uses those derivatives to change the parameters. 

```text
Forward pass → prediction → loss
                             ↓
                   backward pass: gradients
                             ↓
                   optimizer: parameter update
```

## 12. Parameters versus hyperparameters

There are two levels of optimization. 

| Quantity | How it is chosen | Examples |
|---|---|---|
| Parameter $\theta$ | Learned during fitting on training data | Weights and biases |
| Hyperparameter $\lambda$ | Chosen as part of the model/training procedure, usually using validation | Layer widths, learning rate, batch size, regularization strength |

For a fixed set of hyperparameters, training finds a fitted parameter vector. Tuning compares the validation performance of models trained under different settings.

```text
Choose candidate hyperparameters
              ↓
Fit fresh model parameters using training data
              ↓
Score on validation data
              ↓
Compare candidates and select settings
              ↓
Refit under the selected training procedure
              ↓
Evaluate the frozen choice on the untouched test set
```

The number of training epochs is also a setting. If validation determines when to stop, the stopping rule participates in model selection.

## 13. Validation, cross-validation, and leakage

- **Training data** fit parameters.
- **Validation data** guide model choices, hyperparameters, and stopping rules.
- **Test data** evaluate the final frozen choice.

Repeatedly checking test performance while changing the model turns the test set into another validation set. 

### K-fold cross-validation for tuning

Keep test data aside. Within the remaining development data:

1. Divide observations into $K$ folds.
2. For each hyperparameter candidate, fit a fresh model on $K-1$ folds and score it on the remaining fold.
3. Repeat so each fold is used for validation once.
4. Compare mean validation scores and their variability across candidates.
5. Select settings, refit on the development data, and evaluate once on the test set.


### Preprocessing belongs inside the split

Scaling, feature selection, imputation, and other learned preprocessing must be fit on the training portion of each fold, then applied to its validation portion. Scaling the whole dataset before CV leaks information about validation observations.

For repeated patient measurements, use patient/group-aware splits when the goal is generalization to new patients. For future prediction, consider time-respecting splits. This extends the evaluation principles from Weeks 3–4.

### Nested cross-validation

Nested CV separates **selection** from **assessment**:

- The inner loop selects hyperparameters within each outer training split.
- The outer held-out fold evaluates that entire selection-and-training procedure.

Repeat tuning separately inside each outer split. Reporting the best score from the same CV used to select the winner can be optimistic. An independent untouched test set is another way to separate final assessment from tuning.

## 14. Searching the hyperparameter space

| Strategy | How it chooses candidates | Strength | Limitation |
|---|---|---|---|
| Grid search | Every combination in a predefined grid | Simple and systematic | Cost grows quickly with the number of settings |
| Random search | Randomly sampled configurations | Flexible trial budget; explores more values of important dimensions | May miss a narrow useful region |
| Bayesian optimization | Uses previous trials and a surrogate model to select the next candidate | Can use expensive trials efficiently | Adds overhead and depends on the surrogate/search design |
| Genetic algorithm | Selects and modifies populations of configurations | Can explore mixed or irregular search spaces | Can require many evaluations and extra design choices |
| Hyperband | Allocates resources across candidates using repeated successive halving | Avoids fully training every weak candidate | Early performance may undervalue slow-learning models |

Bayesian optimization balances **exploitation** (try settings predicted to work well) and **exploration** (try uncertain regions). A Gaussian process is one possible surrogate, not a required one.

Genetic algorithms use selection, mutation, and recombination of candidate settings. They are an optimization analogy to natural selection, not neural-network backpropagation.

Hyperparameters can be discrete, such as the number of layers, and validation performance need not be a differentiable function of those settings. Ordinary backpropagation through the training loss is therefore not a general hyperparameter-search method.

## 15. Early stopping and resource allocation

If training loss keeps falling while validation loss starts rising, fitting longer may be making the model more specific to the training data. Early stopping monitors a validation criterion, allows a specified amount of patience, and retains the best validation checkpoint. It can reduce overfitting and computation, but does not guarantee good generalization. 

Do not choose the stopping epoch using the test set. If early stopping is used inside CV, its stopping data should come from the fold's training portion, leaving the fold's validation portion available for scoring the candidate.


## 16. High-yield distinctions

| Concept A | Concept B | Key difference |
|---|---|---|
| Graph | Plot | Relationships through edges vs. values through coordinates |
| Degree | Closeness | Immediate connections vs. distances to other nodes |
| Edge strength | Edge cost | Large may mean strong vs. large means expensive |
| Graph data | Neural network | Relations among entities vs. a trainable computation |
| Weight/bias | Activation | Learned parameter vs. input-dependent value |
| Forward pass | Backpropagation | Calculate prediction vs. calculate loss gradients |
| Backpropagation | Gradient descent | Compute derivatives vs. update parameters |
| Batch | Epoch | Examples used in one update vs. a full pass through training data |
| Parameter | Hyperparameter | Fitted value vs. model/training setting |
| Training loss | Validation performance | Fit to observed data vs. evidence for choosing a model |
| Early stopping | Hyperband | Stop a fit using validation vs. allocate budgets across candidates |
| Approximation capacity | Generalization | What can be represented vs. performance on unseen data |