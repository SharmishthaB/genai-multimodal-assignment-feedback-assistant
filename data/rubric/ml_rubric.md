---
rule_id: DT_001
topic: Decision Trees
subtopic: Entropy Calculation
keywords: entropy, H(S), -p log p, log base 2, impurity, class probability
criteria: >
  Student must compute entropy using H(S) = -Σ p_i * log₂(p_i) for each node,
  explicitly stating all class probabilities p_i and using log base 2.
points: 2
common_error: Using natural log or log base 10 instead of log base 2; omitting one class.
socratic_hint: >
  "Your entropy formula structure looks right. In information theory, what base should
  the logarithm use? Also, you have {n} class labels in this set — have you included all of them?"
---
rule_id: DT_002
topic: Decision Trees
subtopic: Information Gain Calculation
keywords: information gain, IG, H(S) - Σ weighted entropy, split, attribute
criteria: >
  Student must compute IG(S, A) = H(S) - Σ (|S_v|/|S|) * H(S_v) for each candidate
  attribute, showing the weighted sum of child node entropies.
points: 3
common_error: Forgetting to weight child entropy by fraction of examples; computing entropy of full set only.
socratic_hint: >
  "You computed H(S) correctly. Now for each branch after splitting on attribute A,
  how do you combine the child entropies? What weight does each child branch get?"
---
rule_id: DT_003
topic: Decision Trees
subtopic: Attribute Selection
keywords: best split, highest information gain, choose attribute, root node
criteria: >
  Student must select the attribute with the HIGHEST information gain as the split node,
  and explicitly state this selection criterion.
points: 1
common_error: Selecting the attribute with the lowest entropy (not highest IG) or failing to compare all attributes.
socratic_hint: >
  "You've calculated IG for each attribute. Which one should you pick as the root — the
  one that tells you the most or the least about the class label?"
---
rule_id: DT_004
topic: Decision Trees
subtopic: Gini Impurity (alternative)
keywords: gini, 1 - Σ p_i², gini index, CART
criteria: >
  If Gini is used: Gini(S) = 1 - Σ p_i². Student must square each class probability
  and subtract from 1. Gini ranges [0, 0.5] for binary; [0, 1-1/k] for k classes.
points: 2
common_error: Writing Gini = Σ p_i² (forgetting the 1 - part); confusing Gini with entropy.
socratic_hint: >
  "Your Gini calculation is missing a step. Gini impurity measures the probability of
  mislabeling — what should the full expression look like starting from 1?"
---  
rule_id: DT_005
topic: Decision Trees
subtopic: Tree Construction (recursive partitioning)
keywords: recursive, branch, leaf node, stopping criterion, pure node, max depth
criteria: >
  Student must show the recursive process: select best attribute → split dataset →
  recurse on each subset → stop when node is pure (all same class) or no attributes remain.
points: 2
common_error: Not applying recursion; building only one level of the tree.
socratic_hint: >
  "You've built the root split correctly. Now, for each branch of this split,
  what do you do with the remaining training examples and remaining attributes?"
---
rule_id: DT_006
topic: Decision Trees
subtopic: Handling Continuous Attributes
keywords: threshold, continuous, split point, binary split, midpoint
criteria: >
  For continuous attributes, student must identify candidate thresholds (midpoints between
  sorted unique values), compute IG for each threshold, and select the optimal one.
points: 2
common_error: Treating a continuous attribute as categorical; not testing multiple thresholds.
socratic_hint: >
  "The attribute here is continuous. How do we find a split point for continuous values?
  If the sorted values are [2, 5, 7, 9], what candidate thresholds would you test?"
---
rule_id: DT_007
topic: Decision Trees
subtopic: Overfitting and Pruning
keywords: overfitting, pruning, reduced error pruning, validation set, depth
criteria: >
  Student must acknowledge that deep trees overfit; pruning removes branches that do
  not improve accuracy on a validation set; reduced error pruning replaces a subtree
  with a leaf if it does not reduce accuracy.
points: 2
common_error: Conflating training accuracy with generalization; not defining pruning criterion.
socratic_hint: >
  "Your tree perfectly classifies all training examples. Is that always good?
  What happens to a new example that wasn't in training — why might we want to simplify the tree?"
---
rule_id: DT_008
topic: Decision Trees
subtopic: Multi-class vs Binary
keywords: multi-class, binary, one-vs-rest, class labels
criteria: >
  For multi-class problems, entropy/Gini extends naturally over k classes.
  Student must use all k classes in entropy computation, not just binary (yes/no).
points: 1
common_error: Using binary entropy formula H = -p*log(p) - (1-p)*log(1-p) for a 3-class problem.
socratic_hint: >
  "Your entropy formula works for 2 classes. This problem has {k} classes.
  How does the entropy formula generalize when there are more than 2 class labels?"
---
rule_id: FP_001
topic: Forward Propagation
subtopic: Linear Combination (Pre-activation)
keywords: z, weighted sum, bias, z = Wx + b, pre-activation, net input
criteria: >
  Student must compute z = W·x + b for each neuron, showing the dot product
  of weights and inputs, plus the bias term explicitly.
points: 2
common_error: Omitting the bias term; using addition instead of dot product.
socratic_hint: >
  "You've multiplied weights by inputs correctly. Every neuron also has an
  additional term that shifts the activation — what is it called, and where does it appear in z?"
---
rule_id: FP_002
topic: Forward Propagation
subtopic: Activation Function Application
keywords: sigmoid, ReLU, tanh, activation, a = f(z), non-linearity
criteria: >
  Student must apply the activation function a = f(z) after computing z,
  specifying which function is used (e.g., sigmoid: 1/(1+e^{-z})).
points: 2
common_error: Forgetting to apply activation; applying activation before the bias addition.
socratic_hint: >
  "You have z computed. In a neural network, what happens to z before it becomes
  the output of that neuron? Which activation function are you using here?"
---
rule_id: FP_003
topic: Forward Propagation
subtopic: Layer-wise Propagation
keywords: layer 1, layer 2, hidden layer, output layer, matrix multiply
criteria: >
  For multi-layer networks, student must propagate layer by layer:
  a^[l] = f(W^[l] · a^[l-1] + b^[l]), showing each layer's computation sequentially.
points: 3
common_error: Jumping directly from input to output without showing hidden layer computations.
socratic_hint: >
  "You computed the output layer directly. In a 2-hidden-layer network, what are the
  intermediate values a^[1] and a^[2] — can you show each layer step by step?"
---
rule_id: FP_004
topic: Forward Propagation
subtopic: Softmax Output Layer
keywords: softmax, e^z_i / Σ e^z_j, probability distribution, output layer, multi-class
criteria: >
  For multi-class classification, student must apply softmax:
  ŷ_i = e^{z_i} / Σ_j e^{z_j}, and verify that outputs sum to 1.
points: 2
common_error: Using sigmoid instead of softmax for multi-class output; not normalizing.
socratic_hint: >
  "You have {k} class outputs. The output layer for multi-class classification should
  give probabilities that sum to 1. Which function achieves that — and what does its formula look like?"
---
rule_id: FP_005
topic: Forward Propagation
subtopic: Loss Computation
keywords: cross-entropy loss, MSE, L = -Σ y log ŷ, loss function, prediction error
criteria: >
  Student must compute the loss L using the appropriate loss function
  (cross-entropy for classification: L = -Σ y_i log(ŷ_i); MSE for regression).
points: 2
common_error: Using MSE for a classification problem or vice versa.
socratic_hint: >
  "You've computed ŷ. Now, this is a classification task — which loss function
  is appropriate here, and how does it compare the prediction to the true label y?"
---
rule_id: FP_006
topic: Forward Propagation
subtopic: Numerical Computation on Given Network
keywords: compute, numerical, plug in, weights given, calculate output
criteria: >
  When weights and inputs are given numerically, student must show each arithmetic
  step: individual multiplications, summation, and activation output values.
points: 3
common_error: Skipping intermediate arithmetic; rounding too early; wrong order of operations.
socratic_hint: >
  "Let me see your step-by-step computation for node h1. What is w_{11}*x_1, then w_{21}*x_2?
  What is their sum plus the bias before applying the activation?"
---
rule_id: BP_001
topic: Backpropagation
subtopic: Chain Rule — Statement
keywords: chain rule, dL/dW, partial derivative, composite function, ∂L/∂W
criteria: >
  Student must explicitly invoke the chain rule before any gradient computation:
  ∂L/∂W^[l] = ∂L/∂a^[l] · ∂a^[l]/∂z^[l] · ∂z^[l]/∂W^[l]
points: 2
common_error: Computing gradient directly without citing chain rule; skipping intermediate terms.
socratic_hint: >
  "Before you compute ∂L/∂W, let's think about the chain. The loss L depends on ŷ,
  which depends on z, which depends on W. Can you write out each link of that chain separately?"
---
rule_id: BP_002
topic: Backpropagation
subtopic: Output Layer Gradient (δ^[L])
keywords: delta, error signal, output layer, dL/dz, δ^L = ŷ - y, softmax gradient
criteria: >
  For cross-entropy + softmax: δ^[L] = ŷ - y.
  Student must correctly compute this output error term before propagating backward.
points: 2
common_error: Using ŷ - y for the wrong layer; confusing δ with ∂L/∂a.
socratic_hint: >
  "For a softmax + cross-entropy combination, there is a convenient simplification
  for the gradient at the output layer. What does ∂L/∂z^[L] simplify to?"
---
rule_id: BP_003
topic: Backpropagation
subtopic: Hidden Layer Gradient (δ^[l])
keywords: backpropagate, hidden layer delta, W^T, δ^[l] = (W^[l+1])^T δ^[l+1] * f'(z^[l])
criteria: >
  δ^[l] = (W^[l+1])^T · δ^[l+1] * f'(z^[l]).
  Student must transpose the weight matrix and element-wise multiply by activation derivative.
points: 3
common_error: Forgetting the transpose; forgetting to multiply by f'(z); using wrong layer index.
socratic_hint: >
  "You have δ^[l+1]. To get δ^[l], you need to 'send back' the error through the weights.
  How do you mathematically reverse the direction of the forward pass through W^[l+1]?"
---
rule_id: BP_004
topic: Backpropagation
subtopic: Activation Derivative
keywords: sigmoid derivative, σ'(z), σ(1-σ), ReLU derivative, tanh', f'(z)
criteria: >
  Student must compute f'(z) for the activation used:
  Sigmoid: σ'(z) = σ(z)(1 - σ(z)); ReLU: f'(z) = 1 if z>0 else 0; tanh: 1 - tanh²(z).
points: 2
common_error: Using f'(z) = σ(z) without the (1-σ) term; applying wrong activation derivative.
socratic_hint: >
  "You're using sigmoid activation. When you differentiate σ(z) = 1/(1+e^{-z}),
  what does σ'(z) equal? Can you express it in terms of σ(z) itself?"
---
rule_id: BP_005
topic: Backpropagation
subtopic: Weight Gradient Computation
keywords: ∂L/∂W, outer product, δ · a^T, weight update gradient
criteria: >
  ∂L/∂W^[l] = δ^[l] · (a^[l-1])^T.
  Student must use the outer product of the error signal and the previous layer's activation.
points: 3
common_error: Forgetting the transpose on a^[l-1]; using element-wise instead of outer product.
socratic_hint: >
  "You have δ^[l] and a^[l-1]. The gradient of L w.r.t. W^[l] involves multiplying these.
  What is the correct matrix operation, and what shape should ∂L/∂W^[l] have?"
---
rule_id: BP_006
topic: Backpropagation
subtopic: Bias Gradient
keywords: ∂L/∂b, bias gradient, sum over batch, δ^[l]
criteria: >
  ∂L/∂b^[l] = δ^[l] (for a single example).
  For a batch: sum over training examples. Student must compute bias gradient separately.
points: 1
common_error: Omitting the bias gradient entirely; using the same formula as weight gradient.
socratic_hint: >
  "You've computed ∂L/∂W. Don't forget the bias — how does L change with respect to b^[l]?
  Look at z^[l] = W^[l]a^[l-1] + b^[l] — what is ∂z^[l]/∂b^[l]?"
---
rule_id: BP_007
topic: Backpropagation
subtopic: Weight Update Rule
keywords: gradient descent, W = W - α * ∂L/∂W, learning rate, update
criteria: >
  After computing gradients, student must apply: W^[l] ← W^[l] - α * ∂L/∂W^[l]
  and b^[l] ← b^[l] - α * ∂L/∂b^[l], with learning rate α explicitly stated.
points: 2
common_error: Adding instead of subtracting gradients; omitting learning rate.
socratic_hint: >
  "You have the gradient. To minimize the loss, in which direction should you move the weights?
  How does the learning rate α control the size of this update?"
---
rule_id: BP_008
topic: Backpropagation
subtopic: Vanishing Gradient Problem
keywords: vanishing gradient, deep network, sigmoid, small gradients, deep layers
criteria: >
  Student must explain that in deep networks with sigmoid/tanh, gradients become
  exponentially small as they propagate backward, causing early layers to learn very slowly.
points: 2
common_error: Confusing vanishing with exploding gradients; not connecting to activation function.
socratic_hint: >
  "When you multiply many sigmoid derivatives together during backprop, what happens to
  the magnitude of the gradient? Why might the first layers in a deep network receive tiny updates?"
---
rule_id: SVM_H_001
topic: SVM - Hard Margin
subtopic: Decision Boundary Definition
keywords: w·x + b = 0, hyperplane, margin, separating hyperplane, decision boundary
criteria: >
  Student must define the decision boundary as w·x + b = 0 and the two margin
  hyperplanes as w·x + b = +1 (positive class) and w·x + b = -1 (negative class).
points: 2
common_error: Using w·x = 0 without the bias; not defining both margin hyperplanes.
socratic_hint: >
  "You've written the decision boundary. A hard-margin SVM also defines two parallel
  planes through the support vectors — what are those two equations?"
---
rule_id: SVM_H_002
topic: SVM - Hard Margin
subtopic: Margin Width
keywords: margin, 2/||w||, maximize margin, width, geometric margin
criteria: >
  Student must derive that the margin = 2/||w|| and state that SVM maximizes
  this margin, equivalently minimizing (1/2)||w||².
points: 2
common_error: Writing margin = 1/||w|| (off by factor of 2); not connecting to optimization objective.
socratic_hint: >
  "The two margin planes are w·x+b=+1 and w·x+b=-1. What is the geometric distance
  between two parallel planes of the form w·x+b=c₁ and w·x+b=c₂?"
---
rule_id: SVM_H_003
topic: SVM - Hard Margin
subtopic: Primal Optimization Problem
keywords: primal, minimize 1/2 ||w||², subject to, constraint, y_i(w·x_i + b) ≥ 1
criteria: >
  Student must write the primal problem: minimize (1/2)||w||²
  subject to y_i(w·x_i + b) ≥ 1 for all i.
points: 3
common_error: Wrong constraint direction (≤ instead of ≥); missing the 1/2 coefficient; forgetting y_i.
socratic_hint: >
  "Good attempt at the optimization. The constraint ensures all points are on the
  correct side of their respective margin planes. For a positive example (y_i=+1),
  what must w·x_i + b be at minimum?"
---
rule_id: SVM_H_004
topic: SVM - Hard Margin
subtopic: Lagrangian Formulation
keywords: Lagrangian, L(w,b,α), KKT, dual, αᵢ ≥ 0, Lagrange multiplier
criteria: >
  Student must write the Lagrangian: L = (1/2)||w||² - Σαᵢ[yᵢ(w·xᵢ+b) - 1]
  where αᵢ ≥ 0 are Lagrange multipliers.
points: 3
common_error: Wrong sign in Lagrangian; missing the subtraction of constraint; not stating αᵢ ≥ 0.
socratic_hint: >
  "To convert a constrained optimization to unconstrained, we introduce multipliers.
  For an inequality constraint g(x) ≥ 0 we subtract αg(x) from the objective.
  How does that look applied to our SVM constraint?"
---
rule_id: SVM_H_005
topic: SVM - Hard Margin
subtopic: KKT Conditions
keywords: KKT, stationarity, complementary slackness, αᵢ(yᵢ(w·xᵢ+b)-1)=0, ∂L/∂w = 0
criteria: >
  Student must derive: (1) ∂L/∂w = 0 → w = Σαᵢyᵢxᵢ;
  (2) ∂L/∂b = 0 → Σαᵢyᵢ = 0;
  (3) complementary slackness: αᵢ[yᵢ(w·xᵢ+b)-1] = 0.
points: 3
common_error: Not deriving w = Σαᵢyᵢxᵢ; skipping complementary slackness.
socratic_hint: >
  "Take the partial derivative of L with respect to w and set it to zero.
  This gives you w as a sum — over which points, and with what coefficients?"
---
rule_id: SVM_H_006
topic: SVM - Hard Margin
subtopic: Support Vectors
keywords: support vectors, αᵢ > 0, margin boundary, closest points
criteria: >
  Student must identify support vectors as the training points where αᵢ > 0,
  i.e., points lying exactly on the margin hyperplanes w·x+b = ±1.
points: 2
common_error: Calling all training points support vectors; not connecting αᵢ > 0 to the margin plane.
socratic_hint: >
  "From complementary slackness: αᵢ[yᵢ(w·xᵢ+b)-1]=0. If αᵢ > 0, what can we conclude
  about the term in brackets? Which points satisfy this — and what do we call them?"
---
rule_id: SVM_H_007
topic: SVM - Hard Margin
subtopic: Dual Optimization Problem
keywords: dual, maximize, W(α) = Σαᵢ - (1/2)ΣΣαᵢαⱼyᵢyⱼxᵢ·xⱼ, kernel trick
criteria: >
  Student must write the dual: maximize Σαᵢ - (1/2)ΣᵢΣⱼ αᵢαⱼyᵢyⱼ(xᵢ·xⱼ)
  subject to αᵢ ≥ 0 and Σαᵢyᵢ = 0.
points: 3
common_error: Wrong sign in dual objective; missing the dot product term; not stating constraints.
socratic_hint: >
  "After substituting w = Σαᵢyᵢxᵢ back into the Lagrangian and simplifying,
  what dual objective do you get? The key insight: the primal variable w disappears."
---
rule_id: SVM_S_001
topic: SVM - Soft Margin
subtopic: Slack Variables
keywords: slack variable, ξᵢ, ξᵢ ≥ 0, allow misclassification, margin violation
criteria: >
  Student must introduce slack variables ξᵢ ≥ 0 that allow points to violate the
  margin: yᵢ(w·xᵢ+b) ≥ 1 - ξᵢ.
points: 2
common_error: Not defining ξᵢ ≥ 0; using wrong constraint with slack.
socratic_hint: >
  "In the real world, data isn't always linearly separable. What do we add to the
  hard-margin constraint yᵢ(w·xᵢ+b) ≥ 1 to allow some tolerance for misclassification?"
---
rule_id: SVM_S_002
topic: SVM - Soft Margin
subtopic: Modified Primal Objective
keywords: C parameter, penalty, minimize (1/2)||w||² + C*Σξᵢ, regularization
criteria: >
  Student must write: minimize (1/2)||w||² + C·Σξᵢ subject to yᵢ(w·xᵢ+b) ≥ 1 - ξᵢ, ξᵢ ≥ 0.
  C controls the bias-variance tradeoff.
points: 3
common_error: Missing the CΣξᵢ penalty; not stating ξᵢ ≥ 0 as a second constraint.
socratic_hint: >
  "The soft-margin adds a cost to slack. If C is very large, what does the model prioritize —
  minimizing margin violations or maximizing margin? What about small C?"
---
rule_id: SVM_S_003
topic: SVM - Soft Margin
subtopic: Effect of C
keywords: C large, C small, overfitting, underfitting, hard margin, wide margin
criteria: >
  Large C → penalizes misclassification heavily → small margin, low bias, high variance (like hard SVM).
  Small C → tolerates misclassification → wide margin, high bias, low variance.
points: 2
common_error: Reversing the effect; not connecting C to bias-variance tradeoff.
socratic_hint: >
  "Think about extreme cases. If C→∞, what happens to our tolerance for misclassified points?
  If C→0, what does the model care about most?"
---
rule_id: NB_001
topic: Naive Bayes
subtopic: Bayes Theorem Application
keywords: posterior, P(y|x), P(x|y)*P(y)/P(x), Bayes rule, likelihood, prior
criteria: >
  Student must write P(y|x) = P(x|y)·P(y) / P(x) and identify each term:
  posterior P(y|x), likelihood P(x|y), prior P(y), evidence P(x).
points: 2
common_error: Inverting numerator and denominator; not labeling terms.
socratic_hint: >
  "You've written Bayes' theorem. Can you label each component — which term is the
  likelihood, which is the prior, and which is the posterior?"
---
rule_id: NB_002
topic: Naive Bayes
subtopic: Conditional Independence Assumption
keywords: naive, independence, P(x|y) = Π P(xᵢ|y), feature independence, assumption
criteria: >
  Student must state the "naive" assumption: features x₁,…,xₙ are conditionally
  independent given y, so P(x|y) = Π P(xᵢ|y).
points: 2
common_error: Computing joint probability without factoring; not stating the independence assumption.
socratic_hint: >
  "Naive Bayes is called 'naive' because of a simplifying assumption about the features.
  What assumption allows us to replace P(x₁,x₂,…,xₙ|y) with a product of individual terms?"
---
rule_id: NB_003
topic: Naive Bayes
subtopic: Parameter Estimation (MLE)
keywords: MLE, count, frequency, P(xᵢ=v|y=c) = count(xᵢ=v, y=c) / count(y=c)
criteria: >
  Student must estimate P(xᵢ=v|y=c) = #(xᵢ=v AND y=c) / #(y=c) from training data.
  For Gaussian NB: estimate mean μ and variance σ² for each feature per class.
points: 2
common_error: Using total dataset count in denominator instead of class count.
socratic_hint: >
  "To estimate P(feature=sunny | class=play), you need to count. Out of all examples
  where class=play, how many have feature=sunny? That fraction is your estimate."
---
rule_id: NB_004
topic: Naive Bayes
subtopic: MAP Classification Decision
keywords: MAP, argmax, predict class, maximize posterior, decision rule, argmax_y
criteria: >
  Predicted class = argmax_y [P(y) · Π P(xᵢ|y)].
  Student must compute this product for each class and pick the maximum.
points: 3
common_error: Forgetting to multiply the prior P(y); normalizing by P(x) (unnecessary for classification).
socratic_hint: >
  "You've computed the likelihood for each class. Don't forget the class prior —
  how does P(y) change your final answer? Do you need to compute P(x) to make the decision?"
---
rule_id: NB_005
topic: Naive Bayes
subtopic: Log-Sum Trick (Numerical Stability)
keywords: log, log probability, sum of logs, underflow, log P(y) + Σlog P(xᵢ|y)
criteria: >
  Student should use log probabilities: log P(y|x) ∝ log P(y) + Σ log P(xᵢ|y)
  to avoid numerical underflow from multiplying many small probabilities.
points: 1
common_error: Multiplying raw probabilities that result in underflow (essentially 0).
socratic_hint: >
  "If each P(xᵢ|y) is 0.01 and there are 10 features, what does their product equal?
  How can the logarithm help you avoid this numerical problem?"
---
rule_id: NB_006
topic: Naive Bayes
subtopic: Laplace Smoothing
keywords: Laplace, additive smoothing, +1 pseudocount, zero probability, unseen
criteria: >
  To handle unseen feature values: P(xᵢ=v|y=c) = (count+1)/(count(y=c)+|vocabulary|).
  Student must explain why it's needed (zero probability problem).
points: 2
common_error: Not applying smoothing; not explaining why zero probability is catastrophic.
socratic_hint: >
  "What happens to the entire product Π P(xᵢ|y) if even one term is zero?
  How does adding a small count to every cell of the frequency table prevent this?"
---
rule_id: HMM_L_001
topic: HMM - Likelihood Problem
subtopic: Problem Definition
keywords: forward algorithm, P(O|λ), observation sequence, probability of observation
criteria: >
  Student must state the problem: given observation sequence O=o₁,o₂,…,oT and model
  λ=(A,B,π), compute P(O|λ) — the probability of the observation sequence.
points: 1
common_error: Confusing likelihood problem with decoding or learning problem.
socratic_hint: >
  "There are three fundamental HMM problems. Which one asks for the probability of
  seeing a specific observation sequence, and what does the forward algorithm compute?"
---
rule_id: HMM_L_002
topic: HMM - Likelihood Problem
subtopic: Forward Variable Definition
keywords: α_t(i), α, forward variable, P(o₁o₂…oT, qT=i|λ)
criteria: >
  Student must define αₜ(i) = P(o₁,o₂,…,oₜ, qₜ=sᵢ | λ) — the probability of
  observing the partial sequence up to time t AND being in state i at time t.
points: 2
common_error: Defining α as only the emission probability; not including the joint state.
socratic_hint: >
  "The forward variable αₜ(i) captures two things simultaneously. What is the state
  at time t in this definition, and what observation sequence is being tracked up to t?"
---
rule_id: HMM_L_003
topic: HMM - Likelihood Problem
subtopic: Initialization
keywords: α₁(i), initialization, π_i, b_i(o₁), initial state probability, emission
criteria: >
  Initialization: α₁(i) = πᵢ · bᵢ(o₁) for all states i,
  where πᵢ is the initial state probability and bᵢ(o₁) is the emission probability.
points: 2
common_error: Using α₁(i) = πᵢ alone (forgetting the emission); wrong time index.
socratic_hint: >
  "At t=1, you start by picking an initial state with probability πᵢ. But you also
  observe o₁ — what additional probability do you need to include at this first step?"
---
rule_id: HMM_L_004
topic: HMM - Likelihood Problem
subtopic: Recursion Step
keywords: αₜ₊₁(j), recursion, Σ αₜ(i) aᵢⱼ, transition, emission bⱼ(oₜ₊₁)
criteria: >
  αₜ₊₁(j) = [Σᵢ αₜ(i) · aᵢⱼ] · bⱼ(oₜ₊₁) for all states j, t=1,…,T-1.
  Student must show the sum over all previous states and the emission multiplication.
points: 3
common_error: Forgetting to sum over all previous states i; forgetting bⱼ(oₜ₊₁); wrong time index.
socratic_hint: >
  "At time t+1, state j could have been reached from any state i at time t.
  How do you combine: (1) the probability of being in each state i at time t,
  (2) the transition from i to j, and (3) the emission of oₜ₊₁ from j?"
---
rule_id: HMM_L_005
topic: HMM - Likelihood Problem
subtopic: Termination
keywords: P(O|λ), termination, Σ αT(i), sum forward variables, final probability
criteria: >
  P(O|λ) = Σᵢ αT(i) — the sum of forward variables over all states at the final time step T.
points: 2
common_error: Taking max instead of sum; not summing over all states.
socratic_hint: >
  "You've computed αT(i) for all states i at the last time step. The observation
  sequence could end in any state. How do you combine these to get the total P(O|λ)?"
---
rule_id: HMM_D_001
topic: HMM - Decoding Problem
subtopic: Problem Definition
keywords: Viterbi, Q*, best state sequence, argmax, most likely path
criteria: >
  Student must state: find Q* = argmax_Q P(Q,O|λ) — the single most likely
  state sequence given the observations and model.
points: 1
common_error: Confusing decoding with likelihood; trying to compute P(O|λ) instead of Q*.
socratic_hint: >
  "Unlike the likelihood problem, decoding asks for a sequence, not a probability.
  What are you maximizing over, and what does Q* represent physically?"
----
rule_id: HMM_D_002
topic: HMM - Decoding Problem
subtopic: Viterbi Variable Definition
keywords: δ_t(i), max, most probable path, δ, Viterbi variable
criteria: >
  δₜ(i) = max_{q₁,…,qₜ₋₁} P(q₁,…,qₜ=i, o₁,…,oₜ|λ) — the probability of the
  most probable path ending in state i at time t.
points: 2
common_error: Using sum instead of max (confusing with forward algorithm).
socratic_hint: >
  "How does δₜ(i) differ from αₜ(i)? Both involve partial paths ending in state i
  at time t — but one sums over all paths, the other takes the maximum. Which is which?"
---
rule_id: HMM_D_003
topic: HMM - Decoding Problem
subtopic: Initialization
keywords: δ₁(i), π_i, b_i(o₁), initialization, Viterbi init
criteria: >
  Initialization: δ₁(i) = πᵢ · bᵢ(o₁) and ψ₁(i) = 0 (no predecessor at t=1).
points: 2
common_error: Forgetting ψ (backpointer) initialization; same initialization error as forward algorithm.
socratic_hint: >
  "Viterbi needs to remember not just the best probability, but also which state
  it came from at each step. What data structure stores this 'came from' information?"
---
rule_id: HMM_D_004
topic: HMM - Decoding Problem
subtopic: Viterbi Recursion
keywords: δₜ₊₁(j) = max_i δₜ(i) aᵢⱼ · bⱼ(oₜ₊₁), backpointer ψ, argmax
criteria: >
  δₜ₊₁(j) = [max_i δₜ(i)·aᵢⱼ] · bⱼ(oₜ₊₁) and ψₜ₊₁(j) = argmax_i δₜ(i)·aᵢⱼ.
  Both the max probability AND the argmax backpointer must be computed and stored.
points: 3
common_error: Computing δ correctly but omitting ψ; using sum instead of max.
socratic_hint: >
  "You've computed the best probability to arrive in state j at t+1. But to recover
  the path later, you need to remember which state i achieved that maximum. Where do you store it?"
---
rule_id: HMM_D_005
topic: HMM - Decoding Problem
subtopic: Backtracking
keywords: backtrack, traceback, optimal path, ψ, q*_T, backpointer
criteria: >
  Student must: (1) find q*_T = argmax_i δT(i); (2) backtrack using ψ:
  q*_t = ψ_{t+1}(q*_{t+1}) for t=T-1,…,1. Reverse the sequence for final answer.
points: 2
common_error: Not backtracking; reporting only the final state; not reversing the path.
socratic_hint: >
  "You found the best final state q*_T. Now, the ψ backpointers you stored tell you
  which state preceded it. How do you use these to recover the full optimal sequence q*₁,…,q*_T?"
---
rule_id: HMM_BW_001
topic: HMM - Learning Problem
subtopic: Problem Definition and EM Connection
keywords: Baum-Welch, re-estimate, λ*, EM algorithm, learn parameters, A, B, π
criteria: >
  Student must state: given observation sequences, find λ=(A,B,π) that maximizes P(O|λ).
  Baum-Welch is an instance of the EM algorithm for HMMs.
points: 2
common_error: Confusing with decoding; not stating that parameters A, B, π are all to be learned.
socratic_hint: >
  "Baum-Welch is the HMM-specific version of a general optimization framework.
  Which framework iterates between estimating hidden quantities and re-estimating parameters?"
---
rule_id: HMM_BW_002
topic: HMM - Learning Problem
subtopic: Backward Variable
keywords: βₜ(i), backward variable, P(oₜ₊₁,…,oT | qₜ=i, λ), backward algorithm
criteria: >
  βₜ(i) = P(oₜ₊₁,oₜ₊₂,…,oT | qₜ=sᵢ, λ) — probability of the future observation
  sequence given state i at time t. Recursion: βₜ(i) = Σⱼ aᵢⱼ·bⱼ(oₜ₊₁)·βₜ₊₁(j).
points: 3
common_error: Not defining the backward variable; confusing forward and backward variables.
socratic_hint: >
  "The forward variable α looks at the past (from t=1 to t). What does the backward
  variable β look at? And how do you compute βₜ(i) given βₜ₊₁(j)?"
---
rule_id: HMM_BW_003
topic: HMM - Learning Problem
subtopic: State Occupancy Probability (γ)
keywords: γₜ(i), P(qₜ=i|O,λ), gamma, state probability, αβ product
criteria: >
  γₜ(i) = P(qₜ=sᵢ|O,λ) = [αₜ(i)·βₜ(i)] / Σⱼ αₜ(j)·βₜ(j).
  Student must use both forward and backward variables.
points: 2
common_error: Using only α (forgetting β); not normalizing.
socratic_hint: >
  "γₜ(i) uses information from both the past (α) and the future (β). How do you
  combine αₜ(i) and βₜ(i) to get the probability of being in state i at time t?"
---
rule_id: HMM_BW_004
topic: HMM - Learning Problem
subtopic: Re-estimation Formulas
keywords: re-estimate, â_ij, b̂_j(v), π̂_i, expected counts, update
criteria: >
  π̂ᵢ = γ₁(i); âᵢⱼ = Σₜ ξₜ(i,j) / Σₜ γₜ(i);
  b̂ⱼ(vₖ) = Σₜ:oₜ=vₖ γₜ(j) / Σₜ γₜ(j).
  Student must show all three re-estimation equations.
points: 3
common_error: Incorrect denominator; showing only one of the three re-estimation formulas.
socratic_hint: >
  "The M-step updates each parameter. For â_ij: the numerator counts expected transitions
  from i to j — what does the denominator normalize by? And what does b̂_j(v) estimate?"
---
rule_id: EM_001
topic: EM Algorithm
subtopic: Problem Setup (Latent Variables)
keywords: latent variable, hidden variable, incomplete data, Z, expectation maximization
criteria: >
  Student must identify latent variables Z and explain that EM maximizes the
  likelihood P(X|θ) by iterating between estimating Z and updating parameters θ.
points: 2
common_error: Not defining latent variables; confusing EM with direct MLE.
socratic_hint: >
  "EM handles missing or hidden information. In this problem, what are the hidden
  (latent) variables Z that we cannot directly observe?"
---
rule_id: EM_002
topic: EM Algorithm
subtopic: E-Step (Expectation)
keywords: E-step, Q function, expected log-likelihood, posterior, E[Z|X,θ_old]
criteria: >
  E-step: compute Q(θ|θ_old) = E_{Z|X,θ_old}[log P(X,Z|θ)] — the expected complete-data
  log-likelihood, where expectation is taken over latent variables given current parameters.
points: 3
common_error: Using current θ in the expectation instead of θ_old; not taking expectation over Z.
socratic_hint: >
  "In the E-step, you 'fill in' the missing data using your current parameter estimate.
  Formally, what are you computing the expectation of, and over what distribution?"
---
rule_id: EM_003
topic: EM Algorithm
subtopic: M-Step (Maximization)
keywords: M-step, maximize Q, update θ, argmax, new parameters
criteria: >
  M-step: θ_new = argmax_θ Q(θ|θ_old) — update parameters by maximizing the Q function.
  Student must show the parameter update equations derived from ∂Q/∂θ = 0.
points: 3
common_error: Not maximizing; updating θ with gradient descent instead of closed-form argmax.
socratic_hint: >
  "You've computed Q. Now, to find the new parameters θ_new, you maximize Q over θ.
  For this model, can you take the derivative ∂Q/∂θ and set it to zero?"
---
rule_id: EM_004
topic: EM Algorithm
subtopic: Convergence Property
keywords: convergence, monotone increase, lower bound, ELBO, Jensen's inequality
criteria: >
  Student must state that EM guarantees monotone non-decreasing log-likelihood:
  log P(X|θ_new) ≥ log P(X|θ_old) at every iteration (via Jensen's inequality).
points: 2
common_error: Claiming EM converges to global maximum (it may reach local maximum).
socratic_hint: >
  "EM is guaranteed to not decrease the likelihood at each iteration. But does it
  always reach the global maximum? What property of the likelihood surface can cause issues?"
---
rule_id: KM_001
topic: K-Means Clustering
subtopic: Initialization
keywords: k, cluster centers, initialize, random, centroids, k clusters
criteria: >
  Student must specify k (number of clusters) and initialize k centroids,
  either by randomly selecting k data points or random assignment.
points: 1
common_error: Not specifying how centroids are initialized; not stating k.
socratic_hint: >
  "Before running K-means, two things must be decided. What is k, and how do you
  choose the starting positions of the k centroids?"
---
rule_id: KM_002
topic: K-Means Clustering
subtopic: Assignment Step
keywords: assign, nearest centroid, Euclidean distance, cluster assignment, argmin ||x-μ||²
criteria: >
  Assignment: for each point xᵢ, assign it to cluster k* = argmin_k ||xᵢ - μₖ||².
  Student must compute Euclidean distances to all centroids and assign to the nearest.
points: 2
common_error: Using Manhattan distance without justification; not assigning ALL points.
socratic_hint: >
  "In the assignment step, each point needs to find its closest centroid.
  What distance metric does standard K-means use, and how do you compute it?"
---
rule_id: KM_003
topic: K-Means Clustering
subtopic: Update Step
keywords: update centroids, mean, μₖ = (1/|Cₖ|)Σ x, recompute, new centroid
criteria: >
  Update: μₖ = (1/|Cₖ|) Σ_{xᵢ ∈ Cₖ} xᵢ — recompute each centroid as the mean
  of all points currently assigned to that cluster.
points: 2
common_error: Using median instead of mean; not including all points in the cluster.
socratic_hint: >
  "After assigning all points to clusters, how do you reposition each centroid?
  The centroid should be the 'center' of all assigned points — what statistic is that?"
---
rule_id: KM_004
topic: K-Means Clustering
subtopic: Convergence Criterion
keywords: convergence, no change, assignments stable, objective, WCSS, inertia
criteria: >
  K-means converges when cluster assignments stop changing between iterations.
  Equivalently, the within-cluster sum of squares (WCSS) stops decreasing.
points: 1
common_error: Running a fixed number of iterations without checking convergence.
socratic_hint: >
  "How do you know when to stop the K-means loop? What condition on the cluster
  assignments or the objective function indicates convergence?"
---
rule_id: KM_005
topic: K-Means Clustering
subtopic: Objective Function
keywords: WCSS, inertia, J = Σₖ Σ_{x∈Cₖ} ||x-μₖ||², minimize, objective
criteria: >
  Student should state the objective: minimize J = Σₖ Σ_{xᵢ∈Cₖ} ||xᵢ - μₖ||².
  K-means is guaranteed to decrease (or maintain) J at every iteration.
points: 2
common_error: Not writing the objective function; claiming K-means maximizes WCSS.
socratic_hint: >
  "K-means is an optimization algorithm. What quantity does it minimize?
  How does the assignment step decrease this, and how does the update step decrease it?"
---
rule_id: CNN_001
topic: CNN Trainable Parameters
subtopic: Convolutional Layer Parameters
keywords: filter, kernel, weights, F*F*C_in*C_out + C_out, bias, conv layer
criteria: >
  Number of parameters in conv layer = (F × F × C_in × C_out) + C_out,
  where F=kernel size, C_in=input channels, C_out=number of filters (output channels).
  The +C_out accounts for one bias per filter.
points: 3
common_error: Forgetting to multiply by C_in (input channels); omitting bias term.
socratic_hint: >
  "Each filter in a conv layer has a weight for every input channel.
  If the input has C_in channels and the filter is F×F, how many weights does ONE filter have?
  Then if you have C_out filters total, plus one bias each...?"
---
rule_id: CNN_002
topic: CNN Trainable Parameters
subtopic: Fully Connected Layer Parameters
keywords: FC layer, dense layer, W*H*C*neurons, weights + biases, linear layer
criteria: >
  FC layer parameters = (input_size × output_size) + output_size.
  After flattening a feature map of shape H×W×C: input_size = H × W × C.
points: 2
common_error: Not flattening the feature map before computing FC parameters; forgetting bias.
socratic_hint: >
  "Before the fully connected layer, what shape does the feature map have, and how do you
  convert it to a 1D vector? Then how many weights connect this vector to {n} output neurons?"
---
rule_id: CNN_003
topic: CNN Trainable Parameters
subtopic: Pooling Layer Parameters
keywords: pooling, max pool, average pool, no parameters, zero trainable parameters
criteria: >
  Pooling layers (max pooling, average pooling) have ZERO trainable parameters.
  They only apply a fixed operation (max or average) over a window.
points: 1
common_error: Assigning parameters to pooling layers.
socratic_hint: >
  "Unlike conv layers, pooling layers have no learnable weights. Why?
  What operation does max pooling perform, and does it involve any learned parameters?"
---
rule_id: CNN_004
topic: CNN Trainable Parameters
subtopic: BatchNorm Layer Parameters
keywords: batch normalization, gamma, beta, 2*C parameters, scale, shift
criteria: >
  BatchNorm has 2 × C trainable parameters per layer (scale γ and shift β,
  one per channel), plus 2 non-trainable running stats (mean, variance).
points: 2
common_error: Saying BatchNorm has zero parameters; including running stats in trainable count.
socratic_hint: >
  "Batch normalization learns two parameters per channel to rescale and shift the
  normalized output. What are they called, and if there are C channels, how many total?"
---
rule_id: CNN_005
topic: CNN Trainable Parameters
subtopic: Total Parameter Count
keywords: total parameters, sum all layers, architecture, trainable
criteria: >
  Student must sum parameters from all layers: conv layers, FC layers, BatchNorm (if any).
  Must show per-layer calculation and then sum. Should exclude pooling (0 params).
points: 3
common_error: Summing wrong values; including pooling; not showing per-layer breakdown.
socratic_hint: >
  "Let's build a table — for each layer in the architecture, compute its parameter count separately,
  then add them up. Have you listed every layer type that has trainable weights?"
---
rule_id: RF_001
topic: Random Forest
subtopic: Bootstrap Sampling
keywords: bootstrap, sampling with replacement, bagging, subset, B datasets
criteria: >
  Student must describe bootstrap sampling: for each tree, sample N examples
  WITH REPLACEMENT from the training set (same size N), producing B different datasets.
points: 2
common_error: Describing sampling WITHOUT replacement; not specifying "with replacement."
socratic_hint: >
  "In Random Forest, each tree is trained on a different dataset. How is each
  dataset created from the original — and importantly, can the same example appear multiple times?"
---
rule_id: RF_002
topic: Random Forest
subtopic: Random Feature Subsets
keywords: random features, m features, sqrt(p), feature subset, decorrelate trees
criteria: >
  At each split in each tree, only m randomly selected features are considered,
  where m ≈ √p (p = total features) for classification, m ≈ p/3 for regression.
points: 2
common_error: Not mentioning the random feature subset; confusing with bagging (which uses all features).
socratic_hint: >
  "What makes Random Forest different from plain Bagging of decision trees?
  At each split node, how many features does Random Forest consider, versus how many are available?"
---
rule_id: RF_003
topic: Random Forest
subtopic: Aggregation (Voting/Averaging)
keywords: majority vote, average, ensemble, aggregate, combine predictions
criteria: >
  For classification: final prediction = majority vote across all B trees.
  For regression: final prediction = average of all B trees' outputs.
points: 2
common_error: Using a single tree's output; not specifying vote vs. average for the task type.
socratic_hint: >
  "After training B trees and getting B predictions for a new example,
  how does Random Forest combine them for classification? And what changes for regression?"
---
rule_id: BB_001
topic: Bagging & Boosting
subtopic: Bagging Definition
keywords: bagging, bootstrap aggregating, parallel, independent, variance reduction
criteria: >
  Bagging: train B models independently on B bootstrap samples; aggregate by majority vote
  (classification) or average (regression). Reduces variance, not bias.
points: 2
common_error: Saying bagging reduces bias; describing sequential training (that's boosting).
socratic_hint: >
  "Bagging stands for Bootstrap AGGregating. Are the B models trained sequentially
  or independently in parallel? Which component of the bias-variance tradeoff does it address?"
---
rule_id: BB_002
topic: Bagging & Boosting
subtopic: Boosting Definition
keywords: boosting, sequential, weighted, AdaBoost, focus on errors, weak learners
criteria: >
  Boosting: train models SEQUENTIALLY where each model focuses on examples misclassified
  by previous models (via increased sample weights). Reduces bias. Combines weak learners.
points: 2
common_error: Saying boosting is parallel; not mentioning the weighted re-sampling mechanism.
socratic_hint: >
  "In boosting, later models are not independent — they focus on the 'hard' examples.
  How do they know which examples to focus on? What changes between rounds?"
---
rule_id: BB_003
topic: Bagging & Boosting
subtopic: AdaBoost Weight Update
keywords: AdaBoost, αₜ, wᵢ update, error εₜ, weight, ln((1-ε)/ε), classifier weight
criteria: >
  In AdaBoost: (1) compute error εₜ = Σᵢ wᵢ·𝟙[yᵢ ≠ hₜ(xᵢ)];
  (2) classifier weight αₜ = (1/2)ln((1-εₜ)/εₜ);
  (3) update sample weights: wᵢ ← wᵢ·exp(-αₜyᵢhₜ(xᵢ)), then normalize.
points: 3
common_error: Wrong formula for αₜ; not normalizing weights; missing the -αy h(x) exponent.
socratic_hint: >
  "A classifier with error εₜ=0.5 (random guessing) should get weight 0. What does
  αₜ = (1/2)ln((1-ε)/ε) evaluate to when ε=0.5? And how should misclassified samples'
  weights change for the next round?"
---
rule_id: GB_001
topic: Gradient Boosting
subtopic: Residual as Pseudo-Residuals
keywords: residual, pseudo-residual, gradient, rᵢ = yᵢ - F(xᵢ), negative gradient
criteria: >
  At each step, compute pseudo-residuals rᵢ = -∂L/∂F(xᵢ) = yᵢ - Fₘ₋₁(xᵢ) (for MSE loss).
  The new tree is fit to these residuals, not the original targets.
points: 3
common_error: Fitting new trees to original y values; not computing residuals from current model.
socratic_hint: >
  "Gradient boosting builds trees sequentially. What does the (m+1)-th tree try to predict —
  the original labels, or the mistakes of the current ensemble Fₘ? How are those 'mistakes' calculated?"
---
rule_id: GB_002
topic: Gradient Boosting
subtopic: Ensemble Update
keywords: F_m, F_{m-1} + η*h_m, learning rate, η, additive model, step
criteria: >
  Update rule: Fₘ(x) = Fₘ₋₁(x) + η · hₘ(x), where hₘ is the new tree and η
  is the learning rate (shrinkage parameter, 0 < η ≤ 1).
points: 2
common_error: Omitting the learning rate; replacing Fₘ₋₁ instead of adding to it.
socratic_hint: >
  "In gradient boosting, you ADD a new tree to the existing ensemble — you don't replace it.
  What role does the learning rate η play in this addition?"
---
rule_id: LR_001
topic: Logistic Regression
subtopic: Sigmoid Function
keywords: sigmoid, σ(z), 1/(1+e^{-z}), probability, bounded [0,1]
criteria: >
  Student must write σ(z) = 1 / (1 + e^{-z}) where z = w·x + b,
  and interpret the output as P(y=1|x).
points: 2
common_error: Using a linear output as probability; not constraining output to [0,1].
socratic_hint: >
  "Linear regression can output values outside [0,1]. For a classification probability,
  what function maps any real number to the range (0,1)?"
---
rule_id: LR_002
topic: Logistic Regression
subtopic: Binary Cross-Entropy Loss
keywords: log loss, binary cross-entropy, L = -[y log ŷ + (1-y)log(1-ŷ)], NLL
criteria: >
  Loss: L = -[y·log(ŷ) + (1-y)·log(1-ŷ)] for binary classification.
  Student must show both terms and explain their roles.
points: 2
common_error: Using MSE for a classification problem; missing the (1-y) term for negative class.
socratic_hint: >
  "If y=1 (positive class), which term in the loss is active? If y=0 (negative class)?
  Why does this loss function penalize confident wrong predictions so heavily?"
---
rule_id: LR_003
topic: Logistic Regression
subtopic: Gradient Computation
keywords: ∂L/∂w, gradient, ŷ-y, (ŷ-y)*x, update weights, backprop logistic
criteria: >
  ∂L/∂w = (1/N)Σ (ŷᵢ - yᵢ)·xᵢ for batch gradient descent.
  This is the same form as linear regression gradient (convenient result of sigmoid + cross-entropy).
points: 3
common_error: Not averaging over N; forgetting the xᵢ multiplication.
socratic_hint: >
  "The gradient of cross-entropy + sigmoid has a surprisingly clean form.
  After working through the chain rule, ∂L/∂w simplifies to (ŷ - y) times what?"
---
rule_id: PM_001
topic: Performance Metrics
subtopic: Confusion Matrix
keywords: TP, TN, FP, FN, confusion matrix, true positive, false positive
criteria: >
  Student must construct a 2×2 confusion matrix with TP, TN, FP, FN correctly placed:
  TP=correctly predicted positive; TN=correctly predicted negative;
  FP=negative predicted as positive; FN=positive predicted as negative.
points: 2
common_error: Swapping FP and FN; not labeling axes (predicted vs actual).
socratic_hint: >
  "The confusion matrix rows represent actual classes; columns represent predicted classes.
  A 'False Positive' is when the model predicts positive but the actual label is...?"
---
rule_id: PM_002
topic: Performance Metrics
subtopic: Precision and Recall
keywords: precision, recall, TP/(TP+FP), TP/(TP+FN), sensitivity, PPV
criteria: >
  Precision = TP/(TP+FP) (of all predicted positives, how many are actually positive).
  Recall = TP/(TP+FN) (of all actual positives, how many did we find).
points: 2
common_error: Swapping precision and recall formulas; using TN in the denominator.
socratic_hint: >
  "Precision answers: 'Of everything I labeled positive, how much was correct?'
  Recall answers: 'Of all actual positives, how many did I catch?' Which formula is which?"
---
rule_id: PM_003
topic: Performance Metrics
subtopic: F1 Score
keywords: F1, harmonic mean, 2*P*R/(P+R), precision-recall balance
criteria: >
  F1 = 2 · Precision · Recall / (Precision + Recall) — harmonic mean of P and R.
  F1 is preferred over accuracy when classes are imbalanced.
points: 2
common_error: Using arithmetic mean instead of harmonic mean; using F1 when classes are balanced.
socratic_hint: >
  "The F1 score balances precision and recall. Why do we use the harmonic mean rather than
  the arithmetic mean (P+R)/2? What happens to harmonic mean when one value is near zero?"
---
rule_id: PM_004
topic: Performance Metrics
subtopic: ROC Curve and AUC
keywords: ROC, AUC, TPR, FPR, threshold, receiver operating characteristic
criteria: >
  ROC plots TPR (recall) vs. FPR (FP rate = FP/(FP+TN)) at varying classification thresholds.
  AUC=area under ROC curve; AUC=0.5 is random; AUC=1.0 is perfect.
points: 2
common_error: Plotting precision vs. recall (that's PR curve, not ROC); confusing FPR with FNR.
socratic_hint: >
  "The ROC curve shows performance across all thresholds. The y-axis is TPR (recall).
  What is on the x-axis — and how is the False Positive Rate defined?"
---
rule_id: BL_001
topic: Bayesian Learning
subtopic: Maximum A Posteriori (MAP)
keywords: MAP, argmax, P(h|D) ∝ P(D|h)*P(h), posterior, prior, likelihood, hypothesis
criteria: >
  h_MAP = argmax_h P(h|D) = argmax_h P(D|h)·P(h).
  Student must state Bayes' theorem application and identify the prior P(h) and likelihood P(D|h).
points: 3
common_error: Ignoring the prior (that gives MLE, not MAP); not applying Bayes' theorem.
socratic_hint: >
  "MAP finds the most probable hypothesis given data. This requires Bayes' theorem.
  What are the two terms being multiplied, and which one does Maximum Likelihood ignore?"
---
rule_id: BL_002
topic: Bayesian Learning
subtopic: Maximum Likelihood Estimation (MLE)
keywords: MLE, argmax P(D|h), likelihood only, no prior, h_ML
criteria: >
  h_ML = argmax_h P(D|h) — maximizes only the likelihood, assuming uniform prior.
  For iid data: log P(D|h) = Σᵢ log P(dᵢ|h) (log-likelihood for numerical stability).
points: 2
common_error: Including the prior in MLE; not using log-likelihood.
socratic_hint: >
  "MLE and MAP look similar but differ in one key way. MAP multiplies likelihood by the prior.
  What happens to the prior term in MLE? What assumption makes this valid?"
---
rule_id: BL_003
topic: Bayesian Learning
subtopic: Bayes Optimal Classifier
keywords: Bayes optimal, expected loss, optimal classifier, posterior predictive, Σ P(h|D)*h(x)
criteria: >
  Bayes optimal prediction = argmax_vⱼ Σₕ P(vⱼ|h)·P(h|D) — combines all hypotheses
  weighted by posterior probability. Minimizes expected error over all hypotheses.
points: 3
common_error: Using only MAP hypothesis; not summing over all hypotheses.
socratic_hint: >
  "The Bayes optimal classifier uses all hypotheses, not just the best one. How does weighting
  each hypothesis by P(h|D) differ from MAP, which picks only the single best hypothesis?"
---
rule_id: BVT_001
topic: Bias-Variance Tradeoff
subtopic: Decomposition of MSE
keywords: bias, variance, noise, MSE = Bias² + Variance + noise, decomposition
criteria: >
  E[(y - ŷ)²] = Bias²(ŷ) + Var(ŷ) + σ²_noise.
  Student must define: Bias = E[ŷ] - y (systematic error); Variance = spread of ŷ across datasets.
points: 3
common_error: Missing the noise term; confusing bias with variance definition.
socratic_hint: >
  "The total prediction error decomposes into three parts. Bias measures systematic
  error (how wrong on average); variance measures sensitivity to training data.
  What is the third irreducible term, and where does it come from?"
---
rule_id: BVT_002
topic: Bias-Variance Tradeoff
subtopic: Underfitting vs. Overfitting
keywords: underfitting, overfitting, high bias, high variance, model complexity, training error
criteria: >
  Underfitting = high bias, low variance (model too simple, large training error).
  Overfitting = low bias, high variance (model too complex, small train error but large test error).
points: 2
common_error: Reversing high bias / high variance; not connecting to model complexity.
socratic_hint: >
  "A degree-1 polynomial fit to complex data will have consistent predictions
  (across different training sets) but systematically wrong ones. Is that high bias or high variance?
  What about a degree-20 polynomial?"
---
rule_id: PCA_001
topic: PCA / SVD
subtopic: Covariance Matrix
keywords: covariance, Σ = (1/N)X^T X, centered data, symmetric, positive semidefinite
criteria: >
  Student must: (1) center the data (subtract mean); (2) compute covariance matrix
  Σ = (1/(N-1)) Xᵀ·X; (3) note it is symmetric and positive semidefinite.
points: 2
common_error: Forgetting to center data; using N instead of N-1 (Bessel's correction).
socratic_hint: >
  "Before computing the covariance matrix, the data must be preprocessed.
  What operation on each feature ensures the covariance matrix measures spread, not mean?"
---
rule_id: PCA_002
topic: PCA / SVD
subtopic: Eigenvalue Decomposition
keywords: eigenvalue, eigenvector, Σv = λv, principal components, largest eigenvalue
criteria: >
  Principal components = eigenvectors of Σ sorted by descending eigenvalue.
  The k-th PC explains λₖ/Σλᵢ fraction of total variance.
points: 3
common_error: Using random vectors as PCs; not sorting by eigenvalue magnitude.
socratic_hint: >
  "The first principal component is the direction of maximum variance in the data.
  Mathematically, which eigenvector of the covariance matrix corresponds to this direction?"
---
rule_id: PCA_003
topic: PCA / SVD
subtopic: Dimensionality Reduction
keywords: project, reduce, X_reduced = X * V_k, k components, variance explained
criteria: >
  To project to k dimensions: X_reduced = X · Vₖ where Vₖ is the matrix of top-k eigenvectors.
  Student should compute the fraction of variance explained by chosen k components.
points: 2
common_error: Not defining k; using wrong eigenvectors (should be k largest).
socratic_hint: >
  "You've found all eigenvectors. To reduce to k=2 dimensions, which eigenvectors do you select,
  and how do you use them to project every data point?"
---
rule_id: GMM_001
topic: GMM
subtopic: Model Definition
keywords: GMM, mixture, K Gaussians, πₖ, μₖ, Σₖ, weighted sum of Gaussians
criteria: >
  P(x) = Σₖ πₖ · N(x; μₖ, Σₖ) where πₖ are mixture weights (Σπₖ=1, πₖ≥0),
  μₖ are means, Σₖ are covariance matrices.
points: 2
common_error: Forgetting the mixing weights πₖ; not stating the constraint Σπₖ=1.
socratic_hint: >
  "A GMM is a weighted combination of Gaussian distributions. What constraint must
  the mixing weights πₖ satisfy? And what parameters describe each Gaussian component?"
---
rule_id: GMM_002
topic: GMM
subtopic: E-Step (Responsibility)
keywords: responsibility, rᵢₖ, posterior, γ(zᵢₖ), P(k|xᵢ), soft assignment
criteria: >
  rᵢₖ = πₖ·N(xᵢ;μₖ,Σₖ) / Σⱼ πⱼ·N(xᵢ;μⱼ,Σⱼ) — the "responsibility" of component k for point xᵢ.
  This is a soft assignment (sums to 1 over k for each i).
points: 3
common_error: Hard-assigning each point to one cluster (that's K-means, not GMM).
socratic_hint: >
  "Unlike K-means, GMM makes 'soft' assignments. What does rᵢₖ represent, and why
  does each point get a fractional membership in multiple components rather than belonging to one?"
---
rule_id: GMM_003
topic: GMM
subtopic: M-Step (Parameter Update)
keywords: update μₖ, Σₖ, πₖ, weighted mean, Nₖ = Σ rᵢₖ, M-step
criteria: >
  Nₖ = Σᵢ rᵢₖ; πₖ = Nₖ/N; μₖ = (1/Nₖ)Σᵢ rᵢₖxᵢ; Σₖ = (1/Nₖ)Σᵢ rᵢₖ(xᵢ-μₖ)(xᵢ-μₖ)ᵀ.
  All three parameters must be updated.
points: 3
common_error: Updating only μₖ; not using the responsibilities rᵢₖ as weights.
socratic_hint: >
  "In the M-step, each update is a weighted version of the standard MLE formula,
  where rᵢₖ acts as the weight for point xᵢ in component k.
  How does the weighted mean formula for μₖ differ from the regular unweighted mean?"
---
rule_id: KNN_001
topic: KNN
subtopic: Algorithm Steps
keywords: K nearest neighbors, distance, vote, k, euclidean, classification
criteria: >
  Steps: (1) compute distance from query point to all training points; (2) select k nearest;
  (3) for classification: majority vote; for regression: average of k neighbors' values.
points: 2
common_error: Not specifying k; using wrong aggregation (mean vs. vote) for the task type.
socratic_hint: >
  "KNN is an instance-based learner with no training phase. Given a new test point,
  what are the three steps to make a prediction, and how does the final answer depend on k?"
---
rule_id: KNN_002
topic: KNN
subtopic: Effect of K
keywords: k=1, small k, large k, overfitting, underfitting, decision boundary, smooth
criteria: >
  Small k (e.g., k=1): complex decision boundary, overfits (high variance).
  Large k: smoother boundary, underfits (high bias). Optimal k found via cross-validation.
points: 2
common_error: Reversing the effect; not connecting k to bias-variance tradeoff.
socratic_hint: >
  "What does the decision boundary look like when k=1 vs k=N (all training points)?
  Which extreme leads to memorizing the training set, and which to ignoring all detail?"
---
