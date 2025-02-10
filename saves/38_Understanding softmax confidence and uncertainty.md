## Understanding softmax confidence and uncertainty
- Author: Pearce, Tim and Brintrup, Alexandra and Zhu, Jun.
- Journal: arXiv
- Year: 2021
- Link: https://arxiv.org/pdf/2106.04972


### Abstract
- **Neural Networks & Uncertainty**: Neural networks often struggle to express uncertainty when encountering data that is different from what they were trained on (out-of-distribution, OOD).
- **Softmax confidence** is a metric derived from the softmax function, which gives probabilities to each class in classification tasks. However, using this confidence directly as a measure of uncertainty only works moderately well for OOD detection tasks.
- This paper explores the reasons behind this contradiction and identifies two key implicit biases:
  - **Approximately Optimal Decision Boundary Structure**: The way decisions are made at the boundaries of different classes can impact the network's ability to express uncertainty.
  - **Filtering Effects of Deep Networks**: Deep networks may filter input data, affecting the representation of features and how uncertainty is conveyed through the softmax scores.
- Simplistic views about how softmax confidence operates can be misleading due to the complexity of feature representation.
- The paper conducts experiments to understand **why softmax confidence sometimes fails**, concluding that overlap between the training dataset and OOD data representations is a significant factor, rather than the network's inability to extrapolate from training to new examples.


### Introduction
- Standard neural networks (without modifications) often fail to represent **epistemic uncertainty**, which is the uncertainty about our knowledge of the model itself.
  - This issue can be illustrated using visual examples in lower-dimensional spaces, showing how networks might make overly confident predictions that are incorrect.
- Researchers explore alternative methods, like ensembling techniques and Bayesian neural networks, which are intended to better handle uncertainty.
  - When testing these new methods, researchers often use the **softmax function** (a mechanism that converts raw model outputs into probabilities) as a benchmark for measuring uncertainty.
  - Despite softmax not being designed to quantify uncertainty, it performs reasonably well, achieving prediction accuracies that range from 75% to 99% in distinguishing if a data point is part of the training data or not (out-of-distribution (OOD) detection).

### Background
- **Neural Network Input and Output**
  - **Input** | A vector $x \in\mathbb{R}^D$
  - **Activation** | The network produces final layer activations $z = \psi (x) \in \mathbb{R}^H$.
  - **Predictions** | These activations are transformed through a softmax function, resulting in predicted probabilities $\hat{y} = \sigma(z)$ in the range $[0,1]^K.$
- **Softmax function**
$$\sigma(\mathbf{z}) = \frac{\exp w^T _i \mathbf{z}}{\sum^K _{j=1} \exp w^T _j \mathbf{z}}.$$
  - $w_i$ | the weight vector corresponding to class $i$.
  -  \mathbf{z} | the activations from the final layer.
  - This function normalizes the activations to produce a probability distribution over classes $K$.
- **Distributions**
  - **Training Distribution** | $x \sim D_{in}$ or its probability $p_{in} (x)$.
  - **Final layer activations distribution** |  $z \sim D_{in}$ or $p_{in} (z)$.
  - **Out-of-distribution (OOD)** | For outliers, $x \sim D_{out}$ or $p_{out} (x)$.
- **Uncertainty measures**
  - **Max predicted probability** measures uncertainty by taking the negative of the maximum predicted probability.
$$U_{max} (z) = - \max_i \sigma(z)_i$$
  - **Entropy** quantifies uncertainty based on the distribution of probabilities across all classes.
$$U_{entropy} (z) =- - \sum^K _{i=1} \sigma(z)_i \log (\sigma(z)_i)$$
  - **Density-based uncertainty** score evaluates uncertainty from the log likelihood.
$$U_{density} (z) = - \log \hat{q(z)}$$
  - $\hat{q(z)}$ estimates the probability density of the activations.
- **Gaussian Mixture Model**
  - Used to estimate the probability density $\hat{q(z)}$
$$\hat{q(z)} = \sum^K _{i=1} \pi _i N (z; \mu_i , \sum_i)$$
  - $\pi_i$ | the prior probability of class $i$
  - $N (z; \mu_i , \sum_i)$ | the Gaussian distribution with mean $\mu_i$ and covariance $\sum_i$

- **Epistemic vs Aleatoric**
  - **Uncertainty**: Generally means being unsure about something. In modeling, it's helpful to split it into two types:
    ➔ **Aleatoric Uncertainty** | This type is due to inherent variability in the data. If a model trained on handwritten digits (like MNIST) encounters a digit that looks ambiguous (e.g., a digit that resembles both '1' and '7'), it exhibits `aleatoric uncertainty`.
    ➔ **Epistemic Uncertainty** | This arises from the model's lack of knowledge or uncertainty in its parameters. If the same model is given an image of clothing (something it has never seen before), it won't know how to classify this image, leading to `epistemic uncertainty`.
  - A softmax layer can predict probabilities between 0 and 1 for different classes and it can handle overlapping classes (showing `aleatoric uncertainty`) but struggles to reduce confidence when faced with unfamiliar data far from its training examples (showing `epistemic uncertainty`).

### Uncertain regions of the softmax
#### Valid OOD Region
- An OOD input must fall into to be correctly labelled as OOD.
    <img src="https://github.com/user-attachments/assets/aca96df3-5144-4e10-bbad-8e4f1f31cefe" width=90%>

- Uncertainty vector fields for each of our estimators formalize common intuition about the danger of **softmax extrapolations**.
    <img src="https://github.com/user-attachments/assets/c44ee683-00b0-4bd5-88e2-c2bdf00ec04a" width=90%>
    <img src="https://github.com/user-attachments/assets/93defffc-2789-4b7d-913d-e95cc3bdc4c9" width=90%>

- For $U_{max}$ and $U_{entropy}$, OOD data must fall closer to a decision boundary than $(1 − \epsilon) \%$ of the training distribution to be in the `valid OOD region`.
  <img src="https://github.com/user-attachments/assets/a1144589-940d-4f10-b89d-d131600bf2e3" width=90%>
  - These regions are formally defined for $U_{max}$ and $U_{entropy}$ for two classes
        <img src="https://github.com/user-attachments/assets/d3dc2bee-5c29-49f8-a880-169ae77c94f6" width=90%>

  - Making the assumption that final-layer features in the training data follow a mixture of Gaussians allows
analytical integration of class clusters in corollary below.
      <img src="https://github.com/user-attachments/assets/7d19b660-1538-49bf-93d4-4128d989ab96" width=90%>

  - The regions becomes more difficult for higher numbers of classes, since intersections between decision
boundary hyperplanes create curved `valid OOD regions`. 
  - However, we define an approximation of the $U_{max}$ `valid OOD region` for general numbers of classes using pairs of linear hyperplanes, offset from the decision boundary by a distance $α(w_i − w_j)$.
      <img src="https://github.com/user-attachments/assets/74cfcac8-8092-4b67-addd-09f5e9563393" width=90%>
      <img src="https://github.com/user-attachments/assets/801b40df-86aa-49ca-83ed-81922a519ec9" width=90%>

  - This linear approximation forms a subset of the exact `valid OOD region`, matching the exact `valid OOD region` for
K = 2.
      <img src="https://github.com/user-attachments/assets/8d0e057e-2f18-4f87-977e-39103450f448" width=90%>

  - Matching the exact `valid OOD region` at nearly any point at large magnitudes.
      <img src="https://github.com/user-attachments/assets/177b2076-f46d-40d7-af16-5afd98ee9ef2" width=90%>

  - Specifying the `valid OOD region` for $U_{density}$ is more straightforward for general classes, K ≥ 2, as per proposition below.

### Approximately Optimal Decision Boundary Structure
#### Optimal decision boundary structure
- Optimal structure
      <img src="https://github.com/user-attachments/assets/e0e5facc-00a8-488b-9e0c-dc0f4cf60c42" width=90%>

#### Empirical measurement of decision boundary structures in trained networks
- Analysing properties of final-layer weights in trained networks reveals three key properties of decision boundary structure.
  - 1) Bias values tend to be small. 2) Weight vectors all have similar magnitude. 3) Weight vectors are approximately evenly distributed, with $\cos \theta \approx \frac{-1}{K-1}$

  <img src="https://github.com/user-attachments/assets/48564f40-65a2-4de7-933a-3ba36fee3534" width=90%>

#### The Effect of Decision Boundary Structure on OOD Detection
- The volume of these regions is larger for the optimal structure. This is useful for OOD detection since it creates an increased opportunity for OOD data to fall into these valid regions.
      <img src="https://github.com/user-attachments/assets/9e9e4daa-2de1-462f-8b72-55c5612f9ce2" width=90%>

### Deep Networks Filter for Task-Specific Features
-  Neural networks typically learn to create decision boundaries that help separate different classes in the data.
- This structure plays a role in identifying data points that are outside the range of the training data (OOD data).
      <img src="https://github.com/user-attachments/assets/5345475b-c8b6-4207-bf9e-c637f70f7567" width=90%>

  - The network correctly identifies OOD data, showing that it maps these points away from the known classes.
  - Despite having a good decision boundary, this network maps OOD data to a region of low confidence in its predictions.

#### Features and layers
- **Final-layer features** | These are the activations produced by the last layer of a neural network, which summarize the information required for making predictions.
- **Convolutional filters** | These are small windows that slide over input features (patches of activations) and are designed to detect specific patterns (features) based on learned weights.
- **Response to OOD data**
  - OOD (Out-Of-Distribution) data doesn't usually contain the same distinguishing patterns that the network has been trained to recognize.
  - When OOD data is passed through the network: The final-layer activations tend to be lower in magnitude because the filters do not find the familiar patterns effectively.
  - When features are somewhat present, they appear in uncommon combinations compared to training data.
- **Quantifying activations**
  - The magnitude of final-layer activation is denoted as  || $z$ ||.
  - The 'familiarity' of the activation is measured using $\max_i \cos (\theta_i, z)$, where:
     - $\cos (\theta_i, z)$ measures the cosine similarity between the activation and known patterns in the training set (specific to class $i$).
  - **Histograms for analysis**: Figure 15 in the study shows histograms comparing the distributions of || $z$ ||  and $\max_i \cos (\theta_i, z)$: 
    - Training data typically shows higher values for both metrics.
    - OOD data shows reduced values, indicating that the activations are less consistent with learned features.


#### OOD Activations and Softmax
- The query explores how out-of-distribution (OOD) final-layer activations impact softmax confidence.
- The softmax function converts a vector of scores (logits) into probabilities:
    <img src='https://github.com/user-attachments/assets/31a64109-e39e-4079-958d-b832b1e6df62' width=70%>

  - || $z$ || acts similarly to a temperature parameter in the softmax scaling. 

  - Increasing || $z$ || (like increasing temperature T) will lower the maximum softmax confidence $U_{max}$ and the uncertainty $U_{entropy}$.
  - $\cos (\theta _{i, z})$ defines how aligned the activation is with the weight vector. 
    - Lower values can lead to lower softmax confidence, depending on the structure of weights in the softmax.
- Increasing || $z$ || lowers softmax confidence overall.
    <img src='https://github.com/user-attachments/assets/4986b7dc-ee3e-4252-aca4-9460884e94d8' width=80%>

- Merely decreasing does not guarantee lower softmax confidence. The relationship is complex and depends on the weight structure.
    <img src='https://github.com/user-attachments/assets/008440c6-c8a9-4f67-a541-c1b0d6479c52' width=80%>

- When the decision boundary structure is optimal (i.e., well-designed), lower $\max_i \cos (\theta_{i, z})$ correlates with lower softmax confidence, particularly for small numbers of classes (like 2 and 3).
    <img src='https://github.com/user-attachments/assets/1323ddf3-8c3c-4cea-9976-32c4f3812b67' width=90%>


#### Empirical Demonstration of the Filtering Effect
<img src='https://github.com/user-attachments/assets/50df00e9-06c7-434e-b2b3-240151c08a3b' width=90%>

- The neural network can be viewed as a filter that reacts more strongly to inputs it has seen in training.
  - When it receives an input, the softmax function evaluates how closely the input matches the trained examples.
- Softmax confidence scores are highest for inputs that are similar to those the network was trained on.
  - This means the network is designed to recognize and correctly classify known patterns.
- **Out-of-Distribution (OOD) Testing**
  - Typical benchmarks test how well the network recognizes inputs that differ from training data.
  - However, many tests only explore a very small portion of the possible input space for OOD testing.
- **Exhaustive OOD Testing**
  - The authors propose a more thorough approach: defining a small training set of MNIST digits (9x9 pixels, binarized).
  - They sampled around 100 million inputs uniformly from the entire input space (281 patterns).
- The examples that were most confidently recognized by the network still retained characteristics of the training data.
  - This indicates that the network is filtering inputs based on the training distribution.


### Limitation and Impact
- The findings improve our understanding of conventional uncertainty assessments in deep learning.
- Caution is necessary when using softmax confidence in real applications because:
  - The study does not address how well softmax probabilities are calibrated.
  - It does not evaluate effectiveness against adversarial attacks or in regression tasks.
- The research focuses only on a particular kind of OOD data, suggesting results may vary with other OOD examples.

### Conclusion
- The ability to capture epistemic uncertainty should not be viewed as a binary feature, but rather as a spectrum.
- Future research could explore how ensemble methods, Bayesian neural networks, and Monte Carlo Dropout might reinforce the biases discussed rather than provide new capabilities.
- There are two recommended approaches to address feature overlap in the final layer:
  - Modifying networks to promote bijective behavior.
  - Learning more diverse representations.
