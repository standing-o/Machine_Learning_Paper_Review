## Weakly supervised instance learning for thyroid malignancy prediction from whole slide cytopathology images
- Authors: Dov et al.
- Journal: Medical image analysis
- Year: 2021
- Link: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7726041/


### Abstract
- We consider machine-learning-based thyroid-malignancy prediction from cytopathology wholeslide images (WSI). 
  - Multiple instance learning (MIL) approaches, typically used for the analysis of WSIs, divide the image (bag) into patches (instances), which are used to predict a single bag-level label.
  - These approaches perform poorly in cytopathology slides due to a unique bag structure: sparsely located informative instances with varying characteristics of abnormality. 
- We address these challenges by considering multiple types of labels: bag-level malignancy and ordered diagnostic scores, as well as instance-level informativeness and abnormality labels.
  - We study their contribution beyond the MIL setting by proposing a `maximum likelihood estimation (MLE)` framework, from which we derive a two-stage deep-learning-based algorithm.
  -  The algorithm identifies informative instances and assigns them local malignancy scores that are incorporated into a global malignancy prediction. 
  - We derive a lower bound of the `MLE`, leading to an improved training strategy based on weak supervision, that we motivate through statistical analysis. 
  - The lower bound further allows us to extend the proposed algorithm to simultaneously predict multiple bag and instance-level labels from a single output of a neural network.


### Introduction
- Each sample comprises a whole slide image (WSI) scanned at a typical resolution of  40,000×25,000 pixels, as well as the postoperative histopathology diagnosis, that is considered the ground truth in this study. 
- The goal in this paper is to predict the ground truth malignancy label from the WSIs. 
  - Each sample also includes the diagnostic score assigned to the slide by a cytopathologist according to the Bethesda System (TBS).
- The vast majority of previous studies consider the analysis of histopathology biopsies, which comprise whole tissues covering large regions of the WSI. 
  - In contrast, FNABs (cytopathology biopsies), as we consider in this paper, contain separate, sparsely located groups of follicular cells, which are informative for diagnosis. 
  - The diagnosis of the FNABs, performed by a trained (cyto-) pathologist, includes the identification of follicular groups
followed by evaluation of their characteristics. 
- A WSI containing even as few as six follicular groups with a size of tens of pixels, which corresponds to less than 0.01% of the area of the slide, is considered sufficient for diagnosis.
- FNABs are considered significantly more challenging for diagnosis by pathologists due to their sparsity, and since in many cases, the characteristics of individual follicular groups are subject to subjective interpretation.
- Here, we investigate how only a few local, instance-level, labels can improve prediction beyond the classical MIL setting, where only a global label at the WSI/bag level is available. 
  - This is important in medical applications, where the collection of local labels requires significant manual effort, raising the question of what kind of labels to collect and what is the expertise required for their collection.
  -  In this context, we note the closely related task of region-of-interest detection, studied extensively for object detection.
  - However, here we are not strictly concerned with the accurate estimation of bounding boxes of individual instances, a difficult challenge in the case of cytopathology, as our goal is to predict the global per-slide label.
- The second gap is related to the structure of the bag in MIL in terms of the prevalence of positive instances (PPI) in a bag, which is typically not taken into account. 
  - In our context, PPI measures the fraction of the positive instances (in a positive WSI), i.e., those containing follicular groups with clear characteristic of malignancy. In contrast, a positive bag also contains non-malignant follicular groups, as well as uninformative instances. 
  - The uninformative instances constitute the vast majority of the scan, mainly containing red blood cells, considered in our case as background. 
  - This forms a unique bag structure of low PPI. On the other hand, once background instances are filtered out, as we propose in our approach, the bags composed of only informative instances have a high PPI structure; namely, the follicular groups are consistent in their indication of malignancy to a certain level, which we explore in this paper.
- The third gap is the question of how to use multiple labels for improving classification. To this end, we consider the joint prediction of the malignancy labels, the TBS categories, and the local abnormality labels. Since both TBS categories and the local labels correspond to the increasing probability of malignancy, we consider their joint prediction using ordinal
regression.
  - The joint prediction is motivated by the observation that the local labels, as well as TBS categories, are a consistent proxy for the probability of malignancy, and so their joint prediction induces cross-regularization.
- We propose a `maximum likelihood estimation (MLE)` framework for classification in the mixed setting, where multiple global and local labels are available for training. While in classical MIL, informative instances are implicitly identified, the `MLE` framework allows explicit identification of them using the local labels, which we show to be especially useful in the low-PPI setting. We further derive a lower bound of the `MLE`, which corresponds to a weakly supervised training strategy, in which the global labels are propagated to the instance level and used as noisy local labels. 
  - Statistical analysis and experiments on synthetic data show that this training strategy is particularly useful for high PPI bags obtained by filtering out the background instances. 
  - From the lower bound of the `MLE`, we derive the algorithm for malignancy prediction, that is based on deep-learning and comprises two stages. The algorithm identifies instances containing groups of follicular cells and incorporates local decisions based on the informative regions into the global slide-level prediction. 
  - The lower bound of the `MLE` further allows us to investigate the simultaneous prediction of the global malignancy and the TBS category scores, as well as the local abnormality scores. 
  - Specifically, using ordinal regression, we extend our framework to jointly predict these labels from a single output of a neural network. 
  - We further show that the proposed ordinal regression approach allows application of the proposed algorithm to augment cytopathologist decisions.

### Problem Formulation
- $\mathbb{X} = (X_l)$: a set of WSIs, where $X_l = (\mathbf{x}_{l,m})$ is the set of $M_l$ instances in the $l$-th WSI.
  - The $m$-th instance $\mathbf{x}_{l,m} \in \mathbb{R}^{w \times h \times 3}$ is a patch from an RGB digital scan.
- $\mathbb{Y} = (\mathbb{Y}_l) \in$ {0, 1}, where 0 and 1 correspond to **benign** and **malignant cases**, respectively.
- The goal is to predict thyroid malignancy $\hat{\mathbf{Y}}_l$.
- Consider the set $\mathbb{S} = (\mathbf{S}_l)$, where $\mathbf{S}_l \in$ {2,3,4,5,6} is the TBS category assigned to a WSI by a pathologist.
- Local labels $\mathbb{U} = (\mathbf{U}_l)$
  - $u_{l,m} = 1$ if instance $\mathbf{x}_{l,m}$ contains a group of follicular cells
  - $u_{l,m} = 0$ otherwise.
  - Our dataset includes 4494 such informative instances, manually selected (by a trained pathologist) from 142 WSIs.
  - These local labels are exploited in the proposed framework for the improved identification of the informative instances.  The instances containing follicular groups are further labeled according to their abnormality, forming the set $\mathbb{V} = (v_{l,m}), v_{l,m} \in$ {0,1,2}. (normal, atypical and malignant).
- While in the classical MIL setting, only the set of binary malignancy label $\mathbb{Y}$ is available, we explore in this paper the contribution of the additional label sets $\mathbb{S}, \mathbb{U}$ and $\mathbb{V}$ for the improved prediction of thyroid malignancy.

### Proposed framework for thyroid malignancy prediction
#### MLE Formulation
- Likelihood over dataset:
  - We drop the right most term by assuming a uniform distribution over the WSIs, and further assume the
following conditional distribution on the label $Y_l$.
    <img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/b54d12d0-d2ae-4871-8d85-1c6156cd0bcc' width=50%>
    <img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/a9e637de-2e2b-4a08-a379-0d62dd231526' width=50%>
  - $g_\theta (x_{l,m}) \in \mathbb{R}$ is the output of a neural network and $\sigma$ is a sigmoid function.
  - $\tilde{M} \triangleq \sum_m u_{l,m}$.
  - This statistical model suggests the estimation of $Y_l$ from an average of local, instance-level estimates $g_\theta (x_{l,m})$, weighted by $u_{l,m}$ according to the level of their informativeness.
  - Substituting these leads to the following log likelihood expression:
    <img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/a45ff2f5-c83b-44a1-8444-07e14d004e15' width=50%>
  - Maximizing the first two terms on the right hand side is equivalent to minimizing the binary cross entropy (BCE) loss in the MIL setting.
- To that end, we propose to greedily maximize above equation in two steps: we use another neural network $r_\phi (\cdot)$, trained using the last term and the local labels to estimate the informativeness of instances $u_{l,m}$, and predict slide-level malignancy from the informative instances.
  - Once trained, the network for the identification of informative instances $r_\phi (\cdot)$ is applied to the WSIs, and the estimated weights $u_{l,m}$ are set to 1 for the $\tilde{M}$ most informative instances, and zero otherwise;
  - We fix $\tilde{M} = 1000$ instances, a value that balances the tradeoff between having a sufficient amount of training data to predict malignancy and using instances that with high probability are informative.
- Once the informative instances are identified, we turn to the prediction of malignancy from the first two terms.
  - Since $\sum_m u_{l,m} / \tilde{M} = 1$,
    <img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/3e9cf0a7-adf9-4718-be19-9f808730cecf' width=50%>


  - Using Jensen's inequality, we get the lower bound:
    <img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/af2bf5ce-6542-4a43-b2db-6139c578d198' width=50%>
- The lower bound implies the global labels $(Y_l)$ are assumed to hold locally, i.e., separately for each instance.
  - We propose to train the $g_\theta (\cdot)$ and consider $g_\theta (\mathbf{x}_{l,m})$ as local, instance-leve, predictions of thyroid malignancy, which are averaged into a global slide-level prediction:
    <img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/7113a12a-0c87-484f-916b-8c5014905114' width=30%>

  - The predicted slide-level thyroid malignancy $\hat{Y_l}$ is:
    <img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/06984d79-b2f3-459e-af21-2010fca8b5b3' width=30%>

      - where $\beta$ is a threshold value.



#### Analysis of the lower bound in the high-PPI setting
- The extent to which the assumption that the global label holds locally and separately for each instance is directly related to the bag structure. 
  - This assumption holds perfectly in the extreme case of PPI = 1, i.e., that all instances are malignant in a malignant WSI and all of them are benign in a benign WSI. 
  - Yet, PPI smaller than 1 corresponds to a weakly supervised setting where instances are paired with noisy labels. 
- We analyze the utility of the lower bound for MIL in the high PPI setting. 
  - We note that the PPI of the bags is indeed high once the uninformative labels were filtered out, as we show by the analysis of the abnormality labels $v_{l,m}$.
  - We analyze logit $(Y_l = 1 | X_l)$, where $logit(\cdot) \triangleq \log(\frac{P(\cdot)}{1 - P(\cdot)})$. The following proposition shows that $f_\theta (X_l)$ is related directly to logit $(Y_l = 1 | \mathbf{x}_{l,m})$.

    <img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/dbfbc146-bdd8-4e7d-9a97-b0c67e44f066' width=80%>
    <img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/21bcdb7c-23e7-46d8-bfc5-f9afb4cd14d7' width=80%>

    - C is a constant and $\tilde{M}$ is the number of the informative instances.
    - This implies that making a prediction by comparing $f_\theta (X_l)$ to a threshold value $\beta$ is equivalent to comparing the estimated logit function to the threshold $\gamma \triangleq \tilde{M} \beta + C$.
    - The logit function is directly related to the likelihood ratio test. Using Bayes Rule:
      - $logit (Y_l = 1 | X_l) = \log \Lambda + P (Y = 0) / P (Y = 1)$, where $\Lambda$ is the likelihood ratio defined as $\Lambda \triangleq P(X_l | Y_l = 1) / P(X_l | Y_l = 0).$
      - This implies that thresholding $f_\theta (X_l)$ is equivalent to applying the likelihood ratio test, widely used for hypothesis testing.
    - The lower bound inequalities above indeed suggests to directly predict the global label from each instance separately. 
      - The higher the PPI is, the lower is the noise level in the the labels used to predict $P(Y_l = 1 | \mathbf{x}_{l,m})$ and, according to the proposition, the better is the global prediction of $P(Y_l = 1 | X_l)$.
      - The network is optimized to predict the global label from the multiple instances, and there is no guarantee on the quality of predictions of individual instances.


#### Simultaneous prediction of multiple global and local label
- We now consider prediction of the TBS categories S and the local abnormality scores V, using the likelihood over the full dataset $P(X, Y, U, S, V)$. 
  -  To make the computation of the likelihood tractable, we assume that $P(Y ∣ X,U), P(S ∣ X,U)$ and $P(V ∣ X,U)$ are independent.
  - The straightforward approach under this assumption is adding two cross entropy loss terms to predict the labels S and V, which leads to a standard multi-label scenario. 
    - However, this does not encode the strong relation between Y, S and V, in the sense that all indicate various abnormality (malignancy) levels.
  - We therefore propose to encode these relations into the architecture of the neural network $g_θ (⋅)$.
    - Specifically, we take advantage of the ordinal nature of S and V, where higher values of the labels indicate a higher probability of malignancy, and propose an ordinal regression framework to predict all three types of labels from a single output of the network. 
    - In what follows, we consider for simplicity only the prediction of the global TBS category S. 
    - Extending the framework to predict the local labels V is straightforward, as our lower bound formulation treats local and global labels in the same manner.

- We propose to predict the TBS category by comparing the output of the network $f_θ(X_l)$ to threshold values $β_0 < β_1 < β_2 < β_3 ∈ \mathbb{R}$.
    <img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/cf9cc55e-facc-41ae-a284-a89672f87311' width=50%>

- The proposed framework for ordinal regression is inspired by the proportional odds model, also termed the cumulative link model. 
  - The original model suggests a relationship between $f_\theta (X_l)$, the threshold $\beta_n$ and the cumulative probability $P(S_1 - 2 \leq n)$, i.e., $logit (S_l - 2 \leq n) = \beta_n - f_\theta (X_l) $

- The proportional odds model imposes order between different TBS by linking them to $f_\theta (X_l)$ so that higher values of $f_\theta (X_l)$ correspond to higher TBS categories. 
  - Recalling that the logit function is a monotone mapping of a probability function into the real line, values of $f_\theta (X_l)$ that are significantly smaller thatn $\beta_n$ correspond to high probability that the TBS category is smaller than n+2.

- Estimating $P (S_l -2 > n)$ rather than $P(S_l - 2 \leq n)$, which gives:
  <img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/c31515d9-6f08-49c3-bed2-261b818aed70' width=50%>

- We note that this deviation is not necessary for the prediction of TBS, yet it allows combining the predictions of the thyroid malignancy and the TBS category in an elegant and interpretable manner. 
  - We observe that the right term in the last equation is the sigmoid function $\sigma (f_\theta (X_l) - \beta_n).$
  - We can train the network to predict $P(S_l - 2 > l)$:
  <img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/a8518b50-db2a-4130-95e6-b1fcadbca3d4' width=50%>

    - where $S^n _l = \mathbb{I} (S_l -2>n)$ and $\mathbb{I} (\cdot)$ is the indicator function. 
- Maximizing log $\mathcal{L}^S$ is equivalent to minimizing 4 BCE loss terms with the labels $S^n _l, n \in (0,1,2,3)$, whose explicit relation to TBS is represented as below.
  <img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/e53414cc-bc13-44a5-a0ea-8c40217b7f2e' width=50%>

- The lower bound also allows us to extend this framework to predict the local abnormality score, which we denote by log $\mathcal{L}^V$, similar to considering two additional thresholds, $\gamma_0, \gamma_1$ and two corresponding BCE loss terms.
- log $\mathcal{L}^Y$ can be considered a special case of ordinal regression with a single fixed threshold value of 0.
  - The total loss function simultaneously optimizes the parameters $\theta$ of the network $g_\theta (\cdot)$ according to 7 classification tasks, corresponding to threshold values $0, \beta_0, \beta_1, \beta_2, \beta_3, \gamma_0, \gamma_1$.
    - The threshold values are learned along with the parameters of the networks, via stochastic gradient descent. While the training procedure does not guarantee the correct order of  $\beta_0 < \beta_1 < \beta_2 < \beta_3$, we have found in our experiments that this order is indeed preserved.
- In some cases, the term of the loss function corresponding to the prediction of malignancy may conflict with that of the TBS category or the local label.
  - ex. Consider a malignant case (Y_l = 1) with TBS category 3 assigned by a pathologist. The term of the loss, in this case, which corresponds to TBS penalize high values of $f_θ (X_l)$ whereas the term corresponding to malignancy encourages them. 
  - We therefore interpret the joint estimation of TBS category, the local labels, and malignancy as a cross-regularization
scheme. Given two scans with the same TBS but different final pathology, the network is trained to provide higher prediction values for the malignant case. 
  - Likewise, in the case of two scans with the same pathology but different local labels, the prediction value of the scan
with the higher abnormality score is expected to be higher.
- Thus, the network adopts properties of the Bethesda system and the abnormality scores, such that the higher the prediction value $f_θ (X_l)$ the higher is the probability of malignancy. 
  - Yet the network is not strictly restricted to the Bethesda system and the local labels, so it can learn to provide better
predictions.


#### Identification of the informative instances
- We predict the informativeness of the instances using a second neural network $r_\phi (x_{l,m})$, optimized according to:

  <img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/83d28ddd-be3b-405b-ad43-a97d36f56fd6' width=50%>

 - Training the network requires sufficiently many labeled examples, the collection of which was done manually by an expert pathologist through an exhaustive examination of the slides. To make the labeling effort efficient, the cytopathologist only marked positive examples of instances containing follicular groups $(u_{l,m} = 1)$.
   - We further observed in our experiments that instances sampled randomly from the whole slide mostly contain background. Therefore, to train the network $r_\phi (\mathbf{x}_{l,m})$, we assume that $u_{l,m} = 0$ for all instances in the last equation except those manually identified as informative.
 - More specifically, we propose the following design of training batches. We use batches comprising an equal number of positive and negative examples to overcome the class imbalance. 
   - As positive examples, we take follicular groups sampled uniformly at random from the set of the labeled instances, i.e., for which $u_{l,m} = 1$.
   - Negative examples are obtained by sampling uniformly at random instances from the whole slide. Since in some cases informative instances can be randomly sampled and wrongly considered uninformative, the proposed training strategy can be considered weakly supervised with noisy negative labels.

- **Complete log likelihood function**:
    <img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/74645a48-1a10-4d54-9765-9096f057861b' width=50%>
  - where log $\mathcal{L}_{total}$ is the lower bound on the full log-likelihood of the probabilistic model we assume for $X, Y, U, S, V$. Note that one can further weight the different likelihood components if desired.


### Experiments
#### PPI analysis on synthetic data
- We compare the proposed weakly supervised training strategy derived from the lower bound to the following MIL algorithms: noisy-or MIL, where the global prediction value is the highest local prediction, the attention-based MIL, and average-pooling MIL.
  - As expected, the performance of the methods is improved with the increase of the PPI since there are more positive instances indicating that a bag is positive. Noisy-or MIL provides inferior performance compared to the other methods for most PPI values, and only for low PPIs it performs comparably. 
  - This is because the global decision is based only on a single instance, so this approach does not benefit from the multiple positive instances present in the slides when the PPI is high. 
  - This method was excluded from the following experiments due to poor performance.
- As expected, the performance of the methods is improved with the increase of the PPI since there are more positive instances indicating that a bag is positive. 
- Noisy-or MIL provides inferior performance compared to the other methods for most PPI values, and only for low
PPIs it performs comparably. 
  - This is because the global decision is based only on a single instance, so this approach does not benefit from the multiple positive instances present in the slides when the PPI is high. This method was excluded from the following experiments due
to poor performance.


#### Thyroid malignancy prediction
- **Experimental Setting**
  - To evaluate the proposed algorithm, we performed a 5-fold cross-validation procedure, splitting the 908 scans by 60%, 20%, 20% for training, validation, and testing, respectively, such that a test scan is never seen during training. 
  - We use instances of size 128 × 128 pixels. This size is large enough to capture large groups of follicular cells
while allowing the use of sufficiently many instances in each minibatch. 
  - Both the network for the identification of the informative instances $r_\phi (⋅)$ and the network for the prediction of
malignancy $g_θ (⋅)$ are based on the small and the fast converging VGG11 architecture.

- **Identification of instances containing follicular groups**
  - We evaluated the performance of the network for the identification of informative instances $r_\phi (⋅)$ using the annotated 142 WSIs, obtaining a test AUC of 0.985.

- **PPI Analysis**
  - While the large number of background instances pose low PPI, filtering them out as a preprocessing step significantly changes the PPI in the bag. To shed light on the structure of the bag, restricted to the subset of the informative instances.
    - Distribution of local abnormality labels. (Top) Distribution in malignant slides $(Y_l= 1)$. (Bottom) Distribution in benign slides $(Y_l= 0)$.
    <img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/4b173b18-becf-469d-8410-561a1f1a30f1' width=40%>

- Histogram of predictions for instances taken from a single slide. High prediction values correspond to high probabilities that an instance contain follicular groups.
    <img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/ab2e9328-0432-49df-998f-3a9f41e55c86' width=50%>





