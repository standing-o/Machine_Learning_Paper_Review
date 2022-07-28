## Lightgbm: A highly efficient gradient boosting decision tree
- Authors : Ke, Guolin, et al.
- Journal : Advances in neural information processing systems
- Year : 2017
- Link : https://proceedings.neurips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf

### Abstract
- GBDTs need to scan all the data instances to estimate the information gain of all possible split points, which is very time consuming.
➔ Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB). 
1. With GOSS, we exclude a significant proportion of data instances with small gradients, and only use the rest to estimate the information gain.
2. With EFB, we bundle mutually exclusive features (i.e., they rarely take nonzero values simultaneously), to reduce the number of features.

### Introduction
-  Conventional implementations of GBDT need to, for every feature, scan all the data instances to estimate the information gain of all the possible split points.
- Their computational complexities will be proportional to both the number of features and the number of instances. 

### Gradient-based One-Side Sampling (GOSS)
-  Instances with larger gradients (i.e., under-trained instances) will contribute more to the information gain.
➔ When down sampling the data instances, in order to retain the accuracy of information gain estimation, we should better keep those instances with large gradients, and only randomly drop those instances with small gradients.

#### Algorithm Description
-  The gradient for each data instance in GBDT provides us with useful information for data sampling.
➔ If an instance is associated with a small gradient, the training error for this instance is small and it is already well-trained.
➔ discard those data instances with small gradients.
- However, the data distribution will be changed by doing so, which will hurt the accuracy of the learned model. 🠊 **GOSS**
- GOSS keeps all the instances with large gradients and performs random sampling on the instances with small gradients. In order to compensate the influence to the data distribution, when computing the information gain, GOSS introduces a constant multiplier for the data instances with small gradients.
  <img src='https://user-images.githubusercontent.com/57218700/160903658-50c8aac5-2449-4082-b2d6-4c0a92dae6ce.png' width=45%>
1. sorts the data instances according to the absolute value of their gradients
2. selects the top a×100% instances
3. it randomly samples b×100% instances from the rest of the data.
4. amplifies the sampled data with small gradients by a constant (1−a)/b when calculating the information gain.

#### Theoretical Analysis
- {g<sub>1</sub>, · · · , g<sub>n</sub>} : the negative gradients of the loss function with respect to the output of the model
- The decision tree model splits each node at the most informative feature (with the largest information gain).
  ➔ For GBDT, the information gain is usually measured by the variance after splitting:
  <img src='https://user-images.githubusercontent.com/57218700/160905401-60235091-751e-431e-998d-8ab79061eff7.png' width=80%>
- For feature j, the decision tree algorithm selects d<sup>∗</sup><sub>j</sub> = argmax<sub>d</sub>V<sub>j</sub>(d) and calculates the largest gain V<sub>j</sub>(d<sup>∗</sup><sub>j</sub>).
  - A :  keep the top-a × 100% instances with the larger gradients
  - B : A<sup>c</sup> consisting (1 − a) × 100% instances with smaller gradients
  <img src='https://user-images.githubusercontent.com/57218700/160905649-b7412f7e-9537-49ea-810d-5c9c230abaf8.png' width=80%>
- GOSS will not lose much training accuracy and will outperform random sampling:
  <img src='https://user-images.githubusercontent.com/57218700/160905827-6b2cbf7a-1a8a-4a2e-beb8-33bd417cf2fe.png' width=80%>
- The generalization error with GOSS will be close to that calculated by using the full data instances if the GOSS approximation is accurate. On the other hand, sampling will increase the diversity of the base learners, which potentially help to improve the generalization performance.
(The larger n, and the more evenly the instances are split into the left and right leaf through the split point, the smallest the approximation error becomes.)

###  Exclusive Feature Bundling
- High-dimensional data are usually very sparse. 
- Designing a nearly lossless approach to reduce the number of features.  Specifically, in a sparse feature space, many features are mutually exclusive, i.e., they never take nonzero values simultaneously.
  <img src='https://user-images.githubusercontent.com/57218700/160906287-c439ee79-aa7c-400d-b4ec-d72a5f8aa40c.png' width=80%>
- we first reduce the optimal bundling problem to the graph coloring problem by taking features as vertices and adding edges for every two features if they are not mutually exclusive, then we use a greedy algorithm which can produce reasonably good results for graph coloring to produce the bundles.
- There are usually quite a few features, although not 100% mutually exclusive, also rarely take nonzero values simultaneously. 
- If our algorithm can allow a small fraction of conflicts, we can get an even smaller number of feature bundles and further improve the computational efficiency.
➔ Random polluting
  <img src='https://user-images.githubusercontent.com/57218700/160906730-c4f19420-de84-48a7-ad59-716138b8ebc1.png' width=80%>
3-1. Construct a graph with weighted edges, whose weights correspond to the total conflicts between features.
3-2. Sort the features by their degrees in the graph in the descending order
3-3. Check each feature in the ordered list, and either assign it to an existing bundle with a small conflict, or create a new bundle. 

4-1. The values of the original features can be identified from the feature bundles. 
4-2. We can construct a feature bundle by letting exclusive features reside in different bins. This can be done by adding offsets to the original values of the features.

### Experiment
  <img src='https://user-images.githubusercontent.com/57218700/160907333-defaeae6-53f8-4798-9414-d122bad2c255.png' width=70%>
  <img src='https://user-images.githubusercontent.com/57218700/160907380-2b719426-9124-4344-9295-e5b3617d699a.png' width=70%>

➔ ``LightGBM`` is the fastest while maintaining almost the same accuracy as baselines.