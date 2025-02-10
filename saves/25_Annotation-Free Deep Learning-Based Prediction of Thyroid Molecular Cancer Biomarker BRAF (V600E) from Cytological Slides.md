## Annotation-Free Deep Learning-Based Prediction of Thyroid Molecular Cancer Biomarker BRAF (V600E) from Cytological Slides
- Authors: Wang, Ching-Wei and Muzakky, Hikam and Lee, Yu-Ching and Lin, Yi-Jia and Chao, Tai-Kuang
- Journal: International Journal of Molecular Sciences
- Year: 2023
- Link: https://www.mdpi.com/1422-0067/24/3/2521

### Abstract
- The proposed deep learning (DL) framework is evaluated on a dataset of 118 whole slide images. The results show that the proposed DL-based technique achieves an accuracy of 87%, a precision of 94%, a sensitivity of 91%, a specificity of 71% and a mean of sensitivity and specificity at 81% and outperformed three state-of-the-art deep learning approaches. 
- This study demonstrates the feasibility of DL-based prediction of critical molecular features in cytological slides, which not only
aid in accurate diagnosis but also provide useful information in guiding clinical decision-making
in patients with thyroid cancer.

### Dataset
- De-identified and digitized 118 WSIs, including 107 PTC cytologic slides (smear, Papanicolaou stained, n = 107) and 11 PTC cytologic slides (ThinPrep, Papanicolaou stained, n = 11) 
- All PTC were cytologically diagnosed, accompanied by cytologically confirmed by the two expert pathologists. 
- All patients underwent thyroidectomy within three months to confirm the presence of PTC, while immunohistochemistry (IHC) recorded positive or negative results for BRAF (V600E).
- All the stained slides were scanned using Leica AT Turbo (Leica, Germany) at 200× overall magnification (with a 20× objective lens). 
- The average slide dimensions are 77,338 × 37,285 pixels with physical size 51.13 × 23.21 mm<sup>2</sup>.
- The training model utilizes 79 Papanicolaou-stained WSIs (67%), and the remaining 39 Papanicolaoustained WSIs (33%) are used as an independent testing set for evaluation.

### Methods
- We examined three recently state-of-the-art DL models and construct a clustering constrained-attention multiple-instance learning (CLAM)-based model for classification BRAF (V600E) status of the individual patient using cytological slides. 
- In 2020, Tolkach et al. introduced a `NasNetLarge`-based model for Gleason pattern (GP) classification in prostate cancer patients with an overall accuracy of more than 98%. This research utilized detailed pixel-wise annotations by three expert pathologists to identify the tumor patches, and then the patchbased annotations that form from GP WSIs are utilized. `NasNetLarge` requires detailed
image annotations for fully supervised learning.
- In 2019, a weakly-supervised model, i.e., `MIL with Resnet34 + RNN`, was presented for classification of prostate cancer, basal cell carcinoma, and breast cancer metastases by Campanella et al., and the main strength of this approach is that it requires slide
labels only without annotating WSIs at the image level. The MIL strategy trains a deep learning network with rich tile-wise feature representations, aggregates the information across WSIs and makes a final diagnosis by RNN pooling-based mechanisms.
- In 2021, Lu et al. proposed an improved MIL-based technique, `CLAM`, that regards each slide as a collection of many patches or instances. Instead of utilizing RNN as an aggregator, `CLAM` adopts an attention-based pooling MIL formulation to tackle the stagnation of AUC limitation. 

### Proposed `CLAM`-Based Model
- Conventional CNNs consist of several levels of convolution nodes, pooling layers, and fully connected layers. We utilized `Resnet101` as a feature extractor on our modified `CLAM`-based network to transform the foreground patches into sets of low-dimensional feature representation. 
- This architecture applies residual blocks made of shortcut connections that perform identity mapping and add their outputs to the outputs of the stacked layers. 
- `Resnet101` comprises one max pooling layer followed by 48 stacks of residual blocks (99 convolutional layers), then ends with a fully connected layer and a softmax output layer.
- **Workflow of the proposed framework.**
  I. Each WSI is segmented into the foreground region of each slide
  II. Each WSI divides each slide into many smaller patches (for example, 256 × 256 pixels)
  III. Through feature extraction, all foreground patches are converted into sets of low-dimensional feature embeddings to be fed to the attention network.
  IV. Then, the attention network aggregates patch-wise evidence into slide-level representations, which are then used to create the diagnostic prediction.
  V. After the slide-wise representations are obtained, the attention network ranks each region in the slide, and an attention score is formed based on its relative importance to the slide-wise diagnosis.
  Next, Attention pooling weighs patches by their respective attention scores and summarizes patch-level features into slide-level representations. Consequently, strongly patched (denoted by red regions) and weakly patched (represented by blue regions)
  are representative samples to supervise the clustering process that separates positive and negative instances.
  VI. Heatmap visualization can be formed from the attention scores to identify ROIs and interpret the vital morphology used for diagnosis. 

<img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/66cb0975-80fc-48f2-afaa-082e5831cd91'>

- Initially, the first fully connected layer $W_1 \in \mathbb{R}^{512×1024}$ further squeezes each fixed patch-level representation $\mathbf{z}_k \in \mathbb{R}^{1024}$ to a 512-dimensional vector $\mathbf{h}_k = W_1 \mathbf{z}_k$.
- For the attention score of the k-th patch for the i-th class, denoted $a_{i,k}$, the slide-level representation aggregated per the attention score distribution for the i-th class, denoted $\mathbf{h}_{slide,i} = \sum$ a<sub>i,k</sub> h<sub>k</sub>.
- To further encourage the learning of class-specific features, we introduce an additional binary clustering objective during training. 
For each N classes, we put a fully connected layer after the first layer $W_1$. If we denote the weight of the clustering layer that corresponds to the i-th class as $W_{inst, 1} \in \mathbb{R}^{2 x 512}$, the cluster assignment scores predicted for the k-th patch, showed by $\mathbf{p}_{i,k} = W$ <sub>inst, i</sub> $\mathbf{h}_k$ .
- For the instance-level clustering task, N-class classification problem, neural network models output a vector of prediction scores s, where each entry in s corresponds to the prediction of the model for a single class made. Given the set of all possible ground-truth
labels Y = {1, 2, 3, . . . , N} and ground-truth label y ∈ Y , the multi-class SVM loss penalizes the classifier linearly in the difference between the prediction score for the ground-truth class and the highest prediction score for the remaining classes only if that difference is greater than a specified margin α. 
  - The smoothed variant adds a temperature scaling τ to the multi-class SVM loss, with which it has been shown to be infinitely differentiable with non-sparse gradients and suitable for the optimization of deep neural networks when the algorithm is implemented efficiently. The smooth SVM loss can be considered as a generalization of the widely used crossentropy classification loss for different choices of finite values for the margin and different temperature scaling.
    <img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/ff918645-93be-4cf4-b9cc-13118dfbab38' width=50%>

- The total loss for a given slide $L_{total}$ is the sum of both the slide-level classification loss $L_{slide}$ and the instance-level clustering loss $L_{patch}$ with optional scaling via scalar $c_1$ and $c_2$: $L_{total} = c_1 L_{slide} + c_2 L_{patch}.$
  - To compute$L_{slide}, s_{slide}$ is compared with the ground-truth slide-level label using the standard cross-entropy loss, and to compute $L_{patch}$, the instance-level clustering prediction scores $p_k$ for each sampled patch are compared against their corresponding pseudo-cluster labels using the binary smooth SVM loss.

### Result
- Results from the quantitative evaluation show that the modified model outperformed the three state-of-the-art benchmark methods, including `NASNetLarge`, `MIL with Resnet34 + RNN`, and the original `CLAM`.
- Quantitative evaluation for classification of BRAF (V600E) results in thyroid FNA and TP slides
    <img src='https://github.com/standing-o/Machine_Learning_Paper_Review/assets/57218700/d2fb8914-6eaa-4827-abaf-50a5ee70fe20' width=80%>
